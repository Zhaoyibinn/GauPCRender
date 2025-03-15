import numpy as np 
import torch 
import torch.nn as nn 
import cv2 
from torchvision.transforms import Resize
import time 
import lpips
lpips_loss_fn = lpips.LPIPS(net='alex').cuda()


from transform_3d import *
from scene import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from simple_knn._C import distCUDA2
from utils.sh_utils import RGB2SH
from pointMLP import pointMLPEncoderBase6
from model_util import MlpConv

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 1.0  # Assuming the pixel values are normalized between 0 and 1
    psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr_value

def inverse_sigmoid(x):
    return np.log(x/(1-x))

def compute_rotation_matrix_cuda(u, v):
    k = torch.cross(u, v)
    k_norm = torch.norm(k, dim=-1, keepdim=True)
    k = k / k_norm
    
    cos_theta = torch.einsum('kij,kij->ki', u, v) / (torch.norm(u, dim=-1) * torch.norm(v, dim=-1))
    sin_theta = torch.norm(torch.cross(u, v), dim=-1) / (torch.norm(u, dim=-1) * torch.norm(v, dim=-1))
    
    K = torch.zeros((u.shape[0], u.shape[1], 3, 3)).float().cuda()
    K[:, :, 0, 1] = -k[:, :, 2]
    K[:, :, 0, 2] =  k[:, :, 1]
    K[:, :, 1, 0] =  k[:, :, 2]
    K[:, :, 1, 2] = -k[:, :, 0]
    K[:, :, 2, 0] = -k[:, :, 1]
    K[:, :, 2, 1] =  k[:, :, 0]
    
    I = torch.eye(3).float().cuda()
    I = torch.tile(I, (u.shape[0], u.shape[1], 1, 1))
    
    sin_theta = sin_theta.unsqueeze(-1).unsqueeze(-1)
    cos_theta = cos_theta.unsqueeze(-1).unsqueeze(-1)
    

    R = I + sin_theta * K + (1 - cos_theta) * torch.einsum('...ij,...jk->...ik', K, K)
    
    return R

class MyGaussionMid:
    def __init__(self, cmd_args):
        self.cmd_args = cmd_args
        self.gaussion = GaussianModel(self.cmd_args.args_2dgs.sh_degree)
        self.bg_color = [1, 1, 1] if self.cmd_args.args_2dgs.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")

    def set_args(self, xyz, rgb, o, sxyz, q, sh=None):
        # print(xyz.shape, rgb.shape, o.shape, sxyz.shape, q.shape, sh.shape)
        self.gaussion._xyz = xyz.float()
        self.gaussion._features_dc = torch.unsqueeze(rgb, 1).float()
        self.gaussion._opacity = o.float()
        self.gaussion._scaling = sxyz.float()
        self.gaussion._rotation = q.float()
        if sh is not None:
            self.gaussion._features_rest = torch.reshape(sh, [sh.shape[0], 3, (self.cmd_args.args_2dgs.sh_degree+1)**2-1]).transpose(1, 2).float()
    
    def render(self, viewpoint_cam):
        return render(viewpoint_cam, self.gaussion, self.cmd_args.pp_2dgs, self.background)

def get_edge_mask(gt_image):
    im = gt_image.detach().cpu().numpy()
    # print(im.shape)
    im = np.transpose(im, [1, 2, 0])
    im = im * 255
    im = im.astype(np.uint8)
    oim = im
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(im, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edge = cv2.dilate(edge, kernel, iterations=1)
    edge_mask = torch.from_numpy(edge > 125).cuda().float()
    # edge = edge_mask.detach().cpu().numpy() * 255
    # edge = edge.astype(np.uint8)
    # red = np.zeros_like(oim)
    # red[:,:,2] = edge
    # edge_img = cv2.addWeighted(oim, 0.5, red, 0.5, 0)
    # cv2.imwrite('trash/test_gt.png', oim)
    # cv2.imwrite('trash/test_edge.png', edge)
    # cv2.imwrite('trash/test_edge_img.png', edge_img)
    return edge_mask.unsqueeze(0)

class GauPCRender(nn.Module):
    def __init__(self, cmd_args, patch=False):
        super().__init__()
        self.cmd_args = cmd_args
        self.op_2dgs = self.cmd_args.op_2dgs
        self.split_n = self.cmd_args.split_number
        self.sh_degree = self.cmd_args.sh
        self.patch = patch
        self.sh_n = (self.cmd_args.sh+1)**2 * 3 - 3
        self.encoder = pointMLPEncoderBase6(feature_channel=8)

        self.decoder_xyz    = MlpConv(512+128+11, [512, 512, 512, 256, 128,         3*self.split_n])
        self.decoder_rgb    = MlpConv(512+128+11, [512, 512, 512, 256, 128,         3*self.split_n])
        self.decoder_o      = MlpConv(512+128+11, [512, 512, 512, 256, 128,         1*self.split_n])
        self.decoder_normal = MlpConv(512+128+11, [512, 512, 512, 256, 128,         3*self.split_n])
        self.decoder_angle  = MlpConv(512+128+11, [512, 512, 512, 256, 128,         1*self.split_n])
        self.decoder_scales = MlpConv(512+128+11, [512, 512, 512, 256, 128,         2*self.split_n])
        if self.sh_n > 0:
            self.decoder_sh     = MlpConv(512+128+11, [512, 512, 512, 256, 128, self.sh_n*self.split_n])

        self.gaussion = MyGaussionMid(self.cmd_args)
        self.mseloss = nn.MSELoss()

    def normlize(self, q):
        l = torch.sqrt(torch.sum(q**2, -1, keepdim=True))+1e-8
        q = q/l
        return q

    def forward(self, data):
        input_xyz, input_rgb, input_normal = data[:3]

        # with open('trash/log.txt', 'a') as f:
        #     f.write(f'{input_xyz.shape[1]:7d}\n')
        # print(input_xyz.shape)

        if self.patch:
            maxv = torch.max(torch.abs(input_xyz), dim=1).values
            maxv = torch.max(maxv,dim=1).values 
            input_xyz = input_xyz / maxv.unsqueeze(1).unsqueeze(2)
            # print(torch.mean(torch.max(input_xyz, dim=1).values, 0))
            # print(torch.mean(torch.min(input_xyz, dim=1).values, 0))

        xyz = input_xyz
        rgb = RGB2SH(input_rgb/255)
        normal = input_normal
        B, N = xyz.shape[:2]
        scale = []
        for i in range(B):        
            dist2 = torch.clamp_min(distCUDA2(xyz[i].float()), 0.0000001)
            _scale = torch.sqrt(dist2)[...,None].repeat(1, 2)
            scale.append(_scale)
        scale = torch.stack(scale, 0)

        xyz = xyz.contiguous().float()
        rgb = rgb.contiguous().float()
        normal = normal.contiguous().float()
        scale = scale.contiguous().float()


        feature = self.encoder(xyz.permute([0, 2, 1]), torch.cat([rgb, normal, scale], -1).permute([0, 2, 1]))
        feature = torch.cat([xyz.permute([0, 2, 1]), rgb.permute([0, 2, 1]), normal.permute([0, 2, 1]), scale.permute([0, 2, 1]), feature], 1)
        p_xyz    = self.decoder_xyz   (feature).permute([0, 2, 1]).reshape([B, N, self.split_n, 3])
        p_rgb    = self.decoder_rgb   (feature).permute([0, 2, 1]).reshape([B, N, self.split_n, 3])
        p_o      = self.decoder_o     (feature).permute([0, 2, 1]).reshape([B, N, self.split_n, 1])
        p_normal = self.decoder_normal(feature).permute([0, 2, 1]).reshape([B, N, self.split_n, 3])
        p_angle  = self.decoder_angle (feature).permute([0, 2, 1]).reshape([B, N, self.split_n, 1])
        p_scales = self.decoder_scales(feature).permute([0, 2, 1]).reshape([B, N, self.split_n, 2])
        if self.sh_n > 0:
            p_sh     = self.decoder_sh    (feature).permute([0, 2, 1]).reshape([B, N, self.split_n, self.sh_n])

        s = 1
        ss = 0.01
        p_xyz    = (        s * torch.tanh   (ss * p_xyz   ) + xyz     .unsqueeze(2)).reshape([B, N*self.split_n, 3]) 
        p_rgb    = (        s * torch.tanh   (ss * p_rgb   ) + rgb     .unsqueeze(2)).reshape([B, N*self.split_n, 3]) 
        p_o      = (                          ss * p_o       + inverse_sigmoid(0.99)).reshape([B, N*self.split_n, 1])
        p_normal = (        s * torch.tanh   (ss * p_normal) + normal  .unsqueeze(2)).reshape([B, N*self.split_n, 3]) 
        p_angle  = (2 * np.pi * torch.sigmoid(ss * p_angle )                        ).reshape([B, N*self.split_n, 1])
        p_scales = (        s * torch.tanh   (ss * p_scales) + scale   .unsqueeze(2)).reshape([B, N*self.split_n, 2])

        if self.sh_n > 0:
            p_sh     = (        s * torch.tanh   (ss * p_sh    )                        ).reshape([B, N*self.split_n, self.sh_n])
        else:
            p_sh = None

        p_scales = torch.abs(p_scales)
        p_normal = self.normlize(p_normal)

        hori_normals = torch.zeros_like(p_normal).float().cuda()
        hori_normals[:, :, 2] = 1
        _rms_1 = compute_rotation_matrix_cuda(hori_normals, p_normal)
        _rms_1 = torch.permute(_rms_1, [0, 1, 3, 2])

        _normals = p_normal * p_angle
        _rms_2 = axis_angle_to_matrix(_normals)
        _rms_2 = torch.permute(_rms_2, [0, 1, 3, 2])
        _rms = torch.matmul(_rms_1, _rms_2)
        _rms = torch.permute(_rms, [0, 1, 3, 2])
        p_q = matrix_to_quaternion(_rms)

        if self.patch:
            p_xyz = p_xyz * maxv.unsqueeze(1).unsqueeze(2)
            p_scales = p_scales * maxv.unsqueeze(1).unsqueeze(2)

        p_scales = torch.log(p_scales + 1e-6)

        # if cmd_args.show_shape_log:
        #     print(p_xyz.shape, p_rgb.shape, p_o.shape, p_scales.shape, p_q.shape, p_sh.shape)

        return p_xyz, p_rgb, p_o, p_scales, p_q, p_sh

    def loss(self, output, data, epoch=None, step=None):
        xyz, p_rgb, p_o, p_sxyz, p_q, p_sh = output 
        scene = data[-1]

        B = xyz.shape[0]
        final_loss = 0
        final_l1 = 0
        final_ssim = 0
        final_edge = 0
        final_normal = 0
        count = 0
        for i in range(B):
            if self.sh_n > 0:
                self.gaussion.set_args(xyz[i], p_rgb[i], p_o[i], p_sxyz[i], p_q[i], p_sh[i])
            else:
                self.gaussion.set_args(xyz[i], p_rgb[i], p_o[i], p_sxyz[i], p_q[i])
            _views = scene[i].getTrainCameras()
            for _view in _views:
                render_pkg = self.gaussion.render(_view)
                image = render_pkg["render"]
                gt_image = _view.original_image.cuda()
                # print(image.shape, gt_image.shape)
                # print(torch.mean(gt_image))

                image = torch.clamp(image, 0, 1)
                gt_image = torch.clamp(gt_image, 0, 1)

                
                
                edge_mask = get_edge_mask(gt_image)

                Ll1 = l1_loss(image, gt_image)
                Lssim = ssim(image, gt_image)

                Ll1_w = (1.0 - self.op_2dgs.lambda_dssim)
                Lssim_w = self.op_2dgs.lambda_dssim 
                
                Ledge = torch.abs((image - gt_image))
                Ledge = edge_mask * Ledge
                Ledge = torch.sum(Ledge) / (torch.sum(edge_mask)+1) / 3
                
                Ledge_w = Ll1_w * 0.5
                Ll1_w = 1 - Lssim_w - Ledge_w
                
                loss = Ll1_w * Ll1 + Lssim_w * (1.0 - Lssim) + Ledge_w * Ledge
                
                rend_normal  = render_pkg['rend_normal']
                surf_normal = render_pkg['surf_normal']
                normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
                normal_loss = self.op_2dgs.lambda_normal * (normal_error).mean()

                # total_loss = loss + normal_loss
                total_loss = loss
                final_loss += total_loss
                final_l1 += Ll1
                final_ssim += Lssim
                final_edge += Ledge
                final_normal += normal_loss
                count += 1
        final_loss = final_loss / count
        final_l1 = final_l1 / count
        final_ssim = final_ssim / count 
        final_edge = final_edge / count
        final_normal = final_normal / count

        return final_loss, final_l1, final_ssim, final_edge, final_normal

    def loss_patch(self, patch_output, all_output, data):
        patch_xyz, patch_p_rgb, patch_p_o, patch_p_sxyz, patch_p_q, patch_p_sh = patch_output 
        scene = data[-1]
        idx = data[-3]
        anchor = data[-4]
        patch_xyz = patch_xyz + anchor
        B = patch_xyz.shape[0]

        if all_output is not None:
            all_xyz, all_p_rgb, all_p_o, all_p_sxyz, all_p_q, all_p_sh = all_output 
            B, N = all_xyz.shape[:2]

            idx = torch.unsqueeze(idx, -1).repeat([1, 1, self.split_n]).reshape([B, N])

            xyz    = torch.zeros_like(all_xyz   ).float().cuda()
            p_rgb  = torch.zeros_like(all_p_rgb ).float().cuda()
            p_o    = torch.zeros_like(all_p_o   ).float().cuda()
            p_sxyz = torch.zeros_like(all_p_sxyz).float().cuda()
            p_q    = torch.zeros_like(all_p_q   ).float().cuda()
            p_sh   = torch.zeros_like(all_p_sh  ).float().cuda()


            for i in range(B):
                xyz   [i, idx[i]] = patch_xyz   [i]
                p_rgb [i, idx[i]] = patch_p_rgb [i]
                p_o   [i, idx[i]] = patch_p_o   [i]
                p_sxyz[i, idx[i]] = patch_p_sxyz[i]
                p_q   [i, idx[i]] = patch_p_q   [i]
                p_sh  [i, idx[i]] = patch_p_sh  [i]
                xyz   [i, ~idx[i]] = all_xyz   [i, ~idx[i]]
                p_rgb [i, ~idx[i]] = all_p_rgb [i, ~idx[i]]
                p_o   [i, ~idx[i]] = all_p_o   [i, ~idx[i]]
                p_sxyz[i, ~idx[i]] = all_p_sxyz[i, ~idx[i]]
                p_q   [i, ~idx[i]] = all_p_q   [i, ~idx[i]]
                p_sh  [i, ~idx[i]] = all_p_sh  [i, ~idx[i]]
        else:
            xyz    = patch_xyz
            p_rgb  = patch_p_rgb
            p_o    = patch_p_o
            p_sxyz = patch_p_sxyz
            p_q    = patch_p_q
            p_sh   = patch_p_sh

        final_loss = 0
        final_l1 = 0
        final_ssim = 0
        final_edge = 0
        final_normal = 0
        count = 0
        for i in range(B):
            self.gaussion.set_args(xyz[i], p_rgb[i], p_o[i], p_sxyz[i], p_q[i], p_sh[i])
            _views = scene[i].getTrainCameras()
            for _view in _views:
                render_pkg = self.gaussion.render(_view)
                image = render_pkg["render"]
                gt_image = _view.original_image.cuda()
                edge_mask = get_edge_mask(gt_image)
                # print(image.shape, gt_image.shape)
                # print(torch.mean(gt_image))
                Ll1 = l1_loss(image, gt_image)
                Lssim = ssim(image, gt_image)

                Ledge = torch.abs((image - gt_image))
                Ledge = edge_mask * Ledge
                Ledge = torch.sum(Ledge) / torch.sum(edge_mask) / 3

                Ll1_w = (1.0 - self.op_2dgs.lambda_dssim)
                Lssim_w = self.op_2dgs.lambda_dssim 
                Ledge_w = Ll1_w * 0.5
                Ll1_w = 1 - Lssim_w - Ledge_w

                loss =  Ll1_w * Ll1 + Lssim_w * (1.0 - Lssim) + Ledge_w * Ledge

                rend_normal  = render_pkg['rend_normal']
                surf_normal = render_pkg['surf_normal']
                normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
                normal_loss = self.op_2dgs.lambda_normal * (normal_error).mean()

                # total_loss = loss + normal_loss
                total_loss = loss
                final_loss += total_loss
                final_l1 += Ll1
                final_ssim += Lssim
                final_edge += Ledge
                final_normal += normal_loss
                count += 1
        final_loss = final_loss / count
        final_l1 = final_l1 / count
        final_ssim = final_ssim / count 
        final_edge = final_edge / count
        final_normal = final_normal / count

        return final_loss, final_l1, final_ssim, final_edge, final_normal

    def loss_test(self, output, data, return_img=False, ignore_missing=False, put_text=True, margin=0):
        xyz, p_rgb, p_o, p_sxyz, p_q, p_sh = output 
        scene = data[-1]

        lpips_resize = Resize(224)

        if return_img:
            res_img = []
        
        if isinstance(xyz, list):
            B = len(xyz)
        else:
            B = xyz.shape[0]
        final_ssim = []
        final_psnr = []
        final_lpips = []
        render_times = []

        for i in range(B):
            batch_final_ssim = []
            batch_final_psnr = []
            batch_final_lpips = []
            if return_img:
                batch_res_img = []
            if self.sh_n > 0:
                self.gaussion.set_args(xyz[i], p_rgb[i], p_o[i], p_sxyz[i], p_q[i], p_sh[i])
            else:
                self.gaussion.set_args(xyz[i], p_rgb[i], p_o[i], p_sxyz[i], p_q[i])
            _views = scene[i].getTrainCameras()
            for _view in _views:
                start_time = time.time()
                render_pkg = self.gaussion.render(_view)
                end_time = time.time()
                render_times.append(end_time - start_time)
                image = render_pkg["render"]
                gt_image = _view.original_image.cuda()

                # print(image.shape, gt_image.shape)
                image = torch.clamp(image, 0, 1)
                gt_image = torch.clamp(gt_image, 0, 1)

                if ignore_missing:
                    missing_mask = image < 1e-5
                    missing_mask = torch.all(missing_mask, dim=0)
                    missing_mask = missing_mask.unsqueeze(0).repeat([3, 1, 1])
                    image[missing_mask] = 0
                    gt_image[missing_mask] = 0
                
                if margin > 0:
                    old_h, old_w = image.shape[1:]
                    image = torch.nn.functional.pad(image, (margin, margin), mode='constant', value=0)
                    gt_image = torch.nn.functional.pad(gt_image, (margin, margin), mode='constant', value=0)
                    image = torch.nn.functional.interpolate(image.unsqueeze(0), (old_h, old_w), mode='bilinear', align_corners=False).squeeze(0)
                    gt_image = torch.nn.functional.interpolate(gt_image.unsqueeze(0), (old_h, old_w), mode='bilinear', align_corners=False).squeeze(0)

                Lssim = ssim(image, gt_image)
                Lpsnr = psnr(image, gt_image)

                _image = lpips_resize(image)
                _gt_image = lpips_resize(gt_image)
                _image = _image.unsqueeze(0)
                _gt_image = _gt_image.unsqueeze(0)
                _image = 2*_image - 1
                _gt_image = 2*_gt_image - 1

                Llpips = lpips_loss_fn(_image, _gt_image)

                if return_img:
                    image_np = (image.detach().cpu().numpy().transpose([1, 2, 0])*255).astype(np.uint8)[:,:,::-1]
                    gt_image_np = (gt_image.detach().cpu().numpy().transpose([1, 2, 0])*255).astype(np.uint8)[:,:,::-1]
                    image_np = np.ascontiguousarray(image_np)
                    gt_image_np = np.ascontiguousarray(gt_image_np)
                    image_np_org = image_np.copy()
                    gt_image_np_org = gt_image_np.copy()
                    if put_text:
                        cv2.putText(image_np, f'ssim: {Lssim.item():.4f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(image_np, f'psnr: {Lpsnr.item():.4f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(image_np, f'lpips: {Llpips.item():.4f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    batch_res_img.append([image_np, gt_image_np, image_np_org, gt_image_np_org])

                batch_final_ssim.append(Lssim)
                batch_final_psnr.append(Lpsnr)
                batch_final_lpips.append(Llpips)
            batch_final_ssim  = torch.mean(torch.stack(batch_final_ssim ))
            batch_final_psnr  = torch.mean(torch.stack(batch_final_psnr ))
            batch_final_lpips = torch.mean(torch.stack(batch_final_lpips))
            if return_img:
                if put_text:
                    cv2.putText(batch_res_img[0][1], f'ssim: {batch_final_ssim.item():.4f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(batch_res_img[0][1], f'psnr: {batch_final_psnr.item():.4f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(batch_res_img[0][1], f'lpips: {batch_final_lpips.item():.4f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                res_img.append(batch_res_img)
            final_ssim.append(batch_final_ssim)
            final_psnr.append(batch_final_psnr)
            final_lpips.append(batch_final_lpips)

        final_ssim  = torch.mean(torch.stack(final_ssim ))
        final_psnr  = torch.mean(torch.stack(final_psnr ))
        final_lpips = torch.mean(torch.stack(final_lpips))
        final_render_time = np.mean(render_times)
        
        
        if return_img:
                return final_ssim, final_psnr, final_lpips, res_img, final_render_time
        else:
            return final_ssim, final_psnr, final_lpips, None, final_render_time
        