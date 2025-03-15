import os 
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
from plyfile import PlyData, PlyElement

from network.model import GauPCRender
from network.dataset import GaussianDataset, GaussianPatchDataset
from tqdm import tqdm


def merge_images(images, images_per_row=10):
    widths = [img.shape[1] for img in images]
    heights = [img.shape[0] for img in images]

    max_width = max(widths)
    max_height = max(heights)
    rows = (len(images) + images_per_row - 1) // images_per_row
    merged_width = max_width * images_per_row
    merged_height = max_height * rows

    merged_image = np.zeros((merged_height, merged_width, 3), np.uint8)

    x_offset = 0
    y_offset = 0
    for i, img in enumerate(images):
        if i % images_per_row == 0 and i != 0:
            x_offset = 0
            y_offset += max_height
        merged_image[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
        x_offset += img.shape[1]

    return merged_image

def gen_gaussians_vis_dir(path, xyz, p_rgb, p_o, p_sxyz, p_q, p_sh):
    one_line = np.full((p_sxyz.shape[0],1), 0.0000000001)
    one_line = np.log(one_line)
    p_sxyz = np.concatenate((p_sxyz, one_line), axis=1)

    attr_name = ['x', 'y', 'z', 'nx', 'ny', 'nz']

    for i in range(3):
        attr_name.append('f_dc_{}'.format(i))
    for i in range(p_sh.shape[1]):
        attr_name.append('f_rest_{}'.format(i))
    attr_name.append('opacity')
    for i in range(3):
        attr_name.append('scale_{}'.format(i))
    for i in range(4):
        attr_name.append('rot_{}'.format(i))

    normals = np.zeros_like(xyz)
    f_dc = p_rgb
    f_rest = p_sh
    opacities = p_o
    scale = p_sxyz
    rotation = p_q

    dtype_full = [(attribute, 'f4') for attribute in attr_name]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    res_root_path = path
    res_path = os.path.join(res_root_path, 'point_cloud', 'iteration_7000')
    os.makedirs(res_path, exist_ok=True)
    os.system(f'cp /home/mcf/data2/works/point23dgs/point23dgs/code3/p2gs/assets/cameras.json {res_root_path}')
    os.system(f'cp /home/mcf/data2/works/point23dgs/point23dgs/code3/p2gs/assets/cfg_args {res_root_path}')
    res_path = os.path.join(res_path, 'point_cloud.ply')
    PlyData([el]).write(res_path)


def eval(cmd_args, logging):
    cmd_args.args_2dgs.pre_data_image_num = None

    logging.info('Evaluating GauPCRender')
    batch_size = cmd_args.batch_size

    if cmd_args.patch:
        batch_size = 1

    train_name = cmd_args.train_name
    save_path = os.path.join(cmd_args.save_path, train_name)

    logging.info(f'Loading dataset')
    if not cmd_args.patch:
        dataset_eval = GaussianDataset(cmd_args, 'eval')
        dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, collate_fn=GaussianDataset.collet_fn)
    else:
        dataset_eval = GaussianPatchDataset(cmd_args, 'eval')
        dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, collate_fn=GaussianPatchDataset.collet_fn)

    logging.info(f'Loading GauPCRender')
    model = GauPCRender(cmd_args, cmd_args.patch)
    if cmd_args.restore is not None:
        weight_path = os.path.join(save_path, f'weight_{int(cmd_args.restore)}.pkl')
        logging.info(f'Load weight from: {weight_path}')
        model.load_state_dict(torch.load(weight_path))
    else:
        assert False, 'Please specify the weight file to restore'
    model.eval()
    model.cuda()

    print('<----------CONFIG---------->')
    print(f'Training name: {train_name}')
    print(f'Category: {cmd_args.cate}')
    print(f'Save path: {save_path}')
    print(f'Batch size: {batch_size}')
    print(f'Restore: {cmd_args.restore}')
    print(f'Point number: {cmd_args.point_number}')
    print(f'Patch: {cmd_args.patch}')
    print('<----------CONFIG---------->')

    # test
    ssim_s = []
    psnr_s = []
    lpips_s = []
    rendering_time_s = []
    with torch.no_grad():
        for data in tqdm(dataloader_eval):
            model_ids = data[-2]
            data = dataset_eval.to_cuda(data)
            if not cmd_args.scene_cate:
                # Object Category evaluation
                if not cmd_args.patch:
                    # Entire Model evaluation
                    batch_batch_number = len(data[0]) // 8
                    if batch_batch_number == 0:
                        output = model(data)
                    else:
                        output = []
                        for i in range(batch_batch_number):
                            if i == batch_batch_number-1:
                                data_batch = [data[0][i*8:], data[1][i*8:], data[2][i*8:]]
                            else:
                                data_batch = [data[0][i*8:i*8+8], data[1][i*8:i*8+8], data[2][i*8:i*8+8]]
                            _output = model(data_batch)
                            if output == []:
                                for _ in range(len(_output)):
                                    output.append([])
                            for j in range(len(_output)):
                                output[j].append(_output[j])
                        output = [torch.cat(x, dim=0) for x in output]
                else:
                    # Patch Model evaluation
                    batch_batch_number = len(data[0][0]) // 8
                    if batch_batch_number == 0:
                        output = model([data[0][0], data[1][0], data[2][0]])
                    else:
                        output = []
                        for i in range(batch_batch_number):
                            if i == batch_batch_number-1:
                                data_batch = [data[0][0][i*8:], data[1][0][i*8:], data[2][0][i*8:]]
                            else:
                                data_batch = [data[0][0][i*8:i*8+8], data[1][0][i*8:i*8+8], data[2][0][i*8:i*8+8]]
                            _output = model(data_batch)
                            if output == []:
                                for _ in range(len(_output)):
                                    output.append([])
                            for j in range(len(_output)):
                                output[j].append(_output[j])
                        output = [torch.cat(x, dim=0) for x in output]
                    xyz, p_rgb, p_o, p_sxyz, p_q, p_sh = output 

                    anchor = data[6][0]
                    anchor = torch.unsqueeze(anchor, 1)
                    xyz    = xyz + anchor
                    xyz    = torch.reshape(xyz,    [1, -1, 3])
                    p_rgb  = torch.reshape(p_rgb,  [1, -1, 3])
                    p_o    = torch.reshape(p_o,    [1, -1, 1])
                    p_sxyz = torch.reshape(p_sxyz, [1, -1, 2])
                    p_q    = torch.reshape(p_q,    [1, -1, 4])
                    p_sh   = torch.reshape(p_sh,   [1, -1, p_sh.shape[2]])
                
                    output = xyz, p_rgb, p_o, p_sxyz, p_q, p_sh
                if cmd_args.save_img_merge or cmd_args.save_img_file:
                    ssim, psnr, lpips, res_img, rendering_time = model.loss_test(output, data, return_img=True, ignore_missing=cmd_args.ignore_missing)
                    save_path = os.path.join(os.path.dirname(weight_path), 'img', f'{cmd_args.cate}_p{cmd_args.point_number}')
                    os.makedirs(save_path, exist_ok=True)
                    for b in range(len(res_img)):
                        if cmd_args.save_img_merge:
                            fuse_imgs = []
                            for k in range(len(res_img[b])):
                                img, gt_img = res_img[b][k][:2]
                                fuse_img = np.concatenate((gt_img, img), axis=0)
                                fuse_imgs.append(fuse_img)
                            fuse_imgs = merge_images(fuse_imgs)
                            cv2.imwrite(os.path.join(save_path, f'{model_ids[b]}.jpg'), fuse_imgs)
                        if cmd_args.save_img_file:
                            save_img_path = os.path.join(save_path, f'{model_ids[b]}')
                            # print(f'save_img_path: {save_img_path}')
                            os.makedirs(save_img_path, exist_ok=True)
                            scene = data[-1][b]
                            views = scene.getTrainCameras()
                            for k in range(len(views)):
                                img, gt_img = res_img[b][k][2:]
                                file_name = views[k].image_name
                                cv2.imwrite(os.path.join(save_img_path, f'{file_name}.jpg'), img)
                else:
                    ssim, psnr, lpips, _, rendering_time  = model.loss_test(output, data, ignore_missing=cmd_args.ignore_missing)
                if cmd_args.save_gs_file:
                    data_xyz = data[0].detach().cpu().numpy()
                    data_rgb = data[1].detach().cpu().numpy()
                    data_normal = data[2].detach().cpu().numpy()
                    output = list(output)
                    for i in range(len(output)):
                        output[i] = output[i].detach().cpu().numpy()
                    save_path = os.path.join(os.path.dirname(weight_path), 'gs_npy', f'{cmd_args.cate}_p{cmd_args.point_number}')
                    save_path_2 = os.path.join(os.path.dirname(weight_path), 'gs_dir', f'{cmd_args.cate}_p{cmd_args.point_number}')
                    os.makedirs(save_path, exist_ok=True)
                    for b in range(batch_size):
                        _output = [x[b] for x in output]
                        save_data = {'input': [data_xyz[b], data_rgb[b], data_normal[b]], 'output': _output}
                        np.save(os.path.join(save_path, f'{model_ids[b]}.npy'), save_data, allow_pickle=True)
                        gen_gaussians_vis_dir(os.path.join(save_path_2, f'{model_ids[b]}'), *_output)
            else:
                # Scene Category evaluation
                data[-2] = [data[-2] for _ in range(len(data[0]))]
                scenes = []
                scene = data[-1][0]
                _views = scene.getTrainCameras()
                for view in _views:
                    class my_scene:
                        def __init__(self):
                            self.view = view
                        def getTrainCameras(self):
                            return [self.view]
                    scenes.append(my_scene())
                data[-1] = scenes
                output = []
                for i in range(len(data[0])):
                    __b = data[0][i].shape[0]
                    if __b == 1:
                        _output = model([data[0][i].repeat(2,1,1), data[1][i].repeat(2,1,1), data[2][i].repeat(2,1,1)])
                        _output = [x[:1] for x in _output]
                    else:
                        _output = model([data[0][i], data[1][i], data[2][i]])
                    xyz, p_rgb, p_o, p_sxyz, p_q, p_sh = _output 

                    anchor = data[6][i]
                    anchor = torch.unsqueeze(anchor, 1)
                    xyz    = xyz + anchor
                    xyz    = torch.reshape(xyz,    [-1, 3])
                    p_rgb  = torch.reshape(p_rgb,  [-1, 3])
                    p_o    = torch.reshape(p_o,    [-1, 1])
                    p_sxyz = torch.reshape(p_sxyz, [-1, 2])
                    p_q    = torch.reshape(p_q,    [-1, 4])
                    p_sh   = torch.reshape(p_sh,   [-1, p_sh.shape[2]])
                    _output = xyz, p_rgb, p_o, p_sxyz, p_q, p_sh

                    if output == []:
                        for _ in range(len(_output)):
                            output.append([])
                    for j in range(len(_output)):
                        output[j].append(_output[j])
                if cmd_args.save_img_merge or cmd_args.save_img_file:
                    ssim, psnr, lpips, res_img, rendering_time = model.loss_test(output, data, return_img=True, ignore_missing=cmd_args.ignore_missing)
                    save_path = os.path.join(os.path.dirname(weight_path), 'img', f'{cmd_args.cate}_p{cmd_args.point_number}')
                    os.makedirs(save_path, exist_ok=True)
                    for b in range(batch_size):
                        res_img = [[x[0] for x in res_img]]
                        if cmd_args.save_img_merge:
                            fuse_imgs = []
                            for k in range(len(res_img[b])):
                                img, gt_img = res_img[b][k][:2]
                                fuse_img = np.concatenate((gt_img, img), axis=0)
                                fuse_imgs.append(fuse_img)
                            fuse_imgs = merge_images(fuse_imgs)
                            cv2.imwrite(os.path.join(save_path, f'{model_ids[b]}.jpg'), fuse_imgs)
                        if cmd_args.save_img_file:
                            save_img_path = os.path.join(save_path, f'{model_ids[b]}')
                            # print(f'save_img_path: {save_img_path}')
                            os.makedirs(save_img_path, exist_ok=True)
                            scenes = data[-1]
                            for k in range(len(scenes)):
                                img, gt_img = res_img[b][k][2:]
                                file_name = scenes[k].getTrainCameras()[0].image_name
                                cv2.imwrite(os.path.join(save_img_path, f'{file_name}.jpg'), img)
                else:
                    ssim, psnr, lpips, _, rendering_time  = model.loss_test(output, data, ignore_missing=cmd_args.ignore_missing)
            ssim_s.append(ssim.item())
            psnr_s.append(psnr.item())    
            lpips_s.append(lpips.item())    
            rendering_time_s.append(rendering_time)
            
    print({'ssim': np.mean(ssim_s), 'psnr': np.mean(psnr_s), 'lpips': np.mean(lpips_s), 'rendering_time': np.mean(rendering_time_s)})