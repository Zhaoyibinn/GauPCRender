import os
import json
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from plyfile import PlyData, PlyElement
import random
from sklearn.neighbors import NearestNeighbors


from scene import Scene

def estimate_normal(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        return normals

class GaussianDataset(Dataset):
    def __init__(self, cmd_args, split):
        self.args_2dgs = cmd_args.args_2dgs
        self.lp_2dgs = cmd_args.lp_2dgs
        self.cate = cmd_args.cate
        self.point_number = cmd_args.point_number
        self.split = split
        if self.split == 'train':
            self.scene_shuffle = True
        else:
            self.scene_shuffle = False

        # self.data_path = os.path.join(cmd_args.data_root, self.cate)
        # self.data_path = "data/dtu_own/scan24_3/0_1_2"
        # self.split_path = os.path.join(cmd_args.data_root, 'split.json')

        # with open(self.split_path,'r') as f:
        #     split_dict = json.load(f)
        # self.ids = split_dict[self.cate][split]

        self.data_path = cmd_args.data_path
        self.split_path = os.path.join(cmd_args.data_root, 'split.json')

        self.test_idx = [23,24,33]

        with open(self.split_path,'r') as f:
            split_dict = json.load(f)
        if split_dict[self.cate][split] != []:
            self.ids = split_dict[self.cate][split]
        else:
            all_idx = sorted(os.listdir(self.data_path))
            train_idx = []
            test_idx = []
            for idx in all_idx:
                scene_name = idx.split('_')[0]
                
                if int(idx.split('_')[1]) not in self.test_idx:
                    train_idx.append(idx)
                else:
                    test_idx.append(idx)
            if split == 'train':
                self.ids = train_idx
            else:
                self.ids = test_idx

    def __len__(self):
        return len(self.ids)    
    
    @staticmethod
    def collet_fn(data):
        xyz = np.array([d[0] for d in data])
        rgb = np.array([d[1] for d in data])
        normal = np.array([d[2] for d in data])
        ids = [d[3] for d in data]
        scene = [d[4] for d in data]
        return torch.from_numpy(xyz), torch.from_numpy(rgb), torch.from_numpy(normal), ids, scene

    @staticmethod
    def to_cuda(data):
        scenes = data[4]
        for scene in scenes:
            for resolution_scale in scene.train_cameras:
                for cam in scene.train_cameras[resolution_scale]:
                    cam.to_cuda()
        return data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3], data[4]


    @staticmethod
    def random_sample(data, number):
        xyz = data[0]
        n = xyz.shape[0]
        # np.random.seed(0)
        idx = np.random.permutation(n)
        if idx.shape[0] < number:
            idx = np.concatenate([idx, np.random.randint(xyz.shape[0], size=number-xyz.shape[0])])
        res_data = []
        for d in data:
            res_data.append(d[idx[:number]])
        return res_data
    
    def __getitem__(self, idx):
        _id = self.ids[idx]
        self.args_2dgs.source_path = os.path.join(self.data_path, _id)
        try:
            scene = Scene(self.lp_2dgs.extract(self.args_2dgs), shuffle=self.scene_shuffle)
        except Exception as e:
            print(f'Error in scene {_id}')
            print(e)
            exit()

        input_points_ply_file_path = os.path.join(self.data_path, _id, 'points3d.ply')
        if os.path.exists(input_points_ply_file_path):
            plydata_input = PlyData.read(input_points_ply_file_path)
            x  = plydata_input.elements[0].data['x']
            y  = plydata_input.elements[0].data['y']
            z  = plydata_input.elements[0].data['z']
            r  = plydata_input.elements[0].data['red']
            g  = plydata_input.elements[0].data['green']
            b  = plydata_input.elements[0].data['blue'] 
            xyz = np.stack([x, y, z], -1)
            rgb = np.stack([r, g, b], -1)    

        else:
            vggt_ply = o3d.io.read_point_cloud(os.path.join(self.data_path, _id, 'sparse','0','points3D.ply'))
            xyz = np.array(vggt_ply.points)
            rgb = np.array(vggt_ply.colors) * 255
        

          
        if self.point_number != -1:
            input_xyz, input_rgb = self.random_sample([xyz, rgb], self.point_number)
        else:
            if xyz.shape[0] >= 200_000:
                input_xyz, input_rgb = self.random_sample([xyz, rgb], 200_000)
            else:
                input_xyz = xyz
                input_rgb = rgb
        input_normal = estimate_normal(input_xyz)
        return input_xyz, input_rgb, input_normal, _id, scene

class GaussianPatchDataset(Dataset):
    # def __init__(self, cate, data_path, split='train', point_number=40_000, patch_point_number=2048, is_all_patch=False,
    # scene_shuffle=True, scene_img_patch=False, scene_add_background=False, all_scene_img_patch=False, noisy_ratio=None):
    def __init__(self, cmd_args, split):
        self.args_2dgs = cmd_args.args_2dgs
        self.lp_2dgs = cmd_args.lp_2dgs
        self.cate = cmd_args.cate
        self.point_number = cmd_args.point_number
        self.scene_cate = cmd_args.scene_cate
        self.split = split
        if self.split == 'train':
            self.scene_shuffle = True
            self.is_all_patch = False
        else:
            self.scene_shuffle = False
            self.is_all_patch = True
        self.patch_point_number = cmd_args.patch_point_number
        # self.data_path = os.path.join(cmd_args.data_root, self.cate)
        self.data_path = cmd_args.data_path
        self.split_path = os.path.join(cmd_args.data_root, 'split.json')

        self.test_idx = [23,24,33]

        with open(self.split_path,'r') as f:
            split_dict = json.load(f)
        if split_dict[self.cate][split] != []:
            self.ids = split_dict[self.cate][split]
        else:
            all_idx = sorted(os.listdir(self.data_path))
            train_idx = []
            test_idx = []
            for idx in all_idx:
                scene_name = idx.split('_')[0]
                
                if int(idx.split('_')[1]) not in self.test_idx:
                    train_idx.append(idx)
                else:
                    test_idx.append(idx)
            if split == 'train':
                self.ids = train_idx
            else:
                self.ids = test_idx

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collet_fn(data):
        if data[0][-1] == 66:
            data[0] = list(data[0])
            data[0].pop()
            collet_data = []
            for i in range(7):
                # _data = torch.from_numpy(np.array([d[i] for d in data]))
                _data = [torch.from_numpy(d) for d in data[0][i]]
                collet_data.append(_data)
            collet_data.append([d[-3] for d in data])
            collet_data.append([d[-2] for d in data])
            collet_data.append([d[-1] for d in data])
            return collet_data
        else:
            collet_data = []
            for i in range(8):
                _data = torch.from_numpy(np.array([d[i] for d in data]))
                collet_data.append(_data)
            collet_data.append([d[-2] for d in data])
            collet_data.append([d[-1] for d in data])
            return collet_data

    @staticmethod
    def to_cuda(data):
        if isinstance(data[0], list):
            data = list(data)
            for i in range(7):
                data[i] = [d.cuda() for d in data[i]]
                # data[i] = data[i].cuda()
            scenes = data[-1]
            for scene in scenes:
                for resolution_scale in scene.train_cameras:
                    for cam in scene.train_cameras[resolution_scale]:
                        cam.to_cuda()
            return data
        else:
            data = list(data)
            for i in range(8):
                data[i] = data[i].cuda()
            scenes = data[-1]
            for scene in scenes:
                for resolution_scale in scene.train_cameras:
                    for cam in scene.train_cameras[resolution_scale]:
                        cam.to_cuda()
            return data

    @staticmethod
    def random_sample(data, number):
        xyz = data[0]
        n = xyz.shape[0]
        # np.random.seed(0)
        idx = np.random.permutation(n)
        if idx.shape[0] < number:
            idx = np.concatenate([idx, np.random.randint(xyz.shape[0], size=number-xyz.shape[0])])
        res_data = []
        for d in data:
            res_data.append(d[idx[:number]])
        return res_data
    
    def random_patch(self, data, knn=None, anchor=None):
        xyz = data[0]
        if self.patch_point_number >= xyz.shape[0]:
            selected_idx = np.ones_like(xyz[:,0], dtype=bool)
            return data, anchor, selected_idx
        if anchor is None:
            anchor = random.choice(xyz)
        anchor = np.expand_dims(anchor, 0)
        if knn is None:
            knn = NearestNeighbors(n_neighbors=self.patch_point_number)
            knn.fit(xyz)
        _, idx = knn.kneighbors(anchor)
        idx = idx[0]
        res_data = []
        for d in data:
            res_data.append(d[idx])
        selected_idx = np.zeros_like(xyz[:,0], dtype=bool)
        selected_idx[idx] = 1
        return res_data, anchor, selected_idx

    def all_patch(self, data):
        xyz = data[0]
        knn = NearestNeighbors(n_neighbors=self.patch_point_number)
        knn.fit(xyz)
        selected_idx = np.zeros_like(xyz[:,0], dtype=bool)
        anchors = []
        points_s = []
        colors_s = []
        normals_s = []
        while True:
            rest_xyz = xyz[~selected_idx]
            anchor = random.choice(rest_xyz)
            anchors.append(anchor)
            res_data, _, idx = self.random_patch(data, knn, anchor=anchor)
            points, colors, normals = res_data 
            points = points - anchor
            points_s.append(points)
            colors_s.append(colors)
            normals_s.append(normals)
            selected_idx[idx] = 1
            if np.sum(selected_idx) >= selected_idx.shape[0]:
                break
        anchors = np.asarray(anchors)
        points_s = np.asarray(points_s)
        colors_s = np.asarray(colors_s)
        normals_s = np.asarray(normals_s)
        return points_s, colors_s, normals_s, anchors

    def get_scene_img_patch(self, id, scene, view_idx=0):
        camera = scene.getTrainCameras()[view_idx]
        image_name = camera.image_name
        idx = int(image_name[:-2])
        img_patch_path = os.path.join(self.data_path, id, 'img_patch', f'{image_name}.ply')
        points = o3d.io.read_point_cloud(img_patch_path)
        xyz = np.asarray(points.points)
        rgb = np.asarray(points.colors) * 255
        return xyz, rgb

    def add_noise(self, xyz):
        noise = np.random.randn(*xyz.shape) * self.noisy_ratio
        xyz += noise
        return xyz

    def __getitem__(self, idx):
        _id = self.ids[idx]
        self.args_2dgs.source_path = os.path.join(self.data_path, _id)
        try:
            scene = Scene(self.lp_2dgs.extract(self.args_2dgs), shuffle=self.scene_shuffle)
        except Exception as e:
            print(f'Error in scene {_id}')
            print(e)
            exit()
        input_points_ply_file_path = os.path.join(self.data_path, _id, 'points3d.ply')
        if not self.scene_cate:
            input_points_ply_file_path = os.path.join(self.data_path, _id, 'points3d.ply')
            if os.path.exists( os.path.join(self.data_path, _id, 'img_patch')):
                plydata_input = PlyData.read(input_points_ply_file_path)
                x  = plydata_input.elements[0].data['x']
                y  = plydata_input.elements[0].data['y']
                z  = plydata_input.elements[0].data['z']
                r  = plydata_input.elements[0].data['red']
                g  = plydata_input.elements[0].data['green']
                b  = plydata_input.elements[0].data['blue']
                xyz = np.stack([x, y, z], -1)
                rgb = np.stack([r, g, b], -1)

            else:
                vggt_ply = o3d.io.read_point_cloud(os.path.join(self.data_path, _id, 'sparse','0','points3D.ply'))
                xyz = np.array(vggt_ply.points)
                rgb = np.array(vggt_ply.colors) * 255
            

            if self.point_number != -1:
                all_xyz, all_rgb = self.random_sample([xyz, rgb], self.point_number)
            else:
                all_xyz = xyz
                all_rgb = rgb
            all_normal = estimate_normal(all_xyz)
            if not self.is_all_patch:
                (patch_xyz, patch_rgb), anchor, idx = self.random_patch([all_xyz, all_rgb])
                patch_xyz = patch_xyz - anchor
                patch_normal = estimate_normal(patch_xyz)
                return patch_xyz, patch_rgb, patch_normal, all_xyz, all_rgb, all_normal, anchor, idx, _id, scene
            else:
                all_patch_xyz, all_patch_rgb, all_patch_normal, anchor = self.all_patch([all_xyz, all_rgb, all_normal])
                # print(f'scene: {_id}, xyz: {xyz.shape}, anchor: {anchor.shape}, all_patch_xyz: {all_patch_xyz.shape}')
                return all_patch_xyz, all_patch_rgb, all_patch_normal, all_xyz, all_rgb, all_normal, anchor, idx, _id, scene
        else:
            if not self.is_all_patch or len(scene.getTrainCameras()) == 1:
                # random select a view of a scene
                if os.path.exists( os.path.join(self.data_path, _id, 'img_patch')):
                    xyz, rgb = self.get_scene_img_patch(_id, scene)   

                else:
                    vggt_ply = o3d.io.read_point_cloud(os.path.join(self.data_path, _id, 'sparse','0','points3D.ply'))
                    xyz = np.array(vggt_ply.points)
                    rgb = np.array(vggt_ply.colors) * 255
                if self.point_number != -1:
                    all_xyz, all_rgb = self.random_sample([xyz, rgb], self.point_number)
                else:
                    all_xyz = xyz
                    all_rgb = rgb
                all_normal = estimate_normal(all_xyz)
                all_patch_xyz, all_patch_rgb, all_patch_normal, anchor = self.all_patch([all_xyz, all_rgb, all_normal])
                # print(f'scene: {_id}, xyz: {xyz.shape}, anchor: {anchor.shape}, all_patch_xyz: {all_patch_xyz.shape}')
                b = all_patch_xyz.shape[0]
                # if b > 50:
                #     return 0, 0, 0, 0, 0, 0, 0, idx, _id, scene
                return all_patch_xyz, all_patch_rgb, all_patch_normal, all_xyz, all_rgb, all_normal, anchor, idx, _id, scene
            else:
                # special process for all the views of a scene for evaluation
                all_patch_xyz = []
                all_patch_rgb = []
                all_patch_normal = []
                all_xyz = []
                all_rgb = []
                all_normal = []
                anchor = []
                for view_idx in range(len(scene.getTrainCameras())):
                    # print(view_idx)
                    if os.path.exists( os.path.join(self.data_path, _id, 'img_patch')):
                        xyz, rgb = self.get_scene_img_patch(_id, scene, view_idx=view_idx)
                    else:
                        vggt_ply = o3d.io.read_point_cloud(os.path.join(self.data_path, _id, 'sparse','0','points3D.ply'))
                        xyz = np.array(vggt_ply.points)
                        rgb = np.array(vggt_ply.colors) * 255
                    # xyz, rgb = self.get_scene_img_patch(_id, scene)
                    # print('org', xyz.shape, self.point_number)

                    if self.point_number != -1:
                        if self.point_number < 100_000:
                            ratio = self.point_number / 100_000
                            _sample_number = int(xyz.shape[0] * ratio)
                            input_xyz, input_rgb = self.random_sample([xyz, rgb], _sample_number)
                        else:
                            input_xyz, input_rgb = self.random_sample([xyz, rgb], self.point_number)
                    else:
                        if xyz.shape[0] >= 100_000:
                            input_xyz, input_rgb = self.random_sample([xyz, rgb], 100_000)
                        else:
                            input_xyz = xyz
                            input_rgb = rgb
                    # print('sampled', input_xyz.shape)
                    if rgb.shape[0] == 0:
                        vggt_ply = o3d.io.read_point_cloud(os.path.join(self.data_path, _id, 'sparse','0','points3D.ply'))
                        input_xyz = np.array(vggt_ply.points)
                        input_rgb = np.array(vggt_ply.colors) * 255
                    input_normal = estimate_normal(input_xyz)
                    _all_xyz = input_xyz
                    _all_rgb = input_rgb
                    _all_normal = input_normal
                    _all_patch_xyz, _all_patch_rgb, _all_patch_normal, _anchor = self.all_patch([_all_xyz, _all_rgb, _all_normal])
                    all_patch_xyz.append(_all_patch_xyz)
                    all_patch_rgb.append(_all_patch_rgb)
                    all_patch_normal.append(_all_patch_normal)
                    anchor.append(_anchor)
                    all_xyz.append(input_xyz)
                    all_rgb.append(input_rgb)
                    all_normal.append(input_normal)
                return all_patch_xyz, all_patch_rgb, all_patch_normal, all_xyz, all_rgb, all_normal, anchor, idx, _id, scene, 66

        
