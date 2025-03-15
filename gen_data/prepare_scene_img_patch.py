import os 
import sys 
import numpy as np
import open3d as o3d
import json
import argparse

parser = argparse.ArgumentParser(description='prepare scene img patch')
parser.add_argument('cate', type=str, choices=['scannet', 'dtu'])
parser.add_argument('data_path', type=str)
cmd_args = parser.parse_args()

sys.path.append('../util/plane_gs')
from scene import Scene
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


def get_2dgs_args():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args([])
    args.save_iterations.append(args.iterations)
    return args, lp, op, pp


args_2dgs, lp_2dgs, op_2dgs, pp_2dgs = get_2dgs_args()

def get_scene_img_patch(xyz, rgb, scene, view_idx=0):
    camera = scene.getTrainCameras()[view_idx]
    R = camera.R
    T = camera.T
    fovx = camera.FoVx
    fovy = camera.FoVy
    width = camera.image_width
    height = camera.image_height
    # print(width, height, camera.image_name)
    def calculate_intrinsic_matrix(fovx, fovy, width, height):
        """
        计算相机内参矩阵

        :param fovx: 水平视场角（度）
        :param fovy: 垂直视场角（度）
        :param width: 渲染图片的宽度
        :param height: 渲染图片的高度
        :return: 相机内参矩阵
        """
        # 将视场角从度转换为弧度
        # fovx_rad = np.radians(fovx)
        # fovy_rad = np.radians(fovy)
        
        # 计算水平和垂直焦距
        fx = width / (2 * np.tan(fovx / 2))
        fy = height / (2 * np.tan(fovy / 2))
        
        # 相机内参矩阵
        intrinsic_matrix = np.array([
            [fx, 0, width / 2],
            [0, fy, height / 2],
            [0, 0, 1]
        ])
        # print(intrinsic_matrix)
        return intrinsic_matrix
    
    def project_points(intrinsic_matrix, rotation_matrix, translation_vector, point_cloud):
        """
        将点云投影到图像平面上，并返回相机可见的点云

        :param intrinsic_matrix: 相机内参矩阵
        :param rotation_matrix: 相机旋转矩阵
        :param translation_vector: 相机位置（平移向量）
        :param point_cloud: 点云（Nx3矩阵）
        :return: 相机可见的点云
        """
        # 将点云从世界坐标系转换到相机坐标系
        point_cloud_camera = np.dot(point_cloud, rotation_matrix) + translation_vector
        # write_point_cloud_as_ply('trash/tmp_point_cloud.ply', point_cloud_camera)
        
        # 将相机坐标系中的点投影到图像平面上
        # print(intrinsic_matrix)
        projected_points = np.dot(intrinsic_matrix, point_cloud_camera.T).T
        projected_points[:, 0] /= projected_points[:, 2]
        projected_points[:, 1] /= projected_points[:, 2]
        
        # 获取图像的宽度和高度
        width = intrinsic_matrix[0, 2] * 2
        height = intrinsic_matrix[1, 2] * 2
        
        # 检查投影点是否在图像范围内
        visible_points_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < width) & \
                            (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height) & \
                            (projected_points[:, 2] > 0)
        
        
        # # 可视化
        # idx = visible_points_mask
        # projected_points = projected_points[idx]
        # _rgb = rgb[idx]/255
        # # write_point_cloud_as_ply('trash/pro_point_cloud.ply', np.concatenate([projected_points, _rgb], axis=1), with_color=True)

        # point_cloud_camera = point_cloud_camera[idx]
        # # write_point_cloud_as_ply('trash/camera_point_cloud.ply', np.concatenate([point_cloud_camera, _rgb], axis=1), with_color=True)

        return visible_points_mask
    
    idx = project_points(calculate_intrinsic_matrix(fovx, fovy, width, height), R, T, xyz)
    return xyz[idx], rgb[idx], camera.image_name

data_path = cmd_args.data_path
split_path = '../data/split.json'

with open(split_path,'r') as f:
    split_dict = json.load(f)
ids_train = split_dict[cmd_args.cate]['train']
ids_eval = split_dict[cmd_args.cate]['eval']
ids_all = ids_train + ids_eval

for scene_id in ids_all:
    # if scene_id != 'scan4':
        # continue
    print(scene_id)
    args_2dgs.source_path = os.path.join(data_path, scene_id)
    scene = Scene(lp_2dgs.extract(args_2dgs), shuffle=False)
    points_path = os.path.join(data_path, scene_id, 'points3d_2.ply')
    points = o3d.io.read_point_cloud(points_path)
    xyz = np.asarray(points.points)
    rgb = np.asarray(points.colors) * 255
    img_patch_dir = os.path.join(data_path, scene_id, 'img_patch')
    os.makedirs(img_patch_dir, exist_ok=True)
    for view_idx in range(len(scene.getTrainCameras())):
        xyz_patch, rgb_patch, img_name = get_scene_img_patch(xyz, rgb, scene, view_idx)
        print(xyz_patch.shape, rgb_patch.shape, img_name)
        img_patch_file_path = os.path.join(img_patch_dir, f'{img_name}.ply')
        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector(xyz_patch)
        points.colors = o3d.utility.Vector3dVector(rgb_patch/255)
        o3d.io.write_point_cloud(img_patch_file_path, points)