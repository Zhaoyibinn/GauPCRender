import os
import sys
import numpy as np
import cv2 
import open3d as o3d


def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # depth min and max: line 11
    if len(lines) >= 12:
        depth_params = np.fromstring(lines[11], dtype=np.float32, sep=' ')
    else:
        depth_params = np.empty(0)

    return intrinsics, extrinsics, depth_params

old_dtu_path = '[Your DTU Path]'
# the content of this folder should be 'scan2', 'scan4', ...
new_dtu_path = '../data/dtu'
os.makedirs(new_dtu_path, exist_ok=True)

scene_id_list = os.listdir(old_dtu_path)
scene_id_list = [x for x in scene_id_list if x.startswith('scan')]

print(scene_id_list)

for scene_id in scene_id_list:
    print(scene_id) 
    scene_data_path = os.path.join(new_dtu_path, scene_id)
    os.makedirs(scene_data_path, exist_ok=True)

    # points
    points_old_path = os.path.join(old_dtu_path, scene_id, 'points_voxel_downsampled.ply')
    points_new_path = os.path.join(new_dtu_path, scene_id, 'points3d_2.ply')
    cmd = f'cp {points_old_path} {points_new_path}'
    os.system(cmd)
    points_new_path = os.path.join(new_dtu_path, scene_id, 'points3d.ply')
    cmd = f'cp {points_old_path} {points_new_path}'
    os.system(cmd)

    # color 
    color_old_path = os.path.join(old_dtu_path, scene_id, 'images', '3')
    mask_old_path = os.path.join(old_dtu_path, scene_id, 'masks')
    color_ids = os.listdir(color_old_path)
    color_ids = [x[:-4] for x in color_ids if x.endswith('.jpg')]
    color_new_path = os.path.join(new_dtu_path, scene_id, 'color')
    os.makedirs(color_new_path, exist_ok=True)
    for color_id in color_ids:
        color_old_file = os.path.join(color_old_path, f'{color_id}.jpg')
        mask_old_file = os.path.join(mask_old_path, f'{color_id}.png')
        color_img = cv2.imread(color_old_file)
        mask_img = cv2.imread(mask_old_file, cv2.IMREAD_GRAYSCALE)/255
        mask_img = mask_img.astype(bool)
        color_img = color_img * mask_img[:, :, np.newaxis]
        cv2.imwrite(os.path.join(color_new_path, f'{color_id}.jpg'), color_img)

    # intrinsics and extrinsics
    intrinsics_new_path = os.path.join(new_dtu_path, scene_id, 'intrinsics_color.txt')
    intrinsics = None
    pose_new_path = os.path.join(new_dtu_path, scene_id, 'pose')
    os.makedirs(pose_new_path, exist_ok=True)
    for color_id in color_ids:
        cam_file = os.path.join(old_dtu_path, scene_id, 'cams', f'{color_id}_cam.txt')
        _intrinsics, _extrinsics, _ = read_cam_file(cam_file)
        if intrinsics is None:
            intrinsics = np.eye(4)
            intrinsics[:3, :3] = _intrinsics
            np.savetxt(intrinsics_new_path, intrinsics)
        pose = np.linalg.inv(_extrinsics)
        pose_file_path = os.path.join(pose_new_path, f'{color_id}.txt')
        np.savetxt(pose_file_path, pose)
    # break