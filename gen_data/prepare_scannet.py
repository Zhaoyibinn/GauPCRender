import os 
import sys 
import trimesh
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import json


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


point_data_path = '[Your ScanNet Meshes Path]'
img_data_path = '[Your ScanNet Frame25k Path]'
# the content of these two directories should be 'scene0000_00', 'scene0000_01',...
split_path = '../data/split.json'

with open(split_path,'r') as f:
    split_dict = json.load(f)
ids_train = split_dict['scannet']['train']
ids_eval = split_dict['scannet']['eval']
ids_all = ids_train + ids_eval

# N = 1
N = len(ids_all)
for i in tqdm(range(N)):
    scene_id = ids_all[i]

    mesh_path = os.path.join(point_data_path, scene_id, f'{scene_id}_vh_clean_2.ply')
    mesh = trimesh.load(mesh_path, force='mesh')
    points, idx, colors = trimesh.sample.sample_surface(mesh, 100_000, sample_color=True)

    points_path = os.path.join(img_data_path, scene_id, 'points3d.ply')
    storePly(points_path, points, colors[:, :3])

    points_patch_2 = os.path.join(img_data_path, scene_id, 'points3d_2.ply')
    cmd = f'cp {mesh_path} {points_patch_2}'
    # print(cmd)
    os.system(cmd)
    # break