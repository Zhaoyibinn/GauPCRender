import os
import open3d as o3d 
import argparse
import numpy as np
import json 
from tqdm import tqdm
from plyfile import PlyData, PlyElement

paser = argparse.ArgumentParser()
paser.add_argument('data_path', type=str)
paser.add_argument('point_num', type=int)
args = paser.parse_args()

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

data_path = os.path.join(args.data_path, 'points')
files = os.listdir(data_path)

points = []

for file_name in tqdm(files):
    file_path = os.path.join(data_path, file_name)
    if os.path.isfile(file_path) and file_name.endswith('_points.npy'):
        p = np.load(file_path)
        points.append(p)

points = np.concatenate(points, 0)
np.random.shuffle(points)
points = points[:args.point_num]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)
# pcd = pcd.farthest_point_down_sample(args.point_num)

xyz = points[:, :3]
rgb = points[:, 3:] * 255
storePly(os.path.join(args.data_path, 'points3d.ply'), xyz, rgb)
# o3d.io.write_point_cloud(os.path.join(args.data_path, 'points3d.ply'), pcd)


'''
python fuse_points.py /home/mcf/data2/works/point23dgs/data/shapenet_color/03001627/3fdc09d3065fa3c524e7e8a625efb2a7 20000
'''