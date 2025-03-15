import sys
import os
import numpy as np
#import cv2
import open3d as o3d

def write_point_cloud_as_ply(path, point_cloud, with_color=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    if with_color:
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:])
    o3d.io.write_point_cloud(path, pcd)

def write_vert_face_as_mesh(path, vert, face):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vert)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, mesh)

def read_mesh_as_points(path, sample_num):
    mesh = o3d.io.read_triangle_mesh(path)
    points = mesh.sample_points_uniformly(sample_num)
    points = np.asarray(points.points)
    return points