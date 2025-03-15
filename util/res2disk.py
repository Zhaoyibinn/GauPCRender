import os 
import sys 
import numpy as np
import open3d as o3d 
from plyfile import PlyData, PlyElement
from tqdm import tqdm

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def quaternion_to_rotation_matrix(quaternion):  
    """  
    Convert a quaternion to a rotation matrix.  
  
    Parameters:  
    quaternion (list or tuple): A quaternion in the form [w, x, y, z]  
  
    Returns:  
    np.ndarray: A 3x3 rotation matrix  
    """  
    w, x, y, z = quaternion  
  
    Nq = w*w + x*x + y*y + z*z  
    if Nq < np.finfo(float).eps:  # If norm is zero, return identity  
        return np.eye(3)  
  
    s = 2.0 / Nq  
    X = x * s  
    Y = y * s  
    Z = z * s  
    wX = w * X; wY = w * Y; wZ = w * Z;  
    xX = x * X; xY = x * Y; xZ = x * Z;  
    yY = y * Y; yZ = y * Z; zZ = z * Z;  
  
    rotation_matrix = np.array([  
        [1.0-(yY+zZ), xY-wZ, xZ+wY],  
        [xY+wZ, 1.0-(xX+zZ), yZ-wX],  
        [xZ-wY, yZ+wX, 1.0-(xX+yY)]  
    ], dtype=np.float64)  
  
    return rotation_matrix  

def res2disk(xyz, rgb, scale, opacity, rot):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    sx = scale[:, 0]
    sy = scale[:, 1]

    o = opacity

    q1 = rot[:, 0]
    q2 = rot[:, 1]
    q3 = rot[:, 2]
    q4 = rot[:, 3]

    sx = np.exp(sx)
    sy = np.exp(sy)
    o = sigmoid(o)

    N = x.shape[0]

    final_res = o3d.geometry.TriangleMesh()
    for i in tqdm(range(N)):
        # if o[i] < 0.2:
        #     continue
        gaussion = o3d.geometry.TriangleMesh.create_sphere(resolution=4)
        gaussion_v = np.asarray(gaussion.vertices)
        gaussion_v = gaussion_v * np.array([sx[i], sy[i], 0.0001])
        # gaussion_v = gaussion_v * np.array([sx[i], sy[i], 0.01])
        rm = quaternion_to_rotation_matrix([q1[i], q2[i], q3[i], q4[i]])
        rm = rm.T
        gaussion_v = np.matmul(gaussion_v, rm)
        gaussion_v = gaussion_v + np.array([x[i], y[i], z[i]])
        gaussion.vertices = o3d.utility.Vector3dVector(gaussion_v)
        color = [1, 1-o[i], 1-o[i]]
        gaussion.paint_uniform_color(color)
        final_res += gaussion
    
    return final_res

if __name__ == '__main__':

    ply_file = sys.argv[1]
    plydata_gt = PlyData.read(ply_file)
    print(plydata_gt.elements[0].properties)

    x  = plydata_gt.elements[0].data['x']
    y  = plydata_gt.elements[0].data['y']
    z  = plydata_gt.elements[0].data['z']

    r  = plydata_gt.elements[0].data['f_dc_0']
    g  = plydata_gt.elements[0].data['f_dc_1']
    b  = plydata_gt.elements[0].data['f_dc_2']

    sx = plydata_gt.elements[0].data['scale_0']
    sy = plydata_gt.elements[0].data['scale_1']

    o  = plydata_gt.elements[0].data['opacity']

    q1 = plydata_gt.elements[0].data['rot_0']
    q2 = plydata_gt.elements[0].data['rot_1']
    q3 = plydata_gt.elements[0].data['rot_2']
    q4 = plydata_gt.elements[0].data['rot_3']

    sx = np.exp(sx)
    sy = np.exp(sy)
    o = sigmoid(o)

    N = x.shape[0]

    final_res = o3d.geometry.TriangleMesh()
    for i in tqdm(range(N)):
        # if o[i] < 0.2:
        #     continue
        gaussion = o3d.geometry.TriangleMesh.create_sphere(resolution=6)
        gaussion_v = np.asarray(gaussion.vertices)
        gaussion_v = gaussion_v * np.array([sx[i], sy[i], 0.0001])
        # gaussion_v = gaussion_v * np.array([sx[i], sy[i], 0.01])
        rm = quaternion_to_rotation_matrix([q1[i], q2[i], q3[i], q4[i]])
        rm = rm.T
        gaussion_v = np.matmul(gaussion_v, rm)
        gaussion_v = gaussion_v + np.array([x[i], y[i], z[i]])
        gaussion.vertices = o3d.utility.Vector3dVector(gaussion_v)
        color = [1, 1-o[i], 1-o[i]]
        gaussion.paint_uniform_color(color)
        final_res += gaussion


    o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(ply_file), 'disk_mesh.ply'), final_res)