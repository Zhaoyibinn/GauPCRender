import blenderproc as bproc
import argparse
import json
import os
import numpy as np
import cv2

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

def add_point_light(energe=3000, location=[-5, -5, 5]):
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_energy(energe)
    # light.set_type("SUN")
    light.set_location(location)


def Fibonacci_grid_sample(num, radius):
    # https://www.jianshu.com/p/8ffa122d2c15
    points = [[0, 0, 0] for _ in range(num)]
    phi = 0.618
    for n in range(num):
        z = (2 * n - 1) / num - 1
        x = np.sqrt(np.abs(1 - z * z)) * np.cos(2 * np.pi * n * phi)
        y = np.sqrt(np.abs(1 - z * z)) * np.sin(2 * np.pi * n * phi)
        points[n][0] = x * radius
        points[n][1] = y * radius
        points[n][2] = z * radius

    points = np.array(points)
    return points


def sphere_angle_sample(num, radius):
    points = []
    for azim in np.linspace(-180, 180, num):
        elev = 60
        razim = np.pi * azim / 180
        relev = np.pi * elev / 180

        center = [0, 0, 0]
        xp = center[0] + np.cos(razim) * np.cos(relev) * radius
        yp = center[1] + np.sin(razim) * np.cos(relev) * radius
        zp = center[2] + np.sin(relev) * radius
        points.append([xp, yp, zp])
    points = np.array(points)
    return points

def sphere_angle_sample_for_video(num, radius):
    points = []
    for indice in range(num):
        azim = indice
        elev = indice / 2
        if 90 < elev <= 180:
            elev = 180 - elev

        razim = np.pi * azim / 180
        relev = np.pi * elev / 180
        center = [0, 0, 0]
        xp = center[0] + np.cos(razim) * np.cos(relev) * radius
        yp = center[1] + np.sin(razim) * np.cos(relev) * radius
        zp = center[2] + np.sin(relev) * radius
        points.append([xp, yp, zp])
    points = np.array(points)
    return points


parser = argparse.ArgumentParser()

parser.add_argument('scene', help="Path to the scene.obj file, should be examples/resources/scene.obj")
parser.add_argument('output_dir', help="Path to where the final files, will be saved, could be examples/basics/basic/output")
parser.add_argument('num', default=100, type=int, help="number of rendering")
parser.add_argument('split', default="train", type=str, help="train, val or test")

args = parser.parse_args()
scene_name = os.path.basename(args.scene)[:-4]



bproc.init()
bproc.renderer.enable_depth_output(False)
objs = bproc.loader.load_obj(args.scene)
obj = objs[0]
# obj = bproc.loader.load_shapenet
# print('---------------------------------------')
# print(len(objs))
# print(obj)
# print('---------------------------------------')

# Scale the 3D model
diag = obj.get_bound_box()    # see blenderproc/python/types/MeshObjectUtility for details
# max_size = max(abs(diag[0]), abs(diag[1]), abs(diag[2]))
max_size = np.max(np.max(diag, 0) - np.min(diag, 0))
scale = 1 / max_size
print("normalize scale:", scale)
obj.set_scale([scale, scale, scale])
poi = bproc.object.compute_poi(objs)
print("poi after scale:", poi)

obj.set_rotation_euler([np.pi/2, 0, 0])
poi = bproc.object.compute_poi(objs)
print("poi after rotation:", poi)

# bbox = obj.get_bound_box()
# # print(bbox)
# x = [k[0] for k in bbox]
# y = [k[1] for k in bbox]
# z = [k[2] for k in bbox]
# print(min(x), min(y), min(z))
# # set the obj all > 0:  if min(x) < 0, then move the distance of abs(min(x))
# set_location = [max(0, -min(x)), max(0, -min(y)), max(0, -min(z))]

set_location = [0.5, 0.5, 0.5] - poi

print("set_location:", set_location)
obj.set_location(set_location)

poi = bproc.object.compute_poi(objs)
print("poi after set location:", poi)

# # save scale and set_location to file
# model_json = {'scale':scale, 'set_location':set_location.tolist()}
# with open(os.path.join(os.path.dirname(args.output_dir), "model_" + args.split + ".json"), "w") as f:
#     json.dump(model_json, f, indent=4)

# define a light and set its location and energy level
# add_point_light(energe=500, location=[-1, 3, 3] + poi)
# add_point_light(energe=300, location=[-1, -1, 3] + poi)
# add_point_light(energe=500, location=[0, 0, -3] + poi)
# add_point_light(energe=800, location=[4, 0, 3] + poi)

# add_point_light(energe=10, location=poi)   # set a light in the center of the obj (calculate the center, [0, 0, 0])

fix_energe = 500
dis = 3
add_point_light(energe=fix_energe, location=[dis, 0, 0] + poi)
add_point_light(energe=fix_energe, location=[-dis, 0, 0] + poi)
add_point_light(energe=fix_energe, location=[0, dis, 0] + poi)
add_point_light(energe=fix_energe, location=[0, -dis, 0] + poi)
# add_point_light(energe=3000, location=[1, 1, 10] + poi)
add_point_light(energe=fix_energe, location=[0, 0, dis] + poi)
add_point_light(energe=fix_energe, location=[0, 0, -dis] + poi)

# !!!!!!!!!IMPORTANT!!!!!!!!!
# CAMERA RESOLUTION 
# bproc.camera.set_resolution(800, 800)
bproc.camera.set_resolution(512, 512)
# !!!!!!!!!IMPORTANT!!!!!!!!!

fov_x, fov_y = bproc.camera.get_fov()    # just to get angle_x. see python/camera/CameraUtility.py for details or changes
# print(fov_x, fov_y)
angle_x = fov_x

# !!!!!!!!!IMPORTANT!!!!!!!!!
# CAMERA DISTANCE 
R = 1.6       # human
# !!!!!!!!!IMPORTANT!!!!!!!!!

if args.split == "test":
    locations = sphere_angle_sample(num=args.num, radius=R)       # for generating test set, and views for videos
    locations += poi
elif args.split == "val":
    locations = [[R, 0, 0], [0, R, 0], [0, 0, R]]
    locations += poi
elif args.split == "video":
    locations = sphere_angle_sample_for_video(num=args.num, radius=R)       # for generating test set, and views for videos
    locations += poi
    print("sample cameras for video:", len(locations))
else:
    locations = Fibonacci_grid_sample(num=args.num, radius=R)       # for generating training set
    locations += poi

# Sample several camera poses
for i in range(args.num):
    # # Sample random camera location above objects
    # location = np.random.uniform([-1, -1, 0], [1, 1, 2])

    if args.split == "val":
        location = locations[args.num - 1 - i]
    else:
        location = locations[i]     # this is normal, the "val" is not.

    # # Sample random camera location around the object
    # location = bproc.sampler.sphere(poi, radius=3, mode="SURFACE")
    # location = bproc.sampler.sphere([0, 0, 0], radius=3, mode="SURFACE")

    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
    # rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # print(rotation_matrix)
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

bproc.renderer.set_output_format(enable_transparency=True)
# render the whole pipeline
data = bproc.renderer.render()

# Collect state of the camera at all frames
cam_states = []
for frame in range(bproc.utility.num_frames()):
    cam_states.append({
        "cam2world": bproc.camera.get_camera_pose(frame),
        "cam_K": bproc.camera.get_intrinsics_as_K_matrix()
    })
# Adds states to the data dict
data["cam_states"] = cam_states

# # write the data to a .hdf5 container
# output_split_dir = os.path.join(args.output_dir, 'hdf5', args.split)
# os.makedirs(output_split_dir, exist_ok=True)
# bproc.writer.write_hdf5(output_split_dir, data)

#存储相机参数到json文件
camera_json = {"camera_angle_x": angle_x, "frames": []}
for i in range(len(cam_states)):
    filename = str(i) + "_" + "colors"
    Rt = cam_states[i]["cam2world"]
    R = Rt[:3, :3]
    T = Rt[:3, 3:4]
    world2cam = np.concatenate((np.concatenate((R.transpose(), -(R.transpose().dot(T))), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
    K = cam_states[i]["cam_K"]
    camera = {"file_path": "./" + args.split + "/" + filename,
                  "rotation": 0,   # not use
                  "camera_intrinsics": K.tolist(),
                  "transform_matrix": Rt.tolist()}

    camera_json["frames"].append(camera)

with open(os.path.join(args.output_dir, "transforms_" + args.split + ".json"), "w") as f:
    json.dump(camera_json, f, indent=4)

# blenderproc run examples/datasets/abc_dataset/main.py  examples/datasets/abc_dataset/00000003/00000003_1ffb81a71e5b402e966b9341_trimesh_002.obj examples/datasets/abc_dataset/output 100 train
# blenderproc vis hdf5 examples/datasets/abc_dataset/output/*.hdf5 --save examples/datasets/abc_dataset/output_rgb/

# extract point cloud
color_save_path = os.path.join(args.output_dir, args.split)
points_save_path = os.path.join(args.output_dir, 'points')
os.makedirs(color_save_path, exist_ok=True)
os.makedirs(points_save_path, exist_ok=True)
# print(data)
# print(data['depth'])
depth_images = data['depth']
color_images = data['colors']
frame_num = bproc.utility.num_frames()
for frame in range(frame_num):
    print(f'{frame}/{frame_num}', end='\r')
    points = bproc.camera.pointcloud_from_depth(depth_images[frame], frame)
    rgba = color_images[frame]
    # image = np.stack([rgba[:, :, 2], rgba[:, :, 1], rgba[:, :, 0]], -1)
    image = np.stack([rgba[:, :, 2], rgba[:, :, 1], rgba[:, :, 0], rgba[:, :, 3]], -1)
    back_bg = np.zeros_like(image[:, :, :3])
    white_bg = np.ones_like(image[:, :, :3])
    # print(np.max(rgba[:, :, 3:]))
    idx = (white_bg * image[:, :, 3:]) > 5
    image = np.where(idx, image[:, :, :3], back_bg)
    cv2.imwrite(os.path.join(color_save_path, f'{frame}_colors.png'), image)
    # print(points.shape)    
    # print(rgba.shape) 
    pc = np.concatenate([points, rgba/255], 2)
    pc = np.reshape(pc, [-1, 7])
    idx = ~np.isnan(np.sum(pc[:,:3], -1))
    pc = pc[idx]  
    np.save(os.path.join(points_save_path, f'{args.split}_{frame}_points.npy'), pc[:, :6])
print('done')