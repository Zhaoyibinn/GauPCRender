import os
import cv2
import shutil
from tqdm import tqdm

root_path = "data/dtu_colmap_3_many"
out_root_path  = "data/dtu_colmap_3_many_all"

scene_list = os.listdir(os.path.join(root_path))
for scene_name in tqdm(scene_list):
    scene_path = os.path.join(root_path,scene_name)


    idx_list = os.listdir(os.path.join(scene_path,'sparse'))

    for idx in idx_list:
        if idx == '0':
            continue
        # print("idx")
        img_idxs = idx.split("_")

        save_scene_path = os.path.join(out_root_path,scene_name+"_"+idx)
        os.makedirs(os.path.join(out_root_path,scene_name+"_"+idx,'images'),exist_ok=True)
        os.makedirs(os.path.join(out_root_path,scene_name+"_"+idx,'sparse/0'),exist_ok=True)

        img_root_path = os.path.join(scene_path,'images')
        img_names = []
        for img_idx in img_idxs:
            img_names.append(os.path.join(img_root_path,f"{int(img_idx):04d}.png"))
            for img_name in img_names:
                shutil.copy(img_name,os.path.join(save_scene_path,'images',os.path.basename(img_name)))

        sparse_colmap_path = os.path.join(scene_path,'sparse',idx)
        sparse_colmap_path_out = os.path.join(save_scene_path,'sparse/0')
        shutil.rmtree(sparse_colmap_path_out)
        shutil.copytree(sparse_colmap_path,sparse_colmap_path_out)
        





print("end")