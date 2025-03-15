import os
import time
import json
import shutil
import random

# set GPU id if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def create_shapenet_dataset(dataset_dir, shape_id, output_dataset_dir):
    print("shape_id:", shape_id)
    obj_path = os.path.join(dataset_dir, shape_id, 'models/model_normalized.obj')

    output_dir = os.path.join(output_dataset_dir, shape_id)
    os.makedirs(output_dir, exist_ok=True)

    print("Processing " + obj_path)

    train_num = 24
    val_num = 3

    command = f'blenderproc run bproc_script_car.py {obj_path} {output_dir} {train_num} train'
    print(command)
    os.system(command)
    time.sleep(2)

    command = f'blenderproc run bproc_script_car.py {obj_path} {output_dir} {val_num} val'
    print(command)
    os.system(command)
    time.sleep(2)

    command = f'mv {os.path.join(output_dir, "transforms_val.json")} {os.path.join(output_dir, "transforms_test.json")}'
    print(command)
    os.system(command)
    time.sleep(2)

    command = f'python fuse_points.py {output_dir} 100000'
    print(command)
    os.system(command)
    time.sleep(2)

    save_points_dir_path = os.path.join(output_dir, 'points')
    command = f'rm -r {save_points_dir_path}'
    print(command)
    os.system(command)
    time.sleep(2)

dataaset_path = '[Your ShapeNet path]/02958343'
output_path = '../data/car'
split_path = '../data/split.json'

os.makedirs(output_path, exist_ok=True)

with open(split_path,'r') as f:
    split_dict = json.load(f)
ids_train = split_dict['car']['train']
ids_eval = split_dict['car']['eval']
ids_all = ids_train + ids_eval

N = len(ids_all)
for i, shape_id in enumerate(ids_all):
    print(f'------------------------- {i} / {N} --------------------')
    create_shapenet_dataset(dataaset_path, shape_id, output_path)
    # break