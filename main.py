import argparse 
import logging
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser() 
parser.add_argument('op', choices=['train', 'eval'], help='operation to perform')
parser.add_argument('train_name', type=str, help='Training name.')
parser.add_argument('--cate', choices=['chair', 'car', 'shoe', 'scannet', 'human', 'dtu'], default='car', help='Category.')
parser.add_argument('--entire_train_name', default=None, type=str, help='Training name of the Entire Model.')
parser.add_argument('--scene_train_name', default=None, type=str, help='Training name of the restored model when training on scenes.')

# model options 
parser.add_argument('--image_number', choices=[1, 2, 4, 8], default=8, type=int, help='Number of images while training gaussians.')
parser.add_argument('--point_number', choices=[-1, 2_000, 10_000, 20_000, 40_000, 80_000, 100_000, 200_000, 300_000, 400_000, 500_000, 1000_000], default=20_000, type=int, help='Number of input points.')
parser.add_argument('--split_number', choices=[1, 2, 4, 8], default=4, type=int, help='Number of splits of gaussians.')
parser.add_argument('--sh', choices=[0, 1, 2], default=2, type=int, help='Number of SH coefficients.')
parser.add_argument('--patch', action='store_true', help='Use patch.')
parser.add_argument('--patch_point_number', default=2048, type=int, help='Number of points in each patch.')

# training options 
parser.add_argument('--gpu', default='0', help='GPU device.')
parser.add_argument('--save_epochs', nargs="+", type=int, default=[480])
parser.add_argument('--save_path', default='weights', help='Save path.')
parser.add_argument('--data_root', default='data', help='Data root.')
parser.add_argument('--max_epoch', default=480, type=int, help='Max training epoch.')
parser.add_argument('--batch_size', choices=[1, 2, 4, 8], default=8, type=int, help='Batch size.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate.')
parser.add_argument('--restore', default=None, type=int, help='Restore from checkpoint.')
parser.add_argument('--entire_restore', default=None, type=int, help='Restore epoch of the Entire Model.')
parser.add_argument('--scene_restore', default=None, type=int, help='Restore epoch of the restored model when training on scenes.')
parser.add_argument('--skip_check', action='store_true', help='Skip check.')

# evaluation options 
parser.add_argument('--save_img_merge', action='store_true', help='Save a merged img.')
parser.add_argument('--save_img_file', action='store_true', help='Save rendered img file.')
parser.add_argument('--save_gs_file', action='store_true', help='Save gs file and view in 3DGS Viewer.')
parser.add_argument('--ignore_missing', action='store_true', help='Ignore missing area when evaluating scenes.')


parser.add_argument('--data_path', default='')
parser.add_argument('--resolution', type=int, default=-1)
cmd_args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s:%(levelname)s:    %(message)s",
                    datefmt="%Y-%m-%d-%H-%M-%S")

import os 
logging.info(f'Using GPU {cmd_args.gpu}') 
cuda_index = cmd_args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index

import sys 
sys.path.append('util')
sys.path.append('util/plane_gs')

from network.train_util import train 
from network.eval_util import eval 
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

SCENE_CATES = ['scannet', 'dtu']

args_2dgs, lp_2dgs, op_2dgs, pp_2dgs = get_2dgs_args()


lp_2dgs._resolution = cmd_args.resolution
args_2dgs.resolution = cmd_args.resolution

lp_2dgs.sh_degree = cmd_args.sh
args_2dgs.sh_degree = cmd_args.sh
args_2dgs.data_device = 'cpu'
args_2dgs.pre_data_image_num = cmd_args.image_number
cmd_args.args_2dgs = args_2dgs
cmd_args.lp_2dgs = lp_2dgs
cmd_args.op_2dgs = op_2dgs
cmd_args.pp_2dgs = pp_2dgs

if cmd_args.cate in SCENE_CATES:
    args_2dgs.pre_data_image_num = 1
    cmd_args.patch = True
    cmd_args.scene_cate = True
    # cmd_args.scene_cate = False
else:
    cmd_args.scene_cate = False

def main():
    if cmd_args.op == 'train':
        train(cmd_args, logging)
    if cmd_args.op == 'eval':
        eval(cmd_args, logging)

if __name__ == '__main__':
    main()

'''
# train an Entire Model 
python main.py train exp_car_default --cate car --gpu 2 
python main.py eval car_p20k_s4_b8_i8 --cate car --gpu 3 --restore 480 --save_img_merge --save_img_file --save_gs_file

# train a Patch Model 
python main.py train exp_car_default_patch --cate car --gpu 3 --patch --entire_train_name car_p20k_s4_b8_i8 --entire_restore 480
python main.py eval car_p20k_s4_b8_i8_p --cate car --gpu 4 --restore 480 --patch --save_img_merge --save_img_file --save_gs_file 

# train a Patch Model on Scenes
python main.py train exp_car_default_patch_scene --gpu 4 --cate scannet --scene_train_name car_p20k_s4_b8_i8_p --scene_restore 480 --point_number -1
python main.py eval scannet_p-1_s4_b8_i8_p --gpu 5 --cate scannet --restore 233 --point_number -1 --save_img_merge --save_img_file --ignore_missing
'''