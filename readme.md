# GauPCRender: Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians

This is the implementation of the paper **Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians**(CVPR 2025).

[[paper link(coming soon)](https://cvpr.thecvf.com/Conferences/2025)]

### Abstract
```
Current learning-based methods predict NeRF or 3D Gaussians from point clouds to achieve photo-realistic rendering but still depend on categorical priors, dense point clouds, or additional refinements. Hence, we introduce a novel point cloud rendering method by predicting 2D Gaussians from point clouds. Our method incorporates two identical modules with an entire-patch architecture enabling the network to be generalized to multiple datasets. The module normalizes and initializes the Gaussians utilizing the point cloud information including normals, colors and distances. Then, splitting decoders are employed to refine the initial Gaussians by duplicating them and predicting more accurate results, making our methodology effectively accommodate sparse point clouds as well. Once trained, our approach exhibits direct generalization to point clouds across different categories. The predicted Gaussians are employed directly for rendering without additional refinement on the rendered images, retaining the benefits of 2D Gaussians. We conduct extensive experiments on various datasets, and the results demonstrate the superiority and generalization of our method, which achieves SOTA performance.
```

### Citation
```
@article{ma2025gaupcrender,
  title={Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians},
  author={Ma, Changfeng and Bi, Ran and Guo, Jie and Wang, Chongjun and Guo, Yanwen},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Getting Started
### Environment
```
Python          3.8
PyTorch         2.0.0
CUDA            11.7
Ubuntu          18.04
```
You can try other versions of the above environment. If you encounter any problems, please let us know by opening an issue. We will try our best to help you.

### Install
Clone the repository.
```
git clone https://github.com/murcherful/GauPCRender.git
cd GauPCRender
```
We strongly recommend using [Anaconda](https://www.anaconda.com/) to manage your Python environment.
```
conda create -n GauPCRender python=3.8
conda activate GauPCRender
```

Then, install the required packages:
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1

cd ./util/pointnet2_ops_lib/
python setup.py install
cd ../..

cd ./util/plane_gs/submodules
cd diff-surfel-rasterization
python setup.py install
cd ../simple-knn
python setup.py install
cd ../../../..

pip install opencv-python lipis plyfile matplotlib open3d==0.11.0 scikit-learn addict pandas rich
```

### Datasets and Weights
You can simply download the datasets and weights from here:
- [NJU Box](https://box.nju.edu.cn/d/c77a6eb38f1849c3bafd/)(code: gaupcrender)
- [Baidu Disk](https://pan.baidu.com/s/1kcv43Sz7lvVzy4AFaZeQKg)(code: 3jsh)

You can also download the orginal datasets and generate the processed datasets following the instructions in `gen_data`.
The datasets and weights directory should be organized as follows:
```
GauPCRender
├── data
│   ├── car
|   |   ├── 1a0bc9ab92c915167ae33d942430658c
|   |   ├── 1a56d596c77ad5936fa87a658faf1d26
|   |   ├── ...
|   ├── chair
|   ├── ...
|   └── split.json
├── weights
│   ├── car_p20k_s4_b8_i8
│   |   └── weights_480.pkl
│   ├── car_p20k_s4_b8_i8_p
|   ├── scannet_p-1_s4_b8_i8_p
|   ├──...
```
Here, `car_p20k_s4_b8_i8` means the entire model trained on car dataset with 20k points, split number is 4, batch size is 8, and image number is 8. `_p` means the patch model.
`p-1` means the point number is -1 that is all points are used for training. 
### Training
Train the Entire Model for car, chair, shoe and human. 
You can check the arguements in `main.py` for more details.
```
python main.py train exp_car --cate car --gpu 0
```
Train the Patch Model for car, chair, shoe and human once the entire model is trained.
```
python main.py train exp_car_patch --cate car --gpu 0 --patch --entire_train_name exp_car --entire_restore 480
```
Fine-tune the Patch Model for scannet.
```
python main.py train exp_scene_patch --gpu 0 --cate scannet --scene_train_name exp_car_patch --scene_restore 480 --point_number 100_000
```
The training script will save the checkpoints in the `weights` directory. 
If meeting `CUDA_OUT_OF_MEMORY`, you can reduce the batch size, image number or split number.
You can increase these arguments to improve the performance if you have more GPU memory.

### Evaluation
Evaluate the Entire Model for car, chair, shoe and human.
```
python main.py eval exp_car --cate car --gpu 0 --restore 480 [--save_img_merge] [--save_img_file] [--save_gs_file]
```
Evaluate the Patch Model for car, chair, shoe and human.
```
python main.py eval exp_car_patch --cate car --gpu 0 --restore 480 --patch
```
Evaluate the Fine-tuned Patch Model for scannet.
```
python main.py eval exp_scene_patch --gpu 0 --cate scannet --restore 233 --point_number 100_000
```
You can change the category and point number to evaluate the generalization performance.
`--save_img_merge` will save the merged image. `--save_img_file` will save the individual images. `--save_gs_file` will save the predicted Gaussians and you can use [3DGS Viewer](https://github.com/graphdeco-inria/gaussian-splatting) to visualize them.
You will find the evaluation results in the `weights` directory.

### License
This project is licensed under the [MIT License](https://mit-license.org/).

### Acknowledgments
We thank the following repositories for their help:

- [PointNet++ Operations Library](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib)

- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)

- [PFGS](https://github.com/Mercerai/PFGS)

- [TriVol](https://github.com/dvlab-research/TriVol)

- [BlenderProc](https://github.com/DLR-RM/BlenderProc)
