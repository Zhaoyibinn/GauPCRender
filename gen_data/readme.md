# Data Generation
This directory contains the scripts to generate the data for training the GauPCRender model.

## Object Categories
The object categories are:
- car (ShapeNet)
- chair (ShapeNet)
- shoe (GSO)
- human (THuman2.0)

Download [ShapeNet](https://shapenet.org/), [GSO](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) and [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset) to your local machine. You can use `download_GSO.py` and `extract_GSO_Shoe.py` to download and extract GSO-Shoe dataset. 

Install [BlenderProc](https://github.com/DLR-RM/BlenderProc) and trimesh to your environment.
```
pip install blenderproc trimesh
```
Run the following commands to generate the data for each object category. 
You must modify the original dataset paths in the scripts `create_{category}_dataset.py`.
You can also modify the camera distance and resolution in the scripts `bproc_script_{category}.py` for higher quality renders but this may take longer to generate and cost more GPU memory while training.
#### Car 
```
python create_car_dataset.py
```
#### Chair 
```
python create_chair_dataset.py
```
#### Shoe 
```
python create_shoe_dataset.py
```
#### Human 
```
python create_human_dataset.py
```

## Scene Categories
The scene categories are:
- scannet (ScanNet)
- dtu (DTU) only for evaluation

Download [ScanNet](http://www.scan-net.org/)(the frames 2k version and meshes) and [DTU](https://1drv.ms/u/c/747194122a3acf02/EdwjDcTXBwpAmyKqDEqjsZMBiUoxXpJ2o1QCYdt8WmMGOA?e=nvceS7)(Offered by [PFGS](https://github.com/Mercerai/PFGS)) to your local machine.

#### Scannet
Prepare the ScanNet.
```
python prepare_scannet.py
ln -s [path to scannet frames25k]  ../data/scannet 
# you can create a soft link or directly copy the scannet-frames-25k to the data directory
python prepare_scene_img_patch.py scannet ../data/scannet
```

#### DTU
Prepare the DTU.
```
python prepare_dtu.py
python prepare_scene_img_patch.py dtu ../data/dtu
```
