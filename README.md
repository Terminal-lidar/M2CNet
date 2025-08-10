# M2CNet
This repository contains the official implementation of the paper:
> **M2CNet: LiDAR 3D Semantic Segmentation Based on Multi-level Multi-view Cross-attention Fusion for Autonomous Vehicles**\
> Mu Zhou, Rui Xiang, Nan Du, Wei He*, Zhao Jiang\
> (Accepted by *IEEE Transactions on Vehicular Technology*)
## Abstract:
<p align="justify">
Light Detection and Ranging (LiDAR) point cloud semantic segmentation is an essential task for autonomous vehicles,
enabling precise environment perception for safe navigation. Existing point-based and voxel-based methods face limitations
related to computational costs and quantization errors, especially dealing with large-scale point clouds. In contrast, 2D projection
methods, including Range View and Bird’s Eye View, have gained popularity for their balance of real-time performance and
accuracy. However, most projection approaches overlook critical issues arising from projection, including 1) inter-class occlusion,
2) boundary-blurring, and 3) inadequate multi-scale feature extraction. To address these issues, we propose a Multi-level
Multi-view Cross-attention Fusion Network (M2CNet), which fully leverages complementary information across different views.
First, we employ parallel encoder-decoder networks to extract semantic information within a multi-task segmentation framework.
Given the high correlation between views, we incorporate a Multiview Feature Cross Fusion Module (MFCFM) to facilitate crossview
feature association, capturing potential complementarities to mitigate information loss from single-view projections. Then,
we place a Multi-cascaded Enhanced Semantic Context Atrous Spatial Pyramid Pooling (MESCASPP) module between the encoder
and decoder to capture multi-scale contextual information, enhancing pixel relationships effectively. Finally, the Kernel Point
Convolution (KPConv) is employed to refine point-wise features at the output level. The proposed method is evaluated on urban
datasets SemanticKITTI and nuScenes (achieving mIoU scores of 65.7% and 78.9% respectively), the off-road dataset RELLIS-3D (44.6%), 
and a real-world test set. Experimental results demonstrate that our method surpasses representative methodsin both effectiveness and scene adaptability.
</p>

## Framework:
![architecture](https://github.com/user-attachments/assets/106ba62b-08d9-448e-844a-165fdb1b85f1)
<img width="6856" height="3006" alt="MFCFM" src="https://github.com/user-attachments/assets/dbaa436c-1b38-4e98-8fb3-9b1163825eba" />
## Installation
Our environment: Ubuntu 20.04, CUDA 12.2, NVIDIA A40 GPU x 2

Create a conda env with
```bash
conda env create -f environment.yml
```
## Dataset:
- **Download SemanticKITTI** from [official website](http://www.semantic-kitti.org/dataset.html)
- **Download nuScenes** from [official website](https://www.nuscenes.org/nuscenes) 
- **Download RELLIS-3D** from [official website](https://pan.baidu.com/s/1akqSm7mpIMyUJhn_qwg3-w?pwd=4gk3 )
```bash
   dataset
   └── SemanticKITTI/nuScenes/RELLIS-3D
       └── sequences
           ├── 00
                └──velodyne
                └──labels
           ├── 01
           └── ...
```
> Note: By default, the data formats of these datasets are **not identical**. To ensure consistency in training and evaluation, **all datasets are converted to the SemanticKITTI format** before use.

## Usage:
### Train
1.Download the [pretrained resnet model](https://drive.google.com/file/d/1I85xLRwUMIeW_7BvdZ4uZ0Lm4j3zxLT1/view?usp=sharing) to pretrained/resnet34-333f7ec4.pth.\
2.Train on SemanticKITTI/nuScenes/RELLIS-3D datasets by:
```bash
./scripts/start.sh
```
### Infer and Eval
- Infer on SemanticKITTI/nuScenes/RELLIS-3D datasets by:
```bash
./scripts/infer.sh
```
- Eval for valid sequences:\
The evaluation of the datasets in our experiments is conducted using the [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api) provided by the SemanticKITTI benchmark.

## Acknowledgments
We would like to thank the developers of the following open-source projects for providing valuable code and tools that supported this work: [GFNet](https://github.com/haibo-qiu/GFNet.git), [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext.git), 
[PolarNet](https://github.com/edwardzhou130/PolarSeg.git), and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer.git). Thanks the contributors of these repositories!
