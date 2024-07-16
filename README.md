# DOPH: Differentiable Occupancy and Mesh Morphing
Differentiable Tool for Mesh2Occupancy &amp; Mesh2SDF 

![Example of a mesh from ShapeNet](data_example/Doph.png)


This repository contains the code for the paper "[Differentiable Voxelization and Mesh Morphing]((https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789))"  by [Yihao Luo](https://yuewang-cv.github.io/) et al. DOPH is a differentiable tool for mesh to occupancy and mesh to SDF conversion. Unlike previous tools implemented on the CPU used in [Mesh2SDF](https://pypi.org/project/mesh2sdf/) and [DOGN](https://github.com/microsoft/DualOctreeGNN) , DOPH is a differentiable tool that can be used to convert mesh to occupancy and SDF in a differentiable way, which can be directly integrated into the deep learning framework. Meanwhile, with GPU acceleration, DOPH can extract occupancy and SDF in **arbitrary resolution** in real-time, even for **non-watertight**，**non-manifold**, **self-intersecting** and **combined** meshes.

## Installation

DOPH only depends on ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-blue.svg) and ![PyTorch3D](https://img.shields.io/badge/PyTorch3D-0.7.5-blue.svg). One can install the dependencies by running the following command:
```setup
pip install -r requirements.txt
```

## Usage
we provide a tutorial `demo_mesh2occp.ipynb` to show how to use DOPH to convert a mesh to occupancy and SDF.

## Citation
If you find this code useful for your research, please consider citing the following paper:
```
@article{lou2024doph,
    title={Differentiable Voxelization and Mesh Morphing},
    author={Yihao Luo, Yikai Wang,Zhengrui Xiang,
    Yuliang Xiu, Guang Yang, ChoonHwai Yap},
    journal={arXiv preprint arXiv:1234.56789},
    year={2024}} 
```
