# DOPH: Differentiable Occupancy and Mesh Morphing
Differentiable Tool for Mesh2Occupancy &amp; Mesh2SDF 

![Example of a mesh from ShapeNet](data_example/Doph.png)


This repository contains the code for the paper "[Differentiable Voxelization and Mesh Morphing](https://arxiv.org/abs/2407.11272)"  by [Yihao Luo](https://github.com/Luo-Yihao) et al. DOPH is a differentiable tool for mesh to occupancy and mesh to SDF conversion. Unlike previous tools implemented on the CPU used in [Mesh2SDF](https://pypi.org/project/mesh2sdf/) and [DOGN](https://github.com/microsoft/DualOctreeGNN) , DOPH is a differentiable tool that can be used to convert mesh to occupancy and SDF in a differentiable way, which can be directly integrated into the deep learning framework. Meanwhile, with GPU acceleration, DOPH can extract occupancy and SDF in **arbitrary resolution** in real-time, even for **non-watertight**，**non-manifold**, **self-intersecting** and **combined** meshes.

## Installation

DOPH only depends on ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-blue.svg) and ![PyTorch3D](https://img.shields.io/badge/PyTorch3D-0.7.5-blue.svg). One can install the dependencies by running the following command:
```setup
pip install -r requirements.txt
```

## Usage
we provide a tutorial `demo_mesh2occp.ipynb` to show how to use DOPH to convert a mesh to occupancy and SDF.
```python
import torch
from ops import occupancy, signed_distance_field, normalize_mesh
from pytorch3d.io import load_objs_as_meshes

# Load a mesh
mesh = load_objs_as_meshes("data_example/Bull.obj")
# normalize the mesh
mesh = normalize_mesh(mesh)

# Query the occupancy
q = torch.rand(100, 3)*2.-1.
q.requires_grad = True

# Get the occupancy
occp = occupancy(mesh, q)
print(occp)
# Get the SDF with gradient
sdf = signed_distance_field(mesh, q, allow_grad=True)
print(sdf)
print(sdf.grad)
```

## Citation
If you find this code useful for your research, please consider citing the following paper:
```
@article{lou2024doph,
    title={Differentiable Voxelization and Mesh Morphing},
    author={Yihao Luo, Yikai Wang,Zhengrui Xiang,
    Yuliang Xiu, Guang Yang, ChoonHwai Yap},
    journal={arXiv preprint arXiv:2407.11272},
    year={2024}} 
```
