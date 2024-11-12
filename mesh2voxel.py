import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d
from pytorch3d import _C
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes, knn_points, estimate_pointcloud_normals, knn_gather, cubify
from pytorch3d.loss.point_mesh_distance import point_mesh_edge_distance, point_mesh_face_distance
import trimesh

from torch.utils.data import DataLoader, Dataset, DistributedSampler  
from torch.nn.parallel import DistributedDataParallel as DDP  

# # from pytorch3d.structures import Meshes

# from .utils import one_hot_sparse
from ops.mesh_geometry import *
# import pymeshlab as ml


def mesh2voxel(mesh, voxel_size=256, spacial_range=1.0, mode='occp', device_ids=[0], max_query_point_batch_size=2000, if_duplicate=True):
    """
    Convert a mesh to a voxel grid

    Args:
        mesh (Meshes): the input mesh: Trimesh or Pytorch3d Meshes or Mesh Path
        voxel_size (int): the resolution of the voxel grid
        spacial_range (float): the range of the voxel grid
        mode (str): occp = 2000 the mode of the voxel grid, 'occp' or 'sdf'
        device (str): the device to run the voxelization
        max_query_point_batch_size (int): default = 2000, the maximum number of points to query the mesh
        if_duplicate (bool): default = True, if the mesh is non-closed, duplicate and flip the non-closed faces
    
    Returns:
        field (torch.Tensor): the voxel grid # (1, voxel_size, voxel_size, voxel_size) #(Z, Y, X)

    Note:
        Mesh is normalized into bbox [-1, 1]*spacial_range
        As torch convention, the voxel grid is in the order of (Z, Y, X). If anyone wants to visualize the voxel grid, please consider the permutation of the tensor into (X, Y, Z)

    """
    if isinstance(mesh, str):
        mesh_path = mesh
        trimesh_tem = trimesh.load(mesh_path, force='mesh')
    elif isinstance(mesh, trimesh.Trimesh):
        trimesh_tem = mesh
    elif isinstance(mesh, Meshes):
        trimesh_tem = trimesh.Trimesh(vertices=mesh.verts_packed().detach().cpu().numpy(), faces=mesh.faces_packed().detach().cpu().numpy())

    if len(device_ids) > 0:
        device = torch.device(f"cuda:{device_ids[0]}")
    else:
        device = torch.device("cpu")

    component_indx = trimesh.graph.connected_component_labels(trimesh_tem.face_adjacency)
    component_indx = torch.from_numpy(component_indx).to(device)


    mesh_tem = Meshes(verts=[torch.from_numpy(trimesh_tem.vertices).float()], 
                    faces=[torch.from_numpy(trimesh_tem.faces).long()]).to(device)

    mesh_tem = normalize_mesh(mesh_tem) # normalize the mesh to fit in the unit cube 

    #### Fipped duplication

    ## duplicate & flip the non-closed faces 

    


    # new_mesh = Meshes(verts=[mesh_verts_new], faces=[face_new])

    # merge the old and new mesh
    

    # voxelizer = Differentiable_Voxelizer(bbox_density=128)

    sample_size = voxel_size

    meshbbox = mesh_tem.get_bounding_boxes()[0]*spacial_range

    

    length = (meshbbox[:,1] - meshbbox[:,0]).max()

    mesh_verts_new = mesh_tem.verts_packed().clone() - mesh_tem.verts_normals_packed()*1e-2*length/2

    faces_new = mesh_tem.faces_packed().clone()

    faces_new = faces_new[:,[0,2,1]]


    new_mesh = Meshes(verts=[mesh_verts_new], faces=[ faces_new])

    coordinates_downsampled = torch.stack(torch.meshgrid(torch.linspace(meshbbox[0,0], meshbbox[0,1], sample_size),
                                                            torch.linspace(meshbbox[1,0], meshbbox[1,1], sample_size),
                                                            torch.linspace(meshbbox[2,0], meshbbox[2,1], sample_size)), dim=-1)

    coordinates_downsampled = coordinates_downsampled.view(-1, 3).to(device)

    n_points = coordinates_downsampled.shape[0]


    if len(device_ids) < 2:
        ## single gpu or cpu
        # %%
        with torch.no_grad():
            #sdf_result = signed_distance_field(mesh_tem, coordinates_downsampled, allow_grad=False)
            # occp_result_ori = occupancy(mesh_tem, coordinates_downsampled, allow_grad=False, max_query_point_batch_size=max_query_point_batch_size)
            # occp_result_inverse = occupancy(new_mesh, coordinates_downsampled, allow_grad=False, max_query_point_batch_size=max_query_point_batch_size)
            occp_result = flexible_occupancy(mesh_tem, coordinates_downsampled, component_indx, allow_grad=False)

    else:
        ## Multi-GPUs Accelerating


        arctan_occp = SolidAngleOccp_components(mesh_tem)

        arctan_occp_inverse = SolidAngleOccp_components(new_mesh)

        # if u wanna use multi-gpus (but may not be faster)

        output_gpu = torch.device(f"cuda:{device_ids[0]}")

        arctan_occp = nn.DataParallel(arctan_occp, device_ids=device_ids, output_device=output_gpu)
        arctan_occp = arctan_occp.cuda()
        arctan_occp = arctan_occp.half()


        dats_set = torch.utils.data.TensorDataset(coordinates_downsampled.half().cpu(), torch.arange(0, n_points).long())

        sampler = DistributedSampler(dats_set)  

        dataloader = torch.utils.data.DataLoader(dats_set, batch_size=128**2, shuffle=False, drop_last=False, sampler=sampler)

        arctan_occp.eval()

        occp_result = torch.zeros(n_points, dtype=torch.half, device=output_gpu)

        occp_result = occp_result.half()


        for i, (data, idx) in enumerate(dataloader): ### multi-gpu
            points = data.cuda()
            indx = idx.to(output_gpu)
            with torch.no_grad():
                occp = arctan_occp.forward(points, max_query_point_batch_size=max_query_point_batch_size, component_indx=component_indx)
                
                
                occp_flipped = arctan_occp_inverse.forward(points, max_query_point_batch_size=max_query_point_batch_size, component_indx=component_indx)

                B_indx, N_indx, C_indx = torch.where(((occp - occp.round()).abs()< 1e-3)*(occp.round().abs() > 0.5))

                occp_result = torch.zeros_like(occp[...,0])

                occp_result[B_indx, N_indx] = 1.0   

                occp_result = torch.where(occp_result == 0, (occp_flipped+occp).sum(-1).abs(), occp_result)

                occp_result = torch.sigmoid(10*(occp_result - 0.5))

                occp_result[indx] = occp


    # occp_result = torch.where((occp_result_ori - occp_result_ori.round()).abs() > 1e-5, occp_result_inverse+occp_result_ori, occp_result_ori)
    occpfield = occp_result.view(1, voxel_size, voxel_size, voxel_size)
    occpfield = occpfield.permute(0, 3, 2, 1)

    if mode == 'occp':
        return occpfield, meshbbox


    dist, _ = _C.point_face_dist_forward(coordinates_downsampled.view(-1, 3).to(device),
                            torch.tensor([0], device=device, dtype=torch.int64),
                            mesh_tem.verts_packed()[mesh_tem.faces_packed(),:].to(device),
                            torch.tensor([0], device=device, dtype=torch.int64),
                            n_points, 1e-5)

    sdf = dist*(0.5-occp_result)

    sdf_trg = sdf.view(1, sample_size, sample_size, sample_size)
    sdf_trg = sdf_trg.permute(0, 3, 2, 1)

    return sdf_trg, meshbbox



if __name__ == '__main__':

    import argparse
    import warnings
    warnings.filterwarnings("ignore")


    args = argparse.ArgumentParser()
    args.add_argument('--mesh_path', type=str, default='data_example/Bull.obj')
    args.add_argument('--voxel_size', type=int, default=256)
    args.add_argument('--output_path', type=str, default='output')
    args.add_argument('--mode', type=str, default='occp')
    args.add_argument('--device_ids', type=int, nargs='+', default=[0])
    args = args.parse_args()

    mesh_path = args.mesh_path
    mesh = trimesh.load(mesh_path, force='mesh')

    # e.g. Bull 
    name = os.path.basename(mesh_path).split('.')[0]
    
    os.makedirs(args.output_path+'/'+name, exist_ok=True)
    sample_size = args.voxel_size
    occpfield, meshbbox = mesh2voxel(mesh, voxel_size=sample_size, spacial_range=1.0, 
                                    mode=args.mode,
                                    device_ids=args.device_ids)


    import mcubes
    if args.mode == 'occp':
        vertices, triangles = mcubes.marching_cubes((occpfield.permute(0, 3, 2, 1))[0].cpu().numpy(), 0.4)
    elif args.mode == 'sdf':
        vertices, triangles = mcubes.marching_cubes(-(occpfield.permute(0, 3, 2, 1))[0].cpu().numpy(), 0.0)

    vertices = meshbbox.mean(dim=-1).cpu().numpy() + (vertices*2/(sample_size-1)-1)*((meshbbox[:,1]-meshbbox[:,0]).view(1, 3).cpu().numpy())/2

    # create a mesh object
    trimesh_cubified = trimesh.Trimesh(vertices=vertices, faces=triangles)
    trimesh_cubified.export('output/'+name+'/'+name+'_'+args.mode+'_%d.obj'%sample_size)
    mesh.export('output/'+name+'/'+name+'_ori.obj')
    torch.save(occpfield, 'output/'+name+'/'+name+'_'+args.mode+'_%d.pt'%sample_size)
    print('Done! The output mesh is saved in '+args.output_path+'/'+name+'/'+name+'_'+args.mode+'_%d.obj'%sample_size)




