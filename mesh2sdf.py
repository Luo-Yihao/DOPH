import os
import sys
import math
import torch
import trimesh
import numpy as np
import multiprocessing as mp
import scipy.ndimage as ndimage
from tqdm import tqdm
from skimage import measure, morphology
from skimage.measure import marching_cubes

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj

from ops.mesh_geometry import *

import pyvista as pv
import cubvh

# Configure pyvista
pv.start_xvfb(wait=0)
pv.set_jupyter_backend('html')

def grid_smaple_sdf(mesh, resolution=128,  mode='sdf', out_dir=None,
                    rescalar = 0.9, threshold=0.499,
                    open_thinkness = 1e-2, if_remesh=True):
    """
    Sample the signed distance field (SDF) of the mesh using octree. 
    The whole space is divided into 8^resolution voxels and nomalized into [-1, 1]^3.
    The default rescale factor is 0.9, which means the mesh is rescaled to fit into the [-0.9, 0.9]^3 box.

    Args:
        mesh: Meshes object representing the mesh or the path to the mesh.
        resolution: the resolution of the SDF.
        out_dir: the directory to save the SDF.
        rescalar: the rescale factor, default is 0.9. The bounding box of the mesh is rescaled to fit into the [-0.9, 0.9]^3 box.
        open_thinkness: the open thinkness for the SDF, default is 1e-2*2. 
    Returns:
        field: the SDF or occupancy field [Z, Y, X].
        remeshed: the remeshed mesh.
    """
    rescalar = rescalar
    open_thinkness = open_thinkness
    if isinstance(mesh, str):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        name = os.path.splitext(os.path.basename(mesh))[0]

        mesh = load_objs_as_meshes([mesh], device=device)
    elif isinstance(mesh, Meshes):
        device = mesh.device
        name = np.random.randint(0, 1000)
        name = 'mesh' + str(name)
        mesh = mesh.to(device)
    elif isinstance(mesh, trimesh.Trimesh):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        name = np.random.randint(0, 1000)
        name = 'mesh' + str(name)
        mesh = Meshes(verts=[torch.tensor(mesh.vertices, dtype=torch.float32, device=device)], faces=[torch.tensor(mesh.faces, dtype=torch.long, device=device)])
    else:
        raise ValueError('mesh type not supported yet: only support str, Meshes and trimesh.Trimesh')
    
    mesh = normalize_mesh(mesh, rescalar=rescalar)

    if mesh.faces_packed().shape[0] > 20000:
        print('Warning: The mesh is too large, it may take a long time to compute the SDF.')

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save_obj(os.path.join(out_dir, name + '_ori.obj'), mesh.verts_packed(), mesh.faces_packed())


    coordinates_downsampled = torch.stack(torch.meshgrid(torch.linspace(-1., 1., resolution),
                                                        torch.linspace(-1., 1., resolution),
                                                        torch.linspace(-1., 1., resolution),
                                                        indexing='ij'
                                                        ), dim=-1)
    
    coordinates_downsampled = rearrange(coordinates_downsampled, 'x y z c -> (x y z) c').to(device)

    if mode == 'sdf':
        field = get_sdf_from_mesh(mesh, 
                                    coordinates_downsampled, 
                                    threshold=threshold,
                                    open_thinkness=open_thinkness)

        
    elif mode == 'occupancy' or mode == 'occp':
        mode = 'occp'
        field = get_occp_from_mesh(mesh, 
                                        coordinates_downsampled, 
                                        threshold=threshold, open_thinkness=open_thinkness)
    else:
        raise ValueError('mode not supported yet: only support sdf and occupancy')
    
    print('Field Done, Remeshing...')
    
    field = rearrange(field, '(x y z) -> x y z', z=resolution, y=resolution, x=resolution)

    if if_remesh:
        if mode == 'sdf':
            field_np = field.cpu().numpy()
            vertices, faces, normals, _ = marching_cubes(field_np, 1e-5, allow_degenerate=False)
            vertices = vertices/(resolution-1)*2.-1.

        elif mode == 'occp':
            field_np = field.cpu().numpy()
            # vertices, faces = mcubes.marching_cubes(mcubes.smooth(fiel_np), 0)
            vertices, faces, normals, _ = marching_cubes(field_np, 0.5, allow_degenerate=False)
            vertices = vertices/(resolution-1)*2.-1.

        trimesh_tem = trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
        # decimate the mesh
        # trimesh_tem = trimesh_tem.simplify_quadric_decimation(20000)
        trimesh_tem = trimesh.smoothing.filter_mut_dif_laplacian(trimesh_tem, lamb=0.5, iterations=50)

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        np.save(os.path.join(out_dir, name + '_' + mode + f'_{resolution}.npy'), field.cpu().numpy())
        print('Field saved to', os.path.join(out_dir, name + '_' + mode + f'_{resolution}.npy'))
        print('Field in format of np.array with shape (X, Y, Z) but returned value is tensor with shape (1, Z, Y, X)')


        if if_remesh:
            obj_path = os.path.join(out_dir, name + '_' +mode +f'_{resolution}.obj')
            trimesh_tem.export(obj_path)
            print('Re-mesh saved to', obj_path)

    field = field.permute(2,1,0).unsqueeze(0)
    if if_remesh:
        mesh = Meshes(verts=[torch.from_numpy(trimesh_tem.vertices).float().to(device)],
                        faces=[torch.from_numpy(trimesh_tem.faces).long().to(device)])

        return field, mesh
    else:
        return field

# main
if __name__ == '__main__':
    # nohup python -u mesh2sdf.py --mesh data_example/car.obj --mode occp --resolution 512 --out_dir ./output --rescalar 0.99 --open_thinkness 1e-2 > log.txt 2>&1 &
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default='data_example/Chival.obj', help='path to the mesh')
    parser.add_argument('--mode', type=str, default='sdf', help='sdf or occupancy')
    parser.add_argument('--resolution', type=int, default=512, help='resolution of the SDF')
    parser.add_argument('--out_dir', type=str, default='./output', help='directory to save the SDF')
    parser.add_argument('--rescalar', type=float, default=0.95, help='rescale factor')
    parser.add_argument('--threshold', type=float, default=0.99, help='threshold')
    parser.add_argument('--open_thinkness', type=float, default=None, help='open thinkness')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--if_remesh', type=bool, default=True, help='if remesh')
    args = parser.parse_args()
    print('Mode:', args.mode)

    out_dir = args.out_dir
    mode = args.mode

    resolution = args.resolution
    eps = 1/(resolution)
    rescalar = args.rescalar
    threshold = args.threshold
    if args.open_thinkness is None:
        open_thinkness = eps/2
    else:
        open_thinkness = args.open_thinkness

    # from mesh_geometry import *
    
    name = os.path.splitext(os.path.basename(args.mesh))[0]

    out_dir = os.path.join(out_dir, name)

    trimesh_tem = trimesh.load(args.mesh)


    device = torch.device(args.device)


    mesh_tem = Meshes(verts=[torch.from_numpy(trimesh_tem.vertices).to(torch.float32)],
                  faces=[torch.from_numpy(trimesh_tem.faces).to(torch.long)])

    mesh_tem = mesh_tem.to(device)
    # src_points_tensor,_,face_normals, points, normals, mesh= load_mesh_and_sample(filename_mesh,10_000_000,100_000)
    # mesh_tem = load_objs_as_meshes([filename_mesh], device=device)
    mesh_tem = normalize_mesh(mesh_tem, rescalar)

    grid_voxel_coarse = torch.stack(
        torch.meshgrid(
            torch.linspace(-1+1/(resolution)/2, 1-1/(resolution)/2, resolution),
            torch.linspace(-1+1/(resolution)/2, 1-1/(resolution)/2, resolution),
            torch.linspace(-1+1/(resolution)/2, 1-1/(resolution)/2, resolution),
            indexing='ij'
        ), dim=-1
    ).to('cpu').float()


    udf_coarse = torch.zeros((resolution**3,)).to('cpu')


    print('Querying UDF values...')

    BVH = cubvh.cuBVH(mesh_tem.verts_packed(), mesh_tem.faces_packed()) # build with numpy.ndarray/torch.Tensor
    max_query = 128**3
    for i in tqdm(range(0, resolution**3, max_query)):
        udf_coarse_tem, _, _ = BVH.unsigned_distance(grid_voxel_coarse.view(-1, 3)[i:i+max_query].to(device)
                                                                , return_uvw=False)
        udf_coarse[i:i+max_query] = udf_coarse_tem.cpu()
        del udf_coarse_tem


    udf_coarse = rearrange(udf_coarse, '(a b c) -> 1 a b c', a=resolution, 
                        b=resolution, c=resolution)


    active_threshold = 1/(resolution)/2*np.sqrt(3)

    active_X, active_Y, active_Z = torch.where(udf_coarse[0] < active_threshold)

    print('Flood filling...')

    occ_floodmask = ~morphology.flood(udf_coarse[0].numpy()<active_threshold,
                                    (0,0,0), connectivity=1)
    occ_floodmask = torch.tensor(occ_floodmask).float().unsqueeze(0)

    ## 
    select_index = torch.where(udf_coarse.view(-1) < 4*eps)[0]

    print('Computing refined occupancy...')


    winding_occp = get_occp_from_mesh(mesh_tem, 
                                    grid_voxel_coarse.view(-1, 3)[select_index,:].to(device),
                                    open_thinkness=open_thinkness,
                                    if_smooth=True, threshold=threshold)

    winding_occp +=  (BVH.signed_distance(grid_voxel_coarse.view(-1, 3)[select_index,:].to(device),
                                        return_uvw=False)[0] < eps).view(-1).float().to(device)*0.5

    winding_occp = winding_occp/1.5


    alpha = 1e2
    winding_occp = torch.sigmoid(alpha*(winding_occp-0.99))/torch.sigmoid(torch.tensor(alpha*0.01))
    winding_occp = torch.clamp(winding_occp, 0, 1)


    occp_refine = occ_floodmask.view(-1)
    occp_refine[select_index] = winding_occp.float().to('cpu')
    occp_refine = rearrange(occp_refine, '(a b c) -> 1 a b c', a=resolution,
                                b=resolution, c=resolution)




    miu = resolution**2
    # sdf_floodmask = sdf_coarse.abs() * (2*(0.5-occ_floodmask))
    sdf_floodmask = (-miu*udf_coarse**2).exp() * (occp_refine).float() \
        + (1 -(-miu*udf_coarse**2).exp()) * occ_floodmask.float()



    sdf_floodmask = torch.tanh(2*(0.5-sdf_floodmask)*alpha)/np.tanh(alpha)

    sdf_floodmask = sdf_floodmask * udf_coarse


    del grid_voxel_coarse, winding_occp

    if args.mode == 'sdf':
        field = sdf_floodmask[0].cpu().numpy()
    elif args.mode == 'occp':
        field = occp_refine[0].cpu().numpy()

    
    
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
    np.save(os.path.join(out_dir, name + '_' + mode + f'_{resolution}.npy'), field)
    print('Field saved to', os.path.join(out_dir, name + '_' + mode + f'_{resolution}.npy'))


    if args.if_remesh:

        print('Marching cubes...')
        vertices, faces, _, _ =  measure.marching_cubes(sdf_floodmask[0].numpy(), level=eps*2)
        vertices = vertices/ (resolution-1) * (2 - 1/(resolution)) - 1 + 1/(resolution)/2

        flood_mesh = Meshes(verts=[torch.from_numpy(vertices.copy()).to(torch.float32)], faces=[torch.from_numpy(faces.copy()).to(torch.long)])

        mesh_plot = trimesh.Trimesh(vertices=flood_mesh.verts_packed().cpu().numpy(), 
                                        faces=flood_mesh.faces_packed().cpu().numpy(), process=False)

        print('The remeshed mesh is watertight: ', mesh_plot.is_watertight)

        print('Smoothing the mesh...')
        mesh_plot = trimesh.smoothing.filter_mut_dif_laplacian(mesh_plot, iterations=3)

        print('%d vertices, %d faces in the remeshed mesh at resolution %d^3.' % (len(mesh_plot.vertices), len(mesh_plot.faces), resolution))
        obj_path = os.path.join(out_dir, name + f'_{resolution}.obj')
        mesh_plot.export(obj_path)
        print('Re-mesh saved to', obj_path)

        pl = pv.Plotter(notebook=False, off_screen=True)
        pv_plot = pv.wrap(mesh_plot)
        pv_plot['face_normals'] = (mesh_plot.vertex_normals + 1) / 2.
        pl.add_mesh(pv_plot, opacity=1, scalars='face_normals', show_edges=0, rgb=True)
        pl.screenshot(os.path.join(out_dir, name  + f'_{resolution}.png'))

        print('Rendered image saved to', os.path.join(out_dir, name +f'_{resolution}.png'))






    

