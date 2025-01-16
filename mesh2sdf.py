import os
import torch

from ops.mesh_geometry import *
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
from pytorch3d.ops import sample_points_from_meshes, cubify

import pytorch3d._C as _C
import trimesh
import math

def grid_smaple_sdf(mesh, resolution=128,  mode='sdf', out_dir=None,
                    rescalar = 0.9, open_thinkness = 1e-2*2):
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


    coordinates_downsampled = torch.stack(torch.meshgrid(torch.linspace(-1., 1., resolution),
                                                        torch.linspace(-1., 1., resolution),
                                                        torch.linspace(-1., 1., resolution),
                                                        indexing='ij'
                                                        ), dim=-1)
    # coordinates_downsampled = torch.stack(torch.meshgrid(torch.linspace(-0.5, 0.5, resolution),
    #                                                         torch.linspace(-0.5, 0.5, resolution),
    #                                                         torch.linspace(-0.5, 0.5, resolution),
    #                                                         indexing='ij'
    #                                                         ), dim=-1)
    coordinates_downsampled = rearrange(coordinates_downsampled, 'x y z c -> (z y x) c').to(device)

    if mode == 'sdf':
        field = get_sdf_from_mesh(mesh, 
                                    coordinates_downsampled, 
                                    threshold=0.4,
                                    open_thinkness=open_thinkness)
    elif mode == 'occupancy' or mode == 'occp':
        mode = 'occp'
        field = get_occp_from_mesh(mesh, 
                                        coordinates_downsampled, 
                                        threshold=0.4, open_thinkness=open_thinkness)
    
    else:
        raise ValueError('mode not supported yet: only support sdf and occupancy')
    
    field = rearrange(field, '(z y x) -> 1 z y x', z=resolution, y=resolution, x=resolution)


    if mode == 'sdf':
        cubified = cubify(-field, 0.0) # march cubes to get the mesh
    elif mode == 'occp':
        cubified = cubify(field, 0.5)

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.save(os.path.join(out_dir, name + '_' + mode + f'_{resolution}.npy'), field.squeeze().cpu().numpy())
        save_obj(os.path.join(out_dir, name + '_ori.obj'),
                    mesh.verts_list()[0], mesh.faces_list()[0])
        save_obj(os.path.join(out_dir, name + '_' +mode +f'_{resolution}.obj'),
                 cubified.verts_list()[0], cubified.faces_list()[0])
        
        print('Field saved to', os.path.join(out_dir, name + '_' + mode + f'_{resolution}.npy'))
        print('Re-mesh saved to', os.path.join(out_dir, name + '_' + mode + f'_{resolution}.obj'))
        
    return field, cubified

# main
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default='data_example/Chival.obj', help='path to the mesh')
    parser.add_argument('--mode', type=str, default='sdf', help='sdf or occupancy')
    parser.add_argument('--resolution', type=int, default=256, help='resolution of the SDF')
    parser.add_argument('--out_dir', type=str, default='./output', help='directory to save the SDF')
    parser.add_argument('--rescalar', type=float, default=0.9, help='rescale factor')
    parser.add_argument('--open_thinkness', type=float, default=1e-2*2, help='open thinkness')
    args = parser.parse_args()
    print('Mode:', args.mode)
    
    name = os.path.splitext(os.path.basename(args.mesh))[0]
    field, remesh = grid_smaple_sdf(args.mesh,
                                    resolution=args.resolution,
                                    mode=args.mode,
                                    out_dir=os.path.join(args.out_dir, name),
                                    rescalar=args.rescalar,
                                    open_thinkness=args.open_thinkness)
    
    

