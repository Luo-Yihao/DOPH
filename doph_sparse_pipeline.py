import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from pytorch3d.structures import Meshes
from ops.mesh_geometry import normalize_mesh, get_occp_from_mesh
from ops.mesh_geometry import SolidAngleOccp

from atom3d.core.mesh_bvh import MeshBVH
from atom3d.apps.sparse_flood_fill import sparse_flood_fill, construct_surface_dual_graph
from atom3d.grid.octree_indexer import OctreeIndexer
from atom3d.mesh_extractor import SparseDMC

# Restoration of original constants for main execution
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RES = 1024
TARGET_SCALE = 0.95
k_ring = 2
DeformScale = 0.88
FILE_PATH = "/root/workspace/XMeshRepair/Atom-VDMR/output/Car2_opt.glb"


class XVoxelPipeline:
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def load_mesh(self, path: str) -> trimesh.Trimesh:
        obj = trimesh.load(path, force="mesh")
        if isinstance(obj, trimesh.Scene):
            if len(obj.geometry) == 0:
                raise ValueError("Empty scene / no geometry found.")
            mesh = trimesh.util.concatenate(tuple(obj.geometry.values()))
            return mesh
        if not isinstance(obj, trimesh.Trimesh):
            raise ValueError(f"Unsupported trimesh type: {type(obj)}")
        return obj

    def normalize_mesh(self, mesh: trimesh.Trimesh, target_scale: float = 0.95) -> trimesh.Trimesh:
        v = mesh.vertices
        vmin, vmax = v.min(0), v.max(0)
        center = (vmin + vmax) / 2.0
        max_dim = np.max(vmax - vmin)
        if max_dim <= 1e-12:
            return mesh
        scale = (2.0 * target_scale) / max_dim
        mesh.vertices = (v - center) * scale
        return mesh

    @torch.no_grad()
    def process(
        self,
        mesh: trimesh.Trimesh,
        res: int = 1024,
        k_ring: int = 1,
        connectivity: int = 6,
        deform_scale: float = 0.99,
        verbose: bool = True
    ):
        logs = []
        
        # Build BVH
        v = torch.from_numpy(mesh.vertices).float().to(self.device)
        f = torch.from_numpy(mesh.faces).long().to(self.device)
        bvh = MeshBVH(v, f)
        if verbose: logs.append("[BVH] built")

        # Flood Fill
        if verbose: logs.append(f"[FloodFill] RES={res} connectivity={connectivity} k_ring={k_ring}")
        fres = sparse_flood_fill(bvh, res, connectivity=connectivity, k_ring=k_ring, verbose=verbose)

        # Sparse Pipeline
        if verbose: logs.append("[SparsePipeline] construct dual graph ...")
        
        dam = fres["dam_ijk"]
        ext_tide = fres["extended_tide_ijk"]
        dry_adj = fres["dry_adjacent_ijk"]

        octree = OctreeIndexer(max_level=int(np.log2(res)), device=self.device)

        grid_verts, grid_verts_ijk, cube_idx, adj_idx, active_cells_ijk = construct_surface_dual_graph(
            dam, ext_tide, dry_adj, octree
        )

        # UDF + closest points
        udf_result = bvh.udf(grid_verts, return_closest=True)
        closest_points = udf_result.closest_points
        udf_values = udf_result.distances

        # deform_vector: grid vertex to closest point on surface
        deform_vector = (closest_points - grid_verts) * deform_scale

        # Winding Number Sign Determination
        if verbose: logs.append("[Sign] Computing signs via get_occp_from_mesh...")
        
        mesh_p3d = Meshes(verts=[v], faces=[f])
        # Use existing library function as requested
        with torch.no_grad():
            solid_angle_occp = SolidAngleOccp(mesh_p3d, False).half()
            occ = solid_angle_occp(grid_verts.half(), 
                                           max_query_point_batch_size=20000).view(-1)
            # occ = F.tanh(occ-0.5)
        
        # map [0,1] to signed distance field signs
        # occupancy > 0.5 means inside (negative sign in SDF)
        
        sign = torch.where(occ > 0.1, -torch.ones_like(occ), torch.ones_like(occ))

        scalar_field = udf_values *sign* (1.0 - deform_scale)

        # Meshing with SparseDMC
        if verbose: logs.append("[SparseDMC] extracting mesh...")
        dmc = SparseDMC(device=self.device)
        output = dmc(
            voxel_coords=active_cells_ijk,
            sdf=scalar_field,
            cube_idx=cube_idx.long(),
            resolution=res,
            deform=deform_vector,
            inference_mode="qef",
        )
        mesh_v, mesh_f = output.vertices, output.faces

        stats = {
            "graph_verts": int(grid_verts.shape[0]),
            "graph_cubes": int(cube_idx.shape[0]),
            "type_counts": {
                "dam(0)": int((adj_idx != -1).any(dim=1).sum().item()), # Placeholder
                "water(1)": 0,
                "dry(-1)": 0,
            },
            "scalar": {
                "pos": int((scalar_field > 0).sum().item()),
                "neg": int((scalar_field < 0).sum().item()),
                "zero": int((scalar_field == 0).sum().item()),
                "min": float(scalar_field.min().item()) if scalar_field.numel() else 0.0,
                "max": float(scalar_field.max().item()) if scalar_field.numel() else 0.0,
            },
        }
        
        if verbose:
            logs.append(f"[Graph] verts={stats['graph_verts']} cubes={stats['graph_cubes']}")
            logs.append(f"[Types] {stats['type_counts']}")
            logs.append(f"[Scalar] {stats['scalar']}")

        return mesh_v, mesh_f, stats, logs



def main():
    print(f"Loading {FILE_PATH}...")
    # Using the pipeline class inside main to reuse logic but keep original flow
    pipeline = XVoxelPipeline(device=DEVICE)
    
    mesh = trimesh.load(FILE_PATH, force='mesh')
    name = os.path.splitext(os.path.basename(FILE_PATH))[0]
    print(f"Loaded mesh '{name}' with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
    mesh = pipeline.normalize_mesh(mesh, target_scale=TARGET_SCALE)

    
    # Process using pipeline
    sv, sf, stats, logs = pipeline.process(
        mesh=mesh,
        res=RES,
        k_ring=k_ring,
        connectivity=6,
        deform_scale=DeformScale,
        verbose=True
    )
    
    
    sm = trimesh.Trimesh(sv.cpu().numpy(), sf.cpu().numpy())

    print("Whether the mesh is watertight:", sm.is_watertight)

    sm.export(f"sparse_{name}.glb")
    print(f"  Saved sparse_{name}.glb")

if __name__ == "__main__":
    main()