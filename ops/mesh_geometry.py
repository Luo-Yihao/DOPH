import torch
import torch.nn as nn
import torch.nn.functional as F

import trimesh

from pytorch3d.structures import Meshes
from pytorch3d.ops import mesh_face_areas_normals, packed_to_padded
from einops import rearrange, einsum, repeat

from torch_scatter import scatter

import numpy as np

from pytorch3d import _C
from tqdm import tqdm


def normalize_mesh(mesh, rescalar=0.99):
    """
    Normalize the mesh to fit in the unit sphere
    Args:
        mesh: Meshes object or trimesh object
        rescalar: float, the scale factor to rescale the mesh
    """
    if isinstance(mesh, Meshes):
       
        bbox = mesh.get_bounding_boxes()
        B = bbox.shape[0]
        center = (bbox[:, :, 0] + bbox[:, :, 1]) / 2
        center = center.view(B, 1, 3)
        size = (bbox[:, :, 1] - bbox[:, :, 0]) 

        scale = 2.0 / (torch.max(size, dim=1)[0]+1e-8).view(B, 1)*rescalar
        scale = scale.view(B, 1, 1)
        mesh = mesh.update_padded((mesh.verts_padded()-center)*scale)
        return mesh

    elif isinstance(mesh, trimesh.Trimesh):
        bbox_min, bbox_max = mesh.bounds
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min

        # Scale factor to normalize to [-1, 1]
        scale_factor = 2.0 / np.max(bbox_size)  # Ensures the longest side fits in [-1, 1]

        # Apply translation and scaling
        mesh.apply_translation(-bbox_center)  # Move the mesh center to the origin
        mesh.apply_scale(scale_factor)

    return mesh

def get_faces_coordinates_padded(meshes: Meshes):
    """
    Get the faces coordinates of the meshes in padded format.
    return:
        face_coord_padded: [B, F, 3, 3]
        
    """
    face_mesh_first_idx = meshes.mesh_to_faces_packed_first_idx()
    face_coord_packed = meshes.verts_packed()[meshes.faces_packed(),:]

    face_coord_padded = packed_to_padded(face_coord_packed, face_mesh_first_idx, max_size=meshes.faces_padded().shape[1])

    return face_coord_padded


def faces_angle(meshs: Meshes)->torch.Tensor:
    """
    Compute the angle of each face in a mesh
    Args:
        meshs: Meshes object
    Returns:
        angles: Tensor of shape (N,3) where N is the number of faces
    """
    Face_coord = meshs.verts_packed()[meshs.faces_packed()]
    A = Face_coord[:,1,:] - Face_coord[:,0,:]
    B = Face_coord[:,2,:] - Face_coord[:,1,:]
    C = Face_coord[:,0,:] - Face_coord[:,2,:]
    
    angle_0 = torch.arccos(-torch.sum(C*A,dim=1)/(1e-8+(torch.norm(C,dim=1)*torch.norm(A,dim=1))))
    

    angle_1 = torch.arccos(-torch.sum(A*B,dim=1)/(1e-8+(torch.norm(A,dim=1)*torch.norm(B,dim=1))))
    angle_2 = torch.arccos(-torch.sum(B*C,dim=1)/(1e-8+(torch.norm(B,dim=1)*torch.norm(C,dim=1))))
    angles = torch.stack([angle_0,angle_1,angle_2],dim=1)
    return angles

def dual_area_weights_faces(Surfaces: Meshes)->torch.Tensor:
    """
    Compute the dual area weights of 3 vertices of each triangles in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_weight: Tensor of shape (N,3) where N is the number of triangles
        the dual area of a vertices in a triangles is defined as the area of the sub-quadrilateral divided by three perpendicular bisectors
    """
    angles = faces_angle(Surfaces)
    sin2angle = torch.sin(2*angles)
    dual_area_weight = sin2angle/(torch.sum(sin2angle,dim=-1,keepdim=True)+1e-8)
    dual_area_weight = (dual_area_weight[:,[2,0,1]]+dual_area_weight[:,[1,2,0]])/2
    
    
    return dual_area_weight


def dual_area_vertex(Surfaces: Meshes)->torch.Tensor:
    """
    Compute the dual area of each vertices in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_per_vertex: Tensor of shape (N,1) where N is the number of vertices
        the dual area of a vertices is defined as the sum of the dual area of the triangles that contains this vertices
    """
    dual_weights = dual_area_weights_faces(Surfaces)
    dual_areas = dual_weights*Surfaces.faces_areas_packed().view(-1,1)

    face2vertex_index = Surfaces.faces_packed().view(-1)

    dual_area_per_vertex = scatter(dual_areas.view(-1), face2vertex_index, reduce='sum', dim_size=Surfaces.verts_packed().shape[0])
    
    return dual_area_per_vertex.view(-1,1)


def gaussian_curvature(Surfaces: Meshes,return_topology=False)->torch.Tensor:
    """
    Compute the gaussian curvature of each vertices in a mesh by local Gauss-Bonnet theorem
    Args:
        Surfaces: Meshes object
        return_topology: bool, if True, return the Euler characteristic and genus of the mesh
    Returns:
        gaussian_curvature: Tensor of shape (N,1) where N is the number of vertices
        the gaussian curvature of a vertices is defined as the sum of the angles of the triangles that contains this vertices minus 2*pi and divided by the dual area of this vertices
    """

    face2vertex_index = Surfaces.faces_packed().view(-1)

    angle_face = faces_angle(Surfaces)

    dual_weights = dual_area_weights_faces(Surfaces)

    dual_areas = dual_weights*Surfaces.faces_areas_packed().view(-1,1)

    dual_area_per_vertex = scatter(dual_areas.view(-1), face2vertex_index, reduce='sum')

    angle_sum_per_vertex = scatter(angle_face.view(-1), face2vertex_index, reduce='sum')

    curvature = (2*torch.pi - angle_sum_per_vertex)/(dual_area_per_vertex+1e-8)

    # Euler_chara = torch.sparse.mm(vertices_to_meshid.float().T,(2*torch.pi - sum_angle_for_vertices).T).T/torch.pi/2
    # Euler_chara = torch.round(Euler_chara)
    # print('Euler_characteristic:',Euler_chara)
    # Genus = (2-Euler_chara)/2
    #print('Genus:',Genus)
    if return_topology:
        Euler_chara = (curvature*dual_area_per_vertex).sum()/2/torch.pi
        return curvature, Euler_chara
    return curvature

def gaussian_curvature_density(Surfaces: Meshes)->torch.Tensor:
    """
    Compute the gaussian curvature of each vertices in a mesh by local Gauss-Bonnet theorem
    Args:
        Surfaces: Meshes object
        return_topology: bool, if True, return the Euler characteristic and genus of the mesh
    Returns:
        gaussian_curvature: Tensor of shape (N,1) where N is the number of vertices
        the gaussian curvature of a vertices is defined as the sum of the angles of the triangles that contains this vertices minus 2*pi and divided by the dual area of this vertices
    """

    face2vertex_index = Surfaces.faces_packed().view(-1)

    angle_face = faces_angle(Surfaces)

    dual_weights = dual_area_weights_faces(Surfaces)

    dual_areas = dual_weights*Surfaces.faces_areas_packed().view(-1,1)

    angle_sum_per_vertex = scatter(angle_face.view(-1), face2vertex_index, reduce='sum')

    curvature_density = (2*torch.pi - angle_sum_per_vertex)

    # Euler_chara = torch.sparse.mm(vertices_to_meshid.float().T,(2*torch.pi - sum_angle_for_vertices).T).T/torch.pi/2
    # Euler_chara = torch.round(Euler_chara)
    # print('Euler_characteristic:',Euler_chara)
    # Genus = (2-Euler_chara)/2
    #print('Genus:',Genus)
    return curvature_density

def Average_from_verts_to_face(Surfaces: Meshes, feature_verts: torch.Tensor)->torch.Tensor:
    """
    Compute the average of feature vectors defined on vertices to faces by dual area weights
    Args:
        Surfaces: Meshes object
        feature_verts: Tensor of shape (N,C) where N is the number of vertices, C is the number of feature channels
    Returns:
        vect_faces: Tensor of shape (F,C) where F is the number of faces
    """
    assert feature_verts.shape[0] == Surfaces.verts_packed().shape[0]

    dual_weight = dual_area_weights_faces(Surfaces).view(-1,3,1)

    feature_faces = feature_verts[Surfaces.faces_packed(),:]
    
    wg = dual_weight*feature_faces
    return wg.sum(dim=-2)

### winding number

def Electric_strength(q, p):
    """
    q: (M, 3) - charge position
    p: (N, 3) - field position
    """
    assert q.shape[-1] == 3 and len(q.shape) == 2, "q should be (M, 3)"
    assert p.shape[-1] == 3 and len(p.shape) == 2, "p should be (N, 3)"
    q = q.unsqueeze(1).repeat(1, p.shape[0], 1)
    p = p.unsqueeze(0)
    return (p-q)/(torch.norm(p-q, dim=-1, keepdim=True)**3+1e-8)



def winding_occupancy(mesh_tem: Meshes, points: torch.Tensor, max_v_per_call=2000):
    """
    Involving the winding number to evaluate the occupancy of the points relative to the mesh
    mesh_tem: the reference mesh
    points: the points to be evaluated Nx3
    """
    dual_areas = dual_area_vertex(mesh_tem)

    normals_areaic = mesh_tem.verts_normals_packed() * dual_areas.view(-1,1)

    winding_field = torch.zeros(points.shape[0], device=points.device)

    for i in range(0, normals_areaic.shape[0], max_v_per_call):
        vert_elefields_temp = Electric_strength(points, mesh_tem.verts_packed()[i:i+max_v_per_call])
        winding_field += torch.einsum('m n c, n c -> m', vert_elefields_temp, normals_areaic[i:i+max_v_per_call])


    winding_field = winding_field/4/np.pi

    return winding_field


def winding_occupancy_face(mesh_tem: Meshes, points: torch.Tensor, max_f_per_call=2000):
    """
    Involving the winding number to evaluate the occupancy of the points relative to the mesh
    mesh_tem: the reference mesh
    points: the points to be evaluated Nx3
    """
    areaic, normals = mesh_tem.faces_areas_packed(), mesh_tem.faces_normals_packed()

    normals_areaic = normals * areaic.view(-1,1)

    face_centers = mesh_tem.verts_packed()[mesh_tem.faces_packed()].view(-1,3,3).mean(dim=-2)


    winding_field = torch.zeros(points.shape[0], device=points.device)

    for i in range(0, normals_areaic.shape[0], max_f_per_call):
        face_elefields_temp = Electric_strength(points, face_centers[i:i+max_f_per_call])
        winding_field += torch.einsum('m n c, n c -> m', face_elefields_temp, normals_areaic[i:i+max_f_per_call])

    winding_field = winding_field/4/np.pi

    return winding_field


class Differentiable_Grid_Voxelizer(nn.Module):
    def __init__(self, bbox_density=128, integrate_method='vertex'):
        super(Differentiable_Grid_Voxelizer, self).__init__()
        self.bbox_density = bbox_density
        if integrate_method == 'face':
            self.winding_occupancy = winding_occupancy_face
        elif integrate_method == 'vertex':
            self.winding_occupancy = winding_occupancy

    def forward(self, mesh_src: Meshes, output_resolution=256, max_v_per_call=2000, if_binary=False):
        """
        mesh_src: the source mesh to be voxelized (should be rescaled into the normalized coordinates [-1,1])
        return_type: the type of the return
        """

        # random sampling in bounding box
        
        resolution = self.bbox_density
        bbox = mesh_src.get_bounding_boxes()[0]
        
        assert torch.abs(bbox.max())<=1 and torch.abs(bbox.min())<=1, "The bounding box should be normalized into [-1,1]"

        # grid sampling in bounding box
        bbox_length = (bbox[:, 1] - bbox[:, 0])
        step_lengths = bbox_length.max() / resolution
        step = (bbox_length / step_lengths).int() + 1

        x = torch.linspace(bbox[0, 0], bbox[0, 1], steps=step[0], device=mesh_src.device)
        y = torch.linspace(bbox[1, 0], bbox[1, 1], steps=step[1], device=mesh_src.device)
        z = torch.linspace(bbox[2, 0], bbox[2, 1], steps=step[2], device=mesh_src.device)

        

        x_index, y_index, z_index = torch.meshgrid(x, y, z)

        slice_length_ranking, slice_direction_ranking = torch.sort(step, descending=False)

        # change the order of the coordinates for the acceleration 
        slice_direction_ranking_reverse = torch.argsort(slice_direction_ranking,descending=False)


        coordinates = torch.stack([x_index, y_index, z_index], dim=-1)


        coordinates = coordinates.permute(slice_direction_ranking.tolist() + [3])


        coordinates = rearrange(coordinates, 'x y z c -> x (y z) c', c = 3, x = slice_length_ranking[0], y = slice_length_ranking[1], z = slice_length_ranking[2])
        occupency_fields = []
        for i in range(0, coordinates.shape[0]):
            tem_charge = coordinates[i]

            occupency_temp = self.winding_occupancy(mesh_src, tem_charge, max_v_per_call=max_v_per_call)

            if if_binary:
                occupency_temp = torch.sigmoid((self.winding_occupancy(mesh_src, tem_charge,max_v_per_call=max_v_per_call)-0.5)*100)

            occupency_fields.append(occupency_temp)

        occupency_fields = torch.stack(occupency_fields, dim=0)

        # embedding the bounding box into the whole space
        resolution_whole = output_resolution
        bbox_index = (bbox +1)*resolution_whole//2
        X_b, Y_b, Z_b = bbox_index.int().tolist()
        whole_image = torch.zeros(resolution_whole, resolution_whole, resolution_whole, device=mesh_src.device)

        bbox_transformed = rearrange(occupency_fields, 'x (y z) -> x y z', x = slice_length_ranking[0], y = slice_length_ranking[1], z = slice_length_ranking[2])

        bbox_transformed = bbox_transformed.permute(slice_direction_ranking_reverse.tolist()).unsqueeze(0).unsqueeze(0)
        # print(bbox_transformed.shape)
        # print((X_b[1]-X_b[0]+1, Y_b[1]-Y_b[0]+1, Z_b[1]-Z_b[0]+1))
        bbox_transformed = F.interpolate(bbox_transformed, size=(X_b[1]-X_b[0]+1, Y_b[1]-Y_b[0]+1, Z_b[1]-Z_b[0]+1), mode='trilinear')
        bbox_transformed = bbox_transformed.squeeze(0).squeeze(0)

        whole_image[X_b[0]:X_b[1]+1, Y_b[0]:Y_b[1]+1, Z_b[0]:Z_b[1]+1] = bbox_transformed

        whole_image = (whole_image.permute(2, 1, 0)).unsqueeze(0)

        return whole_image
    

def differentiable_sdf(mesh: Meshes, query_points: torch.Tensor, max_query_point_batch_size=10000, binary_style='tanh', return_occp=False, integrate_method='face'):
    """
    Compute the signed distance field of the mesh
    Args:
        mesh: Meshes object
        query_points: Tensor of shape (N,3) where N is the number of query points
        max_query_point_batch_size: the maximum number of query points in each batch
        binary_style: the style of the binary occupancy, 'tanh' or 'sign'
        return_occp: if True, return the binary occupancy
        integrate_method: the method to integrate the winding number, 'face' or 'vertex'
    Returns:
        sdf: Tensor of shape (N,1) where N is the number of query points
    """

 
    for i in range(0, query_points.shape[0], max_query_point_batch_size):

        points = query_points[i:i+max_query_point_batch_size]

        dist, _ = _C.point_face_dist_forward(points.view(-1, 3),
                                torch.tensor([0], device=mesh.device),
                                mesh.verts_packed()[mesh.faces_packed(),:],
                                torch.tensor([0], device=mesh.device),
                                points.shape[0], 1e-5)
        dist = dist.view(-1, 1)
            
        if integrate_method == 'face':
            occupancy = winding_occupancy_face(mesh, points, max_f_per_call=2000).view(-1,1)
        else:
            occupancy = winding_occupancy(mesh, points, max_v_per_call=2000).view(-1,1)
        # binary occupancy
        if binary_style == 'tanh':
            occupancy = -torch.tanh((occupancy-0.5)*1000) # [0,1] -> [-1,1]
        if binary_style == 'sign':
            occupancy = (occupancy<0.5).float()
        if i == 0:
            sdf = dist*occupancy
            occp = occupancy
        else:
            sdf = torch.cat([sdf, dist*occupancy])
            occp = torch.cat([occp, occupancy])

    if return_occp:
        return sdf, occp
        
    return sdf
        

    
class Differentiable_Grid_SDF(nn.Module):
    def __init__(self, bbox_density=128, integrate_method='face'):
        super(Differentiable_Grid_SDF, self).__init__()
        self.bbox_density = bbox_density
        if integrate_method == 'face':
            self.winding_occupancy = winding_occupancy_face
        elif integrate_method == 'vertex':
            self.winding_occupancy = winding_occupancy

    def forward(self, mesh_src: Meshes, output_resolution=256, max_v_per_call=2000, if_binary=False):
        """
        mesh_src: the source mesh to be voxelized (should be rescaled into the normalized coordinates [-1,1])
        return_type: the type of the return
        """

        # random sampling in bounding box
        
        resolution = self.bbox_density
        bbox = mesh_src.get_bounding_boxes()[0]
        
        assert torch.abs(bbox.max())<=1 and torch.abs(bbox.min())<=1, "The bounding box should be normalized into [-1,1]"

        # grid sampling in bounding box
        bbox_length = (bbox[:, 1] - bbox[:, 0])
        step_lengths = bbox_length.max() / resolution
        step = (bbox_length / step_lengths).int() + 1

        x = torch.linspace(bbox[0, 0], bbox[0, 1], steps=step[0], device=mesh_src.device)
        y = torch.linspace(bbox[1, 0], bbox[1, 1], steps=step[1], device=mesh_src.device)
        z = torch.linspace(bbox[2, 0], bbox[2, 1], steps=step[2], device=mesh_src.device)

        

        x_index, y_index, z_index = torch.meshgrid(x, y, z)

        slice_length_ranking, slice_direction_ranking = torch.sort(step, descending=False)

        # change the order of the coordinates for the acceleration 
        slice_direction_ranking_reverse = torch.argsort(slice_direction_ranking,descending=False)


        coordinates = torch.stack([x_index, y_index, z_index], dim=-1)


        coordinates = coordinates.permute(slice_direction_ranking.tolist() + [3])


        coordinates = rearrange(coordinates, 'x y z c -> x (y z) c', c = 3, x = slice_length_ranking[0], y = slice_length_ranking[1], z = slice_length_ranking[2])
        occupency_fields = []
        for i in range(0, coordinates.shape[0]):
            tem_charge = coordinates[i]

            occupency_temp = self.winding_occupancy(mesh_src, tem_charge,max_v_per_call=max_v_per_call)

            if if_binary:
                occupency_temp = torch.sigmoid((self.winding_occupancy(mesh_src, tem_charge,max_v_per_call=max_v_per_call)-0.5)*100)

            occupency_fields.append(occupency_temp)

        occupency_fields = torch.stack(occupency_fields, dim=0)

        # embedding the bounding box into the whole space
        resolution_whole = output_resolution
        bbox_index = (bbox +1)*resolution_whole//2
        X_b, Y_b, Z_b = bbox_index.int().tolist()
        whole_image = torch.zeros(resolution_whole, resolution_whole, resolution_whole, device=mesh_src.device)

        bbox_transformed = rearrange(occupency_fields, 'x (y z) -> x y z', x = slice_length_ranking[0], y = slice_length_ranking[1], z = slice_length_ranking[2])

        bbox_transformed = bbox_transformed.permute(slice_direction_ranking_reverse.tolist()).unsqueeze(0).unsqueeze(0)
        # print(bbox_transformed.shape)
        # print((X_b[1]-X_b[0]+1, Y_b[1]-Y_b[0]+1, Z_b[1]-Z_b[0]+1))
        bbox_transformed = F.interpolate(bbox_transformed, size=(X_b[1]-X_b[0]+1, Y_b[1]-Y_b[0]+1, Z_b[1]-Z_b[0]+1), mode='trilinear')
        bbox_transformed = bbox_transformed.squeeze(0).squeeze(0)

        whole_image[X_b[0]:X_b[1]+1, Y_b[0]:Y_b[1]+1, Z_b[0]:Z_b[1]+1] = bbox_transformed

        whole_image = (whole_image.permute(2, 1, 0)).unsqueeze(0)

        return whole_image
    





class SolidAngleOccp(nn.Module):

    def __init__(self, mesh_src: Meshes, allow_grad=True):
        super(SolidAngleOccp, self).__init__()
        self.face_coord = get_faces_coordinates_padded(mesh_src) # [B, F, 3, 3]
        if allow_grad:
            self.face_coord = nn.Parameter(self.face_coord)
        else:
            self.face_coord = nn.Parameter(self.face_coord, requires_grad=False)


    def forward(self, query_points: torch.Tensor, max_query_point_batch_size=10000, max_face_batch_size=1000):
        """
        Compute the occupancy of the query point relative to the mesh
        Args:
            query_points: Tensor of shape (N,3) or (B, N, 3) where N is the number of query points
            max_query_point_batch_size: int, the maximum batch size for query points
        Returns:
            occupancy: Tensor of shape (B, N)
        """

        if len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0).repeat(self.face_coord.shape[0],1,1)# [B, N, 3]
        else:
            assert len(query_points.shape) == 3
            assert query_points.shape[0] == self.face_coord.shape[0] # [B, N, 3]

        occp = []

        for term in range(0, query_points.shape[-2], max_query_point_batch_size):

            points = query_points[...,term:term+max_query_point_batch_size,:]

            occp_term = torch.zeros(points.shape[0], points.shape[1]).to(points.device)

            for face_term in range(0, self.face_coord.shape[1], max_face_batch_size):

                face_coord = self.face_coord[:,face_term:face_term+max_face_batch_size,:,:]

                # face_vect = face_coord.unsqueeze(1).repeat(1,points.shape[-2],1,1,1) - points.unsqueeze(2).unsqueeze(2)
                face_vect = face_coord.unsqueeze(1).repeat(1,points.shape[-2],1,1,1) - points.unsqueeze(2).unsqueeze(2).repeat(1,1,face_coord.shape[-3],3,1)

                a_vert = face_vect[:,:,:,0,:] # [B, N, F, 3]        
                b_vert = face_vect[:,:,:,1,:] # [B, N, F, 3]
                c_vert = face_vect[:,:,:,2,:] # [B, N, F, 3]

                face_det = (a_vert * b_vert.cross(c_vert)).sum(dim=-1) # [B, N, F]

                abc = face_vect.norm(dim=-1).prod(dim=-1) # [B, N, F]

                ab = (a_vert*b_vert).sum(-1) # [B, N, F]
                bc = (b_vert*c_vert).sum(-1) # [B, N, F]
                ac = (a_vert*c_vert).sum(-1) # [B, N, F]

                solid_angle_2 = torch.arctan2(face_det, (abc + bc*a_vert.norm(dim=-1) + ac*b_vert.norm(dim=-1) + ab*c_vert.norm(dim=-1))) # [B, N, F]

                occp_term += solid_angle_2.sum(-1) # solid_angle/pi/4
                
            occp.append(occp_term/np.pi/2)

        return torch.cat(occp, dim=-1)

## differentiable signed distance field

def occupancy(mesh: Meshes, pt_target : torch.Tensor, allow_grad: bool = False, max_query_point_batch_size=2000, max_face_batch_size=10000):
    solid_angle_occp = SolidAngleOccp(mesh, allow_grad)
    if allow_grad:
        occp = solid_angle_occp(pt_target, max_query_point_batch_size, max_face_batch_size)
    else:
        with torch.no_grad():
            occp = solid_angle_occp(pt_target, max_query_point_batch_size, max_face_batch_size)
    return occp

def signed_distance_field(mesh: Meshes, pt_target : torch.Tensor, allow_grad: bool = True, max_query_point_batch_size=2000):
    occp = occupancy(mesh, pt_target, allow_grad, max_query_point_batch_size)
    device = mesh.device
    n_points = pt_target.shape[0]
    dist, _ = _C.point_face_dist_forward(pt_target.view(-1, 3).to(device),
            torch.tensor([0], device=device, dtype=torch.int64),
            mesh.verts_packed()[mesh.faces_packed(),:].to(device),
            torch.tensor([0], device=device, dtype=torch.int64),
            n_points, 1e-6)
    sign = 2*(occp.view(-1)-0.5)
    return sign*dist


## Components-wise Occupancy

class SolidAngleOccp_components(nn.Module):

    def __init__(self, mesh_src: Meshes, allow_grad=False):
        super(SolidAngleOccp_components, self).__init__()
        self.face_coord = get_faces_coordinates_padded(mesh_src) # [B, F, 3, 3]
        if allow_grad:
            self.face_coord = nn.Parameter(self.face_coord)
        else:
            self.face_coord = nn.Parameter(self.face_coord, requires_grad=False)

        self.face_first_index = mesh_src.mesh_to_faces_packed_first_idx()

    
    def forward(self, query_points: torch.Tensor, gather_label, max_query_point_batch_size=10000, max_face_batch_size=1000):
        """
        Compute the occupancy of the query point relative to the mesh
        Args:
            query_points: Tensor of shape (N,3) or (B, N, 3) where N is the number of query points
            gather_label: Tensor of shape (B, ) where C is the number of components
            max_query_point_batch_size: int, the maximum batch size for query points
            
            
        Returns:
            occupancy: Tensor of shape (B, N)
        """
        

        if len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0).repeat(self.face_coord.shape[0],1,1)# [B, N, 3]
        else:
            assert len(query_points.shape) == 3
            assert query_points.shape[0] == self.face_coord.shape[0] # [B, N, 3]

        if len(gather_label.shape) == 1:
            assert gather_label.shape[0] == self.face_coord.shape[0]*self.face_coord.shape[1]
            face_first_index = self.face_first_index
            gather_label = packed_to_padded(gather_label.float(),
                                            first_idxs=face_first_index, max_size=self.face_coord.shape[1])
            gather_label = gather_label.long()
        else:
            assert len(gather_label.shape) == 2
            assert gather_label.shape[0] == self.face_coord.shape[0]
            assert gather_label.shape[1] == self.face_coord.shape[1]
        

        occp = []

        for term in range(0, query_points.shape[-2], max_query_point_batch_size):

            points = query_points[...,term:term+max_query_point_batch_size,:]

            # occp_term = torch.zeros(points.shape[0], points.shape[1]).to(points.device)

            occp_term = torch.zeros(points.shape[0], points.shape[1], gather_label.max()+1).to(points.device)


            for face_term in range(0, self.face_coord.shape[1], max_face_batch_size):

                face_coord = self.face_coord[:,face_term:face_term+max_face_batch_size,:,:]

                gather_label_term = gather_label[:,face_term:face_term+max_face_batch_size]

                gather_label_term = torch.where(gather_label_term == -1, 0, gather_label_term)

                # face_vect = face_coord.unsqueeze(1).repeat(1,points.shape[-2],1,1,1) - points.unsqueeze(2).unsqueeze(2)
                face_vect = face_coord.unsqueeze(1).repeat(1,points.shape[-2],1,1,1) - points.unsqueeze(2).unsqueeze(2).repeat(1,1,face_coord.shape[-3],3,1)

                a_vert = face_vect[:,:,:,0,:] # [B, N, F, 3]        
                b_vert = face_vect[:,:,:,1,:] # [B, N, F, 3]
                c_vert = face_vect[:,:,:,2,:] # [B, N, F, 3]

                face_det = (a_vert * b_vert.cross(c_vert)).sum(dim=-1) # [B, N, F]

                abc = face_vect.norm(dim=-1).prod(dim=-1) # [B, N, F]

                ab = (a_vert*b_vert).sum(-1) # [B, N, F]
                bc = (b_vert*c_vert).sum(-1) # [B, N, F]
                ac = (a_vert*c_vert).sum(-1) # [B, N, F]

                solid_angle_2 = torch.arctan2(face_det, (abc + bc*a_vert.norm(dim=-1) + ac*b_vert.norm(dim=-1) + ab*c_vert.norm(dim=-1))) # [B, N, F]

                solid_angle_2_sum = scatter(solid_angle_2.permute(0,2,1) , gather_label_term, dim=1, reduce='sum', dim_size=gather_label.max()+1) # [B, C, N]

                solid_angle_2_sum = solid_angle_2_sum.permute(0,2,1) # [B, N, C]

                # solid_angle_2_solfmin = F.softmin(10*(solid_angle_2_sum*2 - (2*solid_angle_2_sum).round())**2, dim=-1) # [B, N, C]

                # occp_term += (solid_angle_2_sum * solid_angle_2_solfmin).sum(-1)

                occp_term += solid_angle_2_sum
                # solid_angle/pi/4
                
            occp.append(occp_term/np.pi/2)

        return torch.cat(occp, dim=1)



def flexible_occupancy(mesh: Meshes, pt_target : torch.Tensor, 
                       component_indx: torch.Tensor=None, allow_grad: bool = False, 
                       max_query_point_batch_size=2000, 
                       max_face_batch_size=10000):
                        
    if isinstance(mesh, trimesh.Trimesh):
        component_indx = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        component_indx = torch.from_numpy(component_indx).to(pt_target.device)

        mesh_tem = Meshes(verts=[torch.from_numpy(mesh.vertices).float().to(pt_target.device)],
                            faces=[torch.from_numpy(mesh.faces).long().to(pt_target.device)])
        
    elif isinstance(mesh, Meshes):
        assert component_indx is not None, "The component index should be provided when the input is Meshes"
        component_indx = component_indx
        mesh_tem = mesh

    solid_angle_occp = SolidAngleOccp_components(mesh_tem, allow_grad)

    mesh_verts_new = mesh_tem.verts_padded().clone() - mesh_tem.verts_normals_padded()*1e-2


    # faces_new = mesh_tem.faces_packed().clone()[non_closed_index,:]
    faces_new = mesh_tem.faces_padded().clone()
    faces_new = faces_new[:,:,[0,2,1]]


    # new_mesh = Meshes(verts=[mesh_verts_new], faces=[face_new])

    # merge the old and new mesh
    B = mesh_tem.faces_padded().shape[0]
    mesh_flipped = Meshes(verts=[mesh_verts_new[i] for i in range(B)], faces=[faces_new[i] for i in range(B)])

    if allow_grad:
        occp = solid_angle_occp(pt_target, max_face_batch_size = max_face_batch_size, 
                                max_query_point_batch_size=max_query_point_batch_size, 
                                gather_label=component_indx)
        solid_angle_occp = SolidAngleOccp_components(mesh_flipped, allow_grad)
        occp_flipped = solid_angle_occp(pt_target, max_face_batch_size = max_face_batch_size, 
                                        max_query_point_batch_size=max_query_point_batch_size, 
                                        gather_label=component_indx)
    else:
        with torch.no_grad():
            occp = solid_angle_occp(pt_target, max_face_batch_size= max_face_batch_size, 
                                    max_query_point_batch_size=max_query_point_batch_size, 
                                    gather_label=component_indx)
            solid_angle_occp = SolidAngleOccp_components(mesh_flipped, allow_grad)
            occp_flipped = solid_angle_occp(pt_target, max_face_batch_size= max_face_batch_size, 
                                            max_query_point_batch_size=max_query_point_batch_size, 
                                            gather_label=component_indx)

        B_indx, N_indx, C_indx = torch.where(((occp - occp.round()).abs()< 1e-2)*(occp.round().abs() > 0.5))

        occp_result = torch.zeros_like(occp[...,0])

        occp_result[B_indx, N_indx] = 1.0   

        occp_result = torch.where(occp_result == 0, (occp_flipped+occp).sum(-1).abs(), occp_result)

        occp_result = torch.sigmoid(10*(occp_result - 0.5))

    return occp_result



### -------------Rapid Occupancy Extraction (non-differentiable)--------------------------------------------


@torch.no_grad()
def get_sdf_from_mesh(mesh: Meshes, query_points: torch.Tensor, threshold=0.5, open_thinkness=1e-2):
    """
    Compute the signed distance field (SDF) of the mesh at the query points.
    The SDF is positive inside the mesh and negative outside the mesh.
    Args:
        mesh: Meshes object representing the mesh.
        query_points: Tensor of shape (N, 3) giving the coordinates of the query points.
        threshold: threshold for the occupancy.
        open_thinkness: the thickness of the open surface, to determine the occupancy for a non-watertight mesh.
    Returns:
        sdf: Tensor of shape (N,) giving the signed distance field at the query points.
    """

    occp, mesh = get_occp_from_mesh(mesh, query_points, threshold, open_thinkness, return_meshes=True)


    rescale = 1e5
    udf, _ = _C.point_face_dist_forward(query_points.view(-1, 3)*rescale,
                    torch.tensor([0], device=mesh.device),
                    rescale*mesh.verts_packed()[mesh.faces_packed(),:],
                    torch.tensor([0], device=mesh.device),
                    query_points.shape[0], 1e-8)
    
    udf = udf.view(-1)/rescale
    sdf = torch.where(occp > 0.5, -udf, udf)

    return sdf

@torch.no_grad()
def get_occp_from_mesh(mesh: Meshes, query_points: torch.Tensor, threshold=0.5, 
                       open_thinkness=1e-2, return_meshes=False):
    """
    Compute the occupancy of the mesh at the query points.
    Args:
        mesh: Meshes object representing the mesh.
        query_points: Tensor of shape (N, 3) giving the coordinates of the query points.
        threshold: threshold for the occupancy.
        open_thinkness: the thickness of the open surface, to determine the occupancy for a non-watertight mesh.
        return_meshes: whether to return the meshes with the flipped faces.
    Returns:
        occp: Tensor of shape (N,) giving the occupancy at the query points.
    """
    trimesh_tem = trimesh.Trimesh(vertices=mesh.verts_packed().detach().cpu().numpy(),
                                faces=mesh.faces_packed().detach().cpu().numpy())

    # component_indx = trimesh.graph.connected_component_labels(trimesh_tem.face_adjacency)
    # component_indx = torch.from_numpy(component_indx).to(device)
    trimesh_tem_components = trimesh.graph.split(trimesh_tem, only_watertight=False)
    trimesh_tem_components = sorted(trimesh_tem_components, key=lambda x: x.faces.shape[0], reverse=True)

    occp = torch.zeros(query_points.shape[0], dtype=torch.float32, device=query_points.device)
    loop = tqdm(trimesh_tem_components)

    n_points = query_points.shape[0]
    device = query_points.device
    meshes = [] # store the meshes 
    for component_indx, component in enumerate(loop):
        # component.fix_normals()
        mesh_tem = Meshes(verts=[torch.from_numpy(component.vertices).float().to(device)],
                            faces=[torch.from_numpy(component.faces).long().to(device)])
        
        occp_component = torch.zeros(n_points).to(device)

        meshes.append(mesh_tem)
        bbox_tem = mesh_tem.get_bounding_boxes()[0]
        inbbox_indx = (
            (query_points[:, 0] >= bbox_tem[0, 0]) & (query_points[:, 0] <= bbox_tem[0, 1]) &
            (query_points[:, 1] >= bbox_tem[1, 0]) & (query_points[:, 1] <= bbox_tem[1, 1]) &
            (query_points[:, 2] >= bbox_tem[2, 0]) & (query_points[:, 2] <= bbox_tem[2, 1])
        )
        inbbox_indx = torch.where(inbbox_indx)[0]
        loop.set_description(f'Components {component_indx+1}/{len(trimesh_tem_components)}:  {mesh_tem.faces_packed().shape[0]} faces; {len(inbbox_indx)} queries')
        if len(inbbox_indx) == 0:
            continue
        with torch.no_grad():
            solid_angle_occp = SolidAngleOccp(mesh_tem, False).half()
            occp_inbbox = solid_angle_occp(query_points[inbbox_indx].half(), 
                                           max_query_point_batch_size=20000).view(-1)

        query_points_flipped = query_points[inbbox_indx]
        if not component.is_watertight:
            ## create a new mesh with the flipped faces
            mesh_verts_flipped = mesh_tem.verts_packed().clone() - mesh_tem.verts_normals_packed()*open_thinkness
            faces_flipped = mesh_tem.faces_packed().clone()
            faces_flipped = faces_flipped[:,[0,2,1]]
            mesh_flipped = Meshes(verts=[mesh_verts_flipped], faces=[faces_flipped])
            meshes.append(mesh_flipped)
            with torch.no_grad():
                solid_angle_occp_flipped = SolidAngleOccp(mesh_flipped, False).half()
                occp_flipped = solid_angle_occp_flipped(query_points_flipped.half(), 
                                                        max_query_point_batch_size=20000).view(-1)
            occp_inbbox = occp_flipped+occp_inbbox

        occp_component[inbbox_indx] = torch.sigmoid(10*(occp_inbbox.abs() - threshold))
        
        # Update the occupancy
        occp = torch.max(occp, occp_component)

        # occp = torch.sigmoid(10*(occp.abs() - 0.5))

        occp = torch.where(occp.abs() > threshold, 1., 0.) 

    if return_meshes:
        meshes = Meshes([mesh.verts_packed().clone() for mesh in meshes], [mesh.faces_packed().clone() for mesh in meshes])
        return occp, meshes
    return occp

def get_flipped_mesh(mesh: Meshes, open_thinkness=1e-2):
    """
    Get the mesh with the flipped faces for non-watertight components.
    Args:
        mesh: Meshes object representing the mesh.
        open_thinkness: the thickness of the open surface, to determine the occupancy for a non-watertight mesh.
    Returns:
        mesh_flipped: Meshes object representing the mesh with the flipped faces.
    """
    trimesh_tem = trimesh.Trimesh(vertices=mesh.verts_packed().detach().cpu().numpy(),
                                faces=mesh.faces_packed().detach().cpu().numpy())

    # component_indx = trimesh.graph.connected_component_labels(trimesh_tem.face_adjacency)
    # component_indx = torch.from_numpy(component_indx).to(device)
    trimesh_tem_components = trimesh.graph.split(trimesh_tem, only_watertight=False)
    trimesh_tem_components = sorted(trimesh_tem_components, key=lambda x: x.faces.shape[0], reverse=True)

    meshes = []
    device = mesh.device
    for component_indx, component in enumerate(trimesh_tem_components):
        component.fix_normals()

        mesh_tem = Meshes(verts=[torch.from_numpy(component.vertices).float().to(device)],
                            faces=[torch.from_numpy(component.faces).long().to(device)])
        
        meshes.append(mesh_tem)
        if not component.is_watertight:
            ## create a new mesh with the flipped faces
            mesh_verts_flipped = mesh_tem.verts_packed().clone() - mesh_tem.verts_normals_packed()*open_thinkness
            faces_flipped = mesh_tem.faces_packed().clone()
            faces_flipped = faces_flipped[:,[0,2,1]]
            mesh_flipped = Meshes(verts=[mesh_verts_flipped], faces=[faces_flipped])
            meshes.append(mesh_flipped)

    meshes = Meshes([mesh.verts_packed().clone() for mesh in meshes], [mesh.faces_packed().clone() for mesh in meshes])
    meshes = Meshes(verts=[meshes.verts_packed().clone()], faces=[meshes.faces_packed().clone()])
    return meshes