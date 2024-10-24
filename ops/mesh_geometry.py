import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

from pytorch3d.structures import Meshes
from pytorch3d.ops import mesh_face_areas_normals, packed_to_padded
from einops import rearrange, einsum, repeat

import numpy as np

from pytorch3d import _C

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



def Winding_Occupancy(mesh_tem: Meshes, points: torch.Tensor, max_v_per_call=2000):
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


def Winding_Occupancy_Face(mesh_tem: Meshes, points: torch.Tensor, max_f_per_call=2000):
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





class Differentiable_Voxelizer(nn.Module):
    def __init__(self, bbox_density=128, integrate_method='vertex'):
        super(Differentiable_Voxelizer, self).__init__()
        self.bbox_density = bbox_density
        if integrate_method == 'face':
            self.Winding_Occupancy = Winding_Occupancy_Face
        elif integrate_method == 'vertex':
            self.Winding_Occupancy = Winding_Occupancy

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

            occupency_temp = self.Winding_Occupancy(mesh_src, tem_charge, max_v_per_call=max_v_per_call)

            if if_binary:
                occupency_temp = torch.sigmoid((self.Winding_Occupancy(mesh_src, tem_charge,max_v_per_call=max_v_per_call)-0.5)*100)

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
            occupancy = Winding_Occupancy_Face(mesh, points, max_f_per_call=2000).view(-1,1)
        else:
            occupancy = Winding_Occupancy(mesh, points, max_v_per_call=2000).view(-1,1)
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
        

    
class Differentiable_SDF(nn.Module):
    def __init__(self, bbox_density=128, integrate_method='face'):
        super(Differentiable_Voxelizer, self).__init__()
        self.bbox_density = bbox_density
        if integrate_method == 'face':
            self.Winding_Occupancy = Winding_Occupancy_Face
        elif integrate_method == 'vertex':
            self.Winding_Occupancy = Winding_Occupancy

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

            occupency_temp = self.Winding_Occupancy(mesh_src, tem_charge,max_v_per_call=max_v_per_call)

            if if_binary:
                occupency_temp = torch.sigmoid((self.Winding_Occupancy(mesh_src, tem_charge,max_v_per_call=max_v_per_call)-0.5)*100)

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
    



def normalize_mesh(mesh, rescalar=1.1):
    bbox = mesh.get_bounding_boxes()
    center = (bbox[:, :, 0] + bbox[:, :, 1]) / 2
    center = center.unsqueeze(-2)
    size = (bbox[:, :, 1] - bbox[:, :, 0]).max(dim=-1)[0]
    
    scale = 2.0 / (size*rescalar+1e-8).unsqueeze(-1).unsqueeze(-1)

    mesh = mesh.update_padded((mesh.verts_padded()-center)*scale)
    return mesh




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
            query_points = query_points.unsqueeze(0) # [1, N, 3]
        else:
            assert len(query_points.shape) == 3
            assert query_points.shape[0] == self.face_coord.shape[0] # [B, N, 3]

        occp = []

        for term in range(0, query_points.shape[-2], max_query_point_batch_size):

            points = query_points[...,term:term+max_query_point_batch_size,:]

            occp_term = torch.zeros(points.shape[0], points.shape[1]).to(points.device)

            for face_term in range(0, self.face_coord.shape[1], max_face_batch_size):

                face_coord = self.face_coord[:,face_term:face_term+max_face_batch_size,:,:]

                face_vect = face_coord.unsqueeze(1).repeat(1,points.shape[0],1,1,1) - points.unsqueeze(2).unsqueeze(2)

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