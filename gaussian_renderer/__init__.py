#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer, RenderEquation
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal
from utils.general_utils import points2view

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,vis =False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform, # W2C matrix
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    base_color = pc.get_base_color
    roughness = pc.get_roughness
    metallic = pc.get_metallic
    weights = pc.get_weight
    normal = pc.get_normal
    N = normal.shape[0]
    intensity = viewpoint_camera.light_intensity
    light_position = viewpoint_camera.light_position
    num_basis = base_color.shape[0]
    incidents = light_position - means3D
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
    incident_dirs = F.normalize(incidents, dim=-1)

    cos = (viewdirs * normal).sum(dim=-1) 
    normal[cos<0]*= -1
    half_dirs = (incident_dirs + viewdirs)/2
    half_dirs = F.normalize(half_dirs, dim=-1)
    cos_theta_h = torch.sum(half_dirs * normal, dim=1, keepdim=True) 
    total_brdf = []  
    for i in range(num_basis):
        base = base_color[i].repeat(N,1)
        rough = roughness[i].repeat(N,1)
        metal = metallic[i].repeat(N,1)
        basis_brdf = RenderEquation(base, rough, metal, normal, viewdirs, incident_dirs, debug=False) 
        total_brdf.append(basis_brdf[0])
    total_brdf= torch.stack(total_brdf).permute(1,0,2) 
    basis_color = torch.einsum('ij,ijk->ijk',weights,total_brdf) 
    brdf_color = torch.sum(basis_color,dim=1) 
    dist = torch.norm(incidents.detach(),dim=-1,keepdim=True)
    fall_off = (torch.median(dist)**2)/(dist**2)
    brdf = brdf_color*fall_off*intensity
    basis_color = basis_color.reshape(-1,num_basis*3)*fall_off*intensity

    # visualization of basis BRDF weight maps and images
    if vis == True:
        features = torch.cat([cos_theta_h, weights,basis_color], dim=-1) 
    else:
        features = cos_theta_h

    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        brdf = brdf,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        features = features)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5] 
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1) #C2W
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # get theta_h map
    theta_map = allmap[7:8]

    #basis brdf imgs
    if vis == True:
        weight_imgs = allmap[8:8+num_basis]
        weight_imgs = weight_imgs.reshape(-1,1,viewpoint_camera.image_height,viewpoint_camera.image_width)

        brdf_imgs = allmap[8+num_basis:8+num_basis*4]
        brdf_imgs = brdf_imgs.reshape(-1,3,viewpoint_camera.image_height,viewpoint_camera.image_width)

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal, # normal image
            'rend_dist': render_dist, # depth distortion image
            'surf_depth': surf_depth, # depth image
            'surf_normal': surf_normal, # depth to normal image
            'theta_map': theta_map}) # theta_h image
    
    if vis == True: 
        rets.update({
                'brdf_imgs': brdf_imgs, # basis BRDF images
                'weight_imgs': weight_imgs}) # basis BRDF weight images

    return rets