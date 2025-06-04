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
import random
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import build_scaling_rotation
from utils.image_utils import rendering_equation_theta_h
from sklearn.cluster import KMeans
from pytorch3d.loss import chamfer_distance
#random.seed(0)

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.base_color_activation = torch.sigmoid
        self.roughness_activation = torch.sigmoid
        self.metallic_activation = torch.sigmoid
        self.weight_activation = torch.nn.Softmax(dim=-1) 

    def __init__(self):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._base_color = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metallic = torch.empty(0)
        self._weight = torch.empty(0)
        self.num_basis = 12
        self.optimizer = None
        self.percent_dense = 0
        self.temp = 0.0125
        self.delete_basis = torch.empty(0)
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self._base_color,
            self._roughness,
            self._metallic,
            self._weight,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self._xyz, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_normal(self):
        rot_z = build_rotation(self._rotation)[...,2]
        return rot_z
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_base_color(self):
        return torch.clamp(self.base_color_activation(self._base_color),0.05,1.0)

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness) #* 0.5

    @property
    def get_metallic(self):
        return self.metallic_activation(self._metallic)
    
    @property
    def get_weight(self):
        weight = self.weight_activation(self._weight/self.temp)
        return weight
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, init_base_color, light_scale):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        self.init_colors = init_base_color.float().to("cuda")
        kmeans = KMeans(n_clusters=self.num_basis, random_state=0).fit(init_base_color)
        self.representative_colors = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device="cuda")
        self.light_scale = light_scale
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        print("number of Representative colors : ", self.representative_colors.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2 / 4))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        base_color = inverse_sigmoid(self.representative_colors)
        roughness = torch.zeros((self.num_basis,1), dtype=torch.float, device="cuda")
        metallic = torch.ones((self.num_basis,1), dtype=torch.float, device="cuda") * -2
        weight = torch.zeros((fused_point_cloud.shape[0],self.num_basis),device="cuda")

        self._base_color = nn.Parameter(base_color.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._metallic = nn.Parameter(metallic.requires_grad_(True))
        self._weight = nn.Parameter(weight.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.delete_basis = torch.zeros((self.num_basis),dtype=torch.bool, device="cuda")
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opac_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._base_color], 'lr': training_args.base_color_lr, "name": "base_color"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
            {'params': [self._weight], 'lr': training_args.weight_lr, "name": "weight"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._weight.shape[1]):
            l.append('weight_{}'.format(i))
        return l
    
    def construct_list_of_basis_attributes(self):
        l = ['base_color_r', 'base_color_g','base_color_b', 'roughness', 'metallic']
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        base_color = self._base_color.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()
        weight = self._weight.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        basis_dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_basis_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, opacities, scale, rotation, weight), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        basis_elements = np.empty(base_color.shape[0],dtype=basis_dtype_full)
        basis_attributes = np.concatenate((base_color, roughness, metallic), axis=1)
        basis_elements[:] = list(map(tuple, basis_attributes))
        el = PlyElement.describe(elements, 'vertex')
        mat = PlyElement.describe(basis_elements, 'basis')
        PlyData([el, mat]).write(path)
    
    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def load_ply(self, path):
        plydata = PlyData.read(path)
        num_basis = len(plydata.elements[1])

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        weights = np.zeros((xyz.shape[0], num_basis))
        base_color = np.zeros((num_basis, 3))
        roughness = np.zeros((num_basis, 1))
        metallic = np.zeros((num_basis, 1))

        for i in range(num_basis):
            weights[:, i] = np.asarray(plydata.elements[0]['weight_{}'.format(i)])
            base_color[i,0] = np.asarray(plydata.elements[1]["base_color_r"][i])
            base_color[i,1] = np.asarray(plydata.elements[1]["base_color_g"][i])
            base_color[i,2] = np.asarray(plydata.elements[1]["base_color_b"][i])
            roughness[i] = np.asarray(plydata.elements[1]["roughness"][i])
            metallic[i] = np.asarray(plydata.elements[1]["metallic"][i])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._weight = nn.Parameter(torch.tensor(weights, dtype=torch.float, device="cuda").requires_grad_(True))
        self._base_color = nn.Parameter(torch.tensor(base_color, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def _prune_basis(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "base_color" or group["name"] == "roughness" or group["name"] == "metallic":
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

            elif group["name"] == "weight":
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][:,mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][:,mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][:,mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][:,mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
            else:
                continue

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "base_color" or group["name"] == "roughness" or group["name"] == "metallic":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._weight = optimizable_tensors["weight"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.scale_gradient_accum = self.scale_gradient_accum[valid_points_mask]
        self.opac_gradient_accum = self.opac_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "base_color" or group["name"] == "roughness" or group["name"] == "metallic":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def cat_basis_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "base_color" or group["name"] == "roughness" or group["name"] == "metallic":
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
            elif group["name"] == "weight":
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=1)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=1)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=1).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=1).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
            else:
                continue

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation, new_weight=None):
        d = {"xyz": new_xyz,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "weight": new_weight}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._weight = optimizable_tensors["weight"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opac_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def basis_densification_postfix(self, new_base_color, new_roughness, new_metallic, new_weight):
        d = {"base_color": new_base_color,
        "roughness": new_roughness,
        "metallic" : new_metallic,
        "weight": new_weight}

        optimizable_tensors = self.cat_basis_to_optimizer(d)
        self._base_color = optimizable_tensors["base_color"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._weight = optimizable_tensors["weight"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, pre_mask=True):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_weight = self._weight[selected_pts_mask].repeat(N, 1)


        self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation, new_weight)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, pre_mask=True):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        selected_pts_mask *= pre_mask
        
        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_weight = self._weight[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_opacities, new_scaling, new_rotation, new_weight)
    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def merge_brdf(self, merge_threshold):
        base_color = self.get_base_color 
        roughness = self.get_roughness 
        metallic = self.get_metallic 
        weight = self.get_weight 
        positions = self.get_xyz

        # point cloud for each basis BRDF
        max_values, max_indices = torch.max(weight, dim=-1)
        basis_pc = []
        num_basis = weight.shape[1]
        for idx in range(num_basis):
            max_idx = torch.where(max_values == weight[:,idx])[0]
            basis_pc.append(positions[max_idx].cuda())

        cd_dist = torch.eye(num_basis, num_basis, device="cuda") * 10
        brdf_dist = torch.eye(num_basis, num_basis, device="cuda") * 10
        
        # sample theta_h
        S = 90
        theta_h = torch.linspace(0, torch.pi/2 , S, device="cuda")
        theta_h = theta_h.unsqueeze(-1)

        # BRDF intensity computation for basis BRDFs
        intensity = rendering_equation_theta_h(base_color, roughness, metallic, theta_h)
        N,S,C = intensity.shape # N: number of basis, S: number of theta_h, C: number of channels

        for i in range(num_basis):
            for j in range(num_basis):
                if j>i:
                    cd_dist[i,j] = chamfer_distance(basis_pc[i].unsqueeze(0), basis_pc[j].unsqueeze(0), point_reduction = 'mean')[0] # chamfer distance
                    brdf_dist[i,j] = torch.norm(intensity[i]-intensity[j], p=2, dim=-1).mean()
        cd_dist = (cd_dist + cd_dist.T)/2 # geometric difference
        brdf_dist = (brdf_dist + brdf_dist.T)/2 # radiometric difference
        
        sorted_values, sorted_indices = torch.sort(brdf_dist.view(-1))
        #print("brdf_dist", brdf_dist)
        #print("cd dist", cd_dist)
        rows = sorted_indices // brdf_dist.size(1)
        cols = sorted_indices % brdf_dist.size(1)
        for idx, value in enumerate(sorted_values):
            sidx1, sidx2 = rows[idx], cols[idx]
            if value < merge_threshold and (cd_dist[sidx1,sidx2] == torch.min(cd_dist[sidx1])): # satisfy conditions
                print("Merged idx1: {}, idx2: {}, radio_difference: {}, geo_difference".format(sidx1, sidx2, value, cd_dist[sidx1,sidx2]))
                basis_keep = torch.arange(base_color.shape[0],device="cuda")
                if len(basis_pc[sidx1]) > len(basis_pc[sidx2]): # merge basis BRDFs and weights
                    basis_keep = basis_keep[basis_keep != sidx2]
                    self._weight[:,sidx1] = torch.log(torch.exp(self._weight[:, sidx1]/self.temp)+torch.exp(self._weight[:, sidx2]/self.temp))*self.temp
                else:
                    basis_keep = basis_keep[basis_keep != sidx1]
                    self._weight[:,sidx2] = torch.log(torch.exp(self._weight[:, sidx1]/self.temp)+torch.exp(self._weight[:, sidx2]/self.temp))*self.temp

                optimizable_tensors = self._prune_basis(basis_keep)

                self._base_color = optimizable_tensors["base_color"]
                self._roughness = optimizable_tensors["roughness"]
                self._metallic = optimizable_tensors["metallic"]
                self._weight = optimizable_tensors["weight"]

                print("number of basis BRDFs", self._base_color.shape[0])
                self.delete_basis = torch.zeros((self._base_color.shape[0]), dtype=torch.bool, device="cuda")
                break

    def delete_brdf(self):
        base_color = self.get_base_color
        delete_basis = self.delete_basis
        
        # delete unnecessary basis BRDF
        if not torch.all(delete_basis):
            del_idx = torch.nonzero(~delete_basis)[0] 
            basis_keep = torch.arange(base_color.shape[0],device="cuda")
            basis_keep = basis_keep[basis_keep != del_idx]
            optimizable_tensors = self._prune_basis(basis_keep)

            self._base_color = optimizable_tensors["base_color"]
            self._roughness = optimizable_tensors["roughness"]
            self._metallic = optimizable_tensors["metallic"]
            self._weight = optimizable_tensors["weight"]
            print("deleted idx", del_idx.item())
            print("number of basis BRDFs", self._base_color.shape[0]) 

        self.delete_basis = torch.zeros((self._base_color.shape[0]), dtype=torch.bool, device="cuda")

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.scale_gradient_accum[update_filter] += self._scaling.grad[update_filter, :2].sum(1, True)
        self.opac_gradient_accum[update_filter] += self._opacity[update_filter]
        self.denom[update_filter] += 1

    
    # check unnecessary basis BRDF with weight images
    def add_weight_stats(self, weight_imgs, mask):
        val_pixels = weight_imgs[mask[0]]  # Reshaping after masking
        value = (val_pixels > 0.1).sum(dim=0).float()/mask.sum() # threshold for valid pixel: 0.1
        self.delete_basis[value > 5e-3] = True # threshold for the number of valid pixels : 5e-3
