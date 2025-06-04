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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import cv2
from tqdm import tqdm
from PIL import Image
import io
import os

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def read_exr(path):
    arr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        raise RuntimeError(f"Failed to read\n\t{path}")
    # RGB
    if arr.ndim == 3 or arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return rgb
    raise NotImplementedError(arr.shape)

def read_hdr(path):
    with open(path, 'rb') as h:
        buffer_ = np.fromstring(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cv2tColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def depth2rgb(depth, mask):
    sort_d = torch.sort(depth[mask.to(torch.bool)])[0]
    min_d = sort_d[len(sort_d) // 100 * 5]
    max_d = sort_d[len(sort_d) // 100 * 95]
    # min_d = 2.8
    # max_d = 4.6
    # print(min_d, max_d)
    depth = (depth - min_d) / (max_d - min_d) * 0.9 + 0.1
    viridis = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 256)))
    depth_np = depth.detach().cpu().numpy()[0]
    depth_draw = viridis(depth_np)[..., :3]
    # print(viridis(depth.detach().cpu().numpy()).shape, depth_draw.shape, mask.shape)
    depth_draw = torch.from_numpy(depth_draw).to(depth.device).permute([2, 0, 1]) * mask
    return depth_draw


def weight2rgb(weight, mask):
    weight_draw = weight.expand(3,-1,-1)
    #print("material_draw",material_draw.shape)
    weight_draw = weight_draw * mask
    magma = ListedColormap(plt.cm.magma(np.linspace(0, 1, 256)))
    weight_draw = magma(weight_draw.detach().cpu().numpy()[0])[..., :3]
    weight_draw = torch.from_numpy(weight_draw).to(weight.device).permute([2, 0, 1]) * mask
    return weight_draw

def resize_image(img, factor, mode='bilinear'):
    # print(type(img))
    if factor == 1:
        return img
    is_np = type(img) == np.ndarray
    if is_np:
        resize = torch.from_numpy(img)
    else:
        resize = img.clone()
    dtype = resize.dtype
    resize = torch.nn.functional.interpolate(resize[None].to(torch.float32), scale_factor=1/factor, mode=mode)[0].to(dtype)
    if is_np:
        resize = resize.numpy()
    # print(type(img))
    return resize

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    cor_image = image ** inv_gamma
    return cor_image

def read_mat_file(filename):
    """
    :return: Normal_ground truth in shape: (height, width, 3)
    """
    mat = sio.loadmat(filename)
    gt_n = mat['Normal_gt']
    return gt_n.astype(np.float32)

def sphere_basis(base_color, roughness, metallic, ball_normal, valid_idx):
    view = torch.zeros_like(ball_normal)
    view[..., 2] = 1
    light1 = torch.ones_like(ball_normal)
    light1[...,2] = 1
    light2 = torch.ones_like(ball_normal)*-1
    light2[...,2] = -0.5
    light1 = F.normalize(light1, dim=-1)
    light2 = F.normalize(light2, dim=-1)
    view = F.normalize(view, dim=-1)
    normal = F.normalize(ball_normal, dim=-1)
    basis_imgs = []
    for basis_idx in range(base_color.shape[0]):
        basis_map = torch.zeros((512, 612, 3), dtype=torch.float32) # black background
        base = base_color[basis_idx].repeat(512, 612, 1)
        rough = roughness[basis_idx].repeat(512, 612, 1)
        metal = metallic[basis_idx].repeat(512, 612, 1)
        for light in [light1, light2]:
            basis = rendering_equation_python(base, rough, metal, normal, view, light)
            basis_map[valid_idx] += basis[valid_idx]
        basis_map = basis_map.clip(0,1)
        basis_imgs.append(basis_map.permute(2,0,1)) 
    return basis_imgs

def rendering_equation_python(base_color, roughness, metallic, normals, viewdirs, incident_dirs):

    base_color = base_color.unsqueeze(-2).contiguous()
    roughness = roughness.unsqueeze(-2).contiguous()
    metallic = metallic.unsqueeze(-2).contiguous()
    normals = normals.unsqueeze(-2).contiguous()
    viewdirs = viewdirs.unsqueeze(-2).contiguous()
    incident_dirs = incident_dirs.unsqueeze(-2).contiguous()

    def _dot(a, b):
        return (a * b).sum(dim=-1, keepdim=True)  

    def _f_diffuse(base_color, metallic):
        return (1 - metallic) * base_color / np.pi  

    def _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic):
        # used in SG, wrongly normalized
        def _d_sg(r, cos):
            r2 = (r * r).clamp(min=1e-7)
            amp = 1 / (r2 * np.pi)
            sharp = 2 / r2
            return amp * torch.exp(sharp * (cos - 1))

        D = _d_sg(roughness, h_d_n)
       
        # Fresnel term F
        F_0 = 0.04 * (1 - metallic) + base_color * metallic  
        F = F_0 + (1.0 - F_0) * ((1.0 - h_d_o) ** 5)  

        # geometry term V, we use V = G / (4 * cos * cos) here
        def _v_schlick_ggx(r, cos):
            r2 = ((1 + r) ** 2) / 8
            return 0.5 / (cos * (1 - r2) + r2).clamp(min=1e-7)

        V = _v_schlick_ggx(roughness, n_d_i) * _v_schlick_ggx(roughness, n_d_o) 

        return D * F * V

    # half vector and all cosines
    half_dirs = incident_dirs + viewdirs
    half_dirs = F.normalize(half_dirs, dim=-1)

    h_d_n = _dot(half_dirs, normals).clamp(min=0)
    h_d_o = _dot(half_dirs, viewdirs).clamp(min=0)
    n_d_i = _dot(normals, incident_dirs).clamp(min=0)
    n_d_o = _dot(normals, viewdirs).clamp(min=0)
    
    f_d = _f_diffuse(base_color, metallic)
    f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic)

    transport = n_d_i
    rgb_d = (f_d * transport).mean(dim=-2)
    rgb_s = (f_s * transport).mean(dim=-2)
    rgb = rgb_d + rgb_s
    
    return rgb

def rendering_equation_theta_h(base_color, roughness, metallic, theta_h):

    base_color = base_color.unsqueeze(-2).contiguous()
    roughness = roughness.unsqueeze(-2).contiguous()
    metallic = metallic.unsqueeze(-2).contiguous()

    def _dot(a, b):
        return (a * b).sum(dim=-1, keepdim=True)  

    def _f_diffuse(base_color, metallic):
        return (1 - metallic) * base_color / np.pi  

    def _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic):
        # used in SG, wrongly normalized
        def _d_sg(r, cos):
            r2 = (r * r).clamp(min=1e-7)
            amp = 1 / (r2 * np.pi)
            sharp = 2 / r2
            return amp * torch.exp(sharp * (cos - 1))

        D = _d_sg(roughness, h_d_n)
       
        # Fresnel term F
        F_0 = 0.04 * (1 - metallic) + base_color * metallic  
        F = F_0 + (1.0 - F_0) * ((1.0 - h_d_o) ** 5)  

        # geometry term V, we use V = G / (4 * cos * cos) here
        def _v_schlick_ggx(r, cos):
            r2 = ((1 + r) ** 2) / 8
            return 0.5 / (cos * (1 - r2) + r2).clamp(min=1e-7)

        V = _v_schlick_ggx(roughness, n_d_i) * _v_schlick_ggx(roughness, n_d_o)  

        return D * F * V

    h_d_n = torch.cos(theta_h).clamp(min=0)
    h_d_o = torch.ones_like(theta_h)
    n_d_i = torch.cos(theta_h).clamp(min=0)
    n_d_o = torch.cos(theta_h).clamp(min=0)
    
    f_d = _f_diffuse(base_color, metallic)
    f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic)

    transport = n_d_i
    rgb_d = f_d
    rgb_s = f_s
    rgb = (rgb_d + rgb_s) * transport
    
    return rgb
