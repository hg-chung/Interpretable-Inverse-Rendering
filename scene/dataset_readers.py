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

import os
import sys
import torch
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import imageio
import skimage
import cv2
import torch.nn.functional as F
from tqdm import tqdm

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    cx: np.array
    cy: np.array
    light_position: np.array
    light_intensity: float
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    init_colors: np.array
    light_scale: float


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    camera_centers = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            cx = intr.params[1]
            cy = intr.params[2]
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx = intr.params[2]
            cy = intr.params[3]
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image = Image.open(image_path)
        image_name = os.path.basename(image_path).split(".")[0]

        im_data = np.array(image.convert("RGBA"))

        norm_data = im_data / 255.0
        mask = norm_data[..., 3:]
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4]
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        mask = mask.transpose([2, 0, 1]).astype(np.float32)
        light_position = np.array([-0.01,0.0,0.0])
        light_intensity = 0.6

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, light_position=light_position, light_intensity=light_intensity, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, mask=mask)
        cam_infos.append(cam_info)
        camera_centers.append(qvec2rotmat(extr.qvec))
    sys.stdout.write('\n')
    camera_centers = np.hstack(camera_centers).T
    return cam_infos, camera_centers

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=6):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted, camera_centers = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    init_colors = np.load(os.path.join(path, "init_colors_12.npy"))
    np.random.seed(40)
    eval_idx = np.random.permutation(len(cam_infos))[:(len(cam_infos)//llffhold)]

    #eval= False

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in eval_idx]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in eval_idx]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []


    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    points = np.asarray(pcd.points)
    light_dist = (camera_centers - points.mean(axis=0).reshape(1,3)).mean(axis=0)
    light_scale = np.linalg.norm(light_dist)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, init_colors = init_colors,
                           light_scale = light_scale)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    camera_centers = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        light_intensity = contents["light_intensity"]
        light_position = contents["light_position"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:]
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            mask = mask.transpose([2, 0, 1]).astype(np.float32)
            cx = image.size[0] / 2
            cy = image.size[0] / 2
            camera_centers.append(c2w[:3,3:4])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, light_position=light_position, light_intensity=light_intensity, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], mask=mask))
    camera_centers = np.hstack(camera_centers).T
    return cam_infos, camera_centers

def readNerfSyntheticInfo(path, white_background, eval, init, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos, train_camera_centers = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos, test_camera_centers = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    init_colors = np.load(os.path.join(path, "init_colors_12.npy"))
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if init:
        print(f"Generating point cloud")
        imgs ,poses, [H,W,focal] = load_blender_data(path)
        K = np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]])
        train_n = imgs.shape[0]
        poses = torch.tensor(poses).cuda()[:train_n]
        images = torch.tensor(imgs).cuda()[:train_n] 

        pc,color,N = [],[],H
        [xs,ys,zs],[xe,ye,ze] = [-1,-1,-1],[1,1,1]
        pts_all = []
        for h_id in tqdm(range(N)):
            i, j = torch.meshgrid(torch.linspace(xs, xe, N).cuda(),
                                    torch.linspace(ys, ye, N).cuda()) 
            i, j = i.t(), j.t()
            pts = torch.stack([i, j, torch.ones_like(i).cuda()], -1)
            pts[...,2] = h_id / N * (ze - zs) + zs 
            pts_all.append(pts.clone())
            uv = batch_get_uv_from_ray(H,W,K,poses,pts) 
            result = F.grid_sample(images.permute(0, 3, 1, 2).float(), uv).permute(0,2,3,1) 
            margin = 0.05
            result[(uv[..., 0] >= 1.0) * (uv[..., 0] <= 1.0 + margin)] = 1
            result[(uv[..., 0] >= -1.0 - margin) * (uv[..., 0] <= -1.0)] = 1
            result[(uv[..., 1] >= 1.0) * (uv[..., 1] <= 1.0 + margin)] = 1
            result[(uv[..., 1] >= -1.0 - margin) * (uv[..., 1] <= -1.0)] = 1
            result[(uv[..., 0] <= -1.0 - margin) + (uv[..., 0] >= 1.0 + margin)] = 0
            result[(uv[..., 1] <= -1.0 - margin) + (uv[..., 1] >= 1.0 + margin)] = 0

            img = ((result>0.).sum(0)[...,0]>train_n-1).float()
            pc.append(img)
            color.append(result.mean(0))
        pc = torch.stack(pc,-1)
        color = torch.stack(color,-1)
        r, g, b = color[:, :, 0], color[:, :, 1], color[:, :, 2]
        idx = torch.where(pc > 0)
        color = torch.stack((r[idx],g[idx],b[idx]),-1)
        idx = (idx[1],idx[0],idx[2])
        pts = torch.stack(idx,-1).float()/N
        pts[:,0] = pts[:,0]*(xe-xs)+xs
        pts[:,1] = pts[:,1]*(ye-ys)+ys
        pts[:,2] = pts[:,2]*(ze-zs)+zs
        print("initiallized pts",pts.shape[0])
        ids = np.random.permutation(pts.shape[0])[:20000] 
        pts = pts[ids]
        color= color[ids]
        num_pts =pts.shape[0]
        normal = np.random.random((num_pts, 3)) - 0.5
        normal /= np.linalg.norm(normal, 2, 1, True)
        pcd = BasicPointCloud(points=pts.detach().cpu().numpy(), colors=color.detach().cpu().numpy(), normals=normal)
        
    else:
        pts = torch.rand(10000,3)*2.0 - 1.0
        color = torch.rand(10000,3)/255
        normal = np.random.random((10000, 3)) - 0.5
        pcd = BasicPointCloud(points=pts.detach().cpu().numpy(), colors=color.detach().cpu().numpy(), normals=normal)
    
    points = np.asarray(pcd.points)
    light_dist = (train_camera_centers - points.mean(axis=0).reshape(1,3)).mean(axis=0)
    light_scale = np.linalg.norm(light_dist)
    print("light_scale", light_scale)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,init_colors=init_colors,
                           light_scale = light_scale)
    return scene_info

def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train'] # data type
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir,frame['file_path'] + '.png') 
            imgs.append(imageio.imread(fname)) 
            poses.append(np.array(frame['transform_matrix'])) 
        imgs = (np.array(imgs) / 255.).astype(np.float32) 
        poses = np.array(poses).astype(np.float32) 
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    print("imgs shape", imgs.shape)
        
    return imgs, poses, [H, W, focal]

def batch_get_uv_from_ray(H,W,K,poses,pts):
    RT = (poses[:, :3, :3].transpose(1, 2)) 
    pts_local = torch.sum((pts[..., None, :] - poses[:, :3, -1])[..., None, :] * RT, -1) # move points world to camera space  (N,N,batch,3)
    pts_local = pts_local / (-pts_local[..., -1][..., None] + 1e-7) 
    u = pts_local[..., 0] * K[0][0] + K[0][2] 
    v = -pts_local[..., 1] * K[1][1] + K[1][2] #
    uv0 = torch.stack((u, v), -1)
    uv0[...,0] = uv0[...,0]/W*2-1 
    uv0[...,1] = uv0[...,1]/H*2-1
    uv0 = uv0.permute(2,0,1,3)
    return uv0


def load_mask(path):
    alpha = imageio.imread(path, pilmode='F')
    alpha = skimage.img_as_float32(alpha) / 255
    return alpha

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
