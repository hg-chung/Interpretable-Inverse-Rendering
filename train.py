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
import torch
import numpy as np
import cv2
from random import randint
from utils.loss_utils import l1_loss, ssim, H_loss, theta_l1_loss
from gaussian_renderer import render, network_gui
from torchvision.utils import save_image
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, weight2rgb, depth2rgb, sphere_basis, read_mat_file,gamma_correction
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, merge_iterations, delete_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel()
    scene = Scene(dataset, gaussians,shuffle=False,resolution_scales=[1, 2, 4])
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    opt.densification_interval = max(opt.densification_interval, len(scene.getTrainCameras()))
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    print("threshold, weight", opt.basis_merge_threshold, opt.lambda_weight_img_sparse)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        # Pick a random Camera
        scale = 1.0

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale).copy()[:]
            num_input = len(viewpoint_stack)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, vis =True)
        image, viewspace_point_tensor, visibility_filter, radii, weight_imgs, opac, theta_h = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], \
            render_pkg["weight_imgs"], render_pkg["rend_alpha"], render_pkg["theta_map"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        gt_mask = viewpoint_cam.gt_alpha_mask
        gt_mask = gt_mask.cuda()

        if iteration >= 1000:
            theta_h = torch.clamp(theta_h.detach(),0.0,1.0)
            theta_h_pow = torch.pow(theta_h,20)
            Ll1 = theta_l1_loss(image, gt_image, 1+5*theta_h_pow)
            ssim_loss = (1.0 - ssim(image, gt_image))*(1+5*theta_h_pow)
        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = 1.0 - ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss.mean()
        o = opac.clamp(1e-6,1-1e-6)
        mask_loss = -(gt_mask*torch.log(o)+(1-gt_mask)*torch.log(1-o)).mean()
        mask_vis = (opac.detach() > 1e-3)

        # regularization
        lambda_normal = opt.lambda_normal 
        lambda_dist = opt.lambda_dist if iteration > 5000 else 0.0
        lambda_weight_sparse= opt.lambda_weight_sparse if iteration >5000 else 1e-8 
        lambda_weight_img_sparse = opt.lambda_weight_img_sparse if iteration > 9000 else 1e-8 

        loss_weight_sparse =lambda_weight_sparse * H_loss(scene.gaussians.get_weight)
        num_basis = scene.gaussians.get_base_color.shape[0]
        weight_imgs = weight_imgs.squeeze().permute(1,2,0)
        masked_weight_imgs = weight_imgs[mask_vis[0]]
        loss_weight_img_sparse = lambda_weight_img_sparse * H_loss(masked_weight_imgs.reshape(-1,num_basis))



        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss +  normal_loss + dist_loss +  loss_weight_sparse + loss_weight_img_sparse + mask_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, dist_loss, normal_loss, mask_loss, loss_weight_sparse, loss_weight_img_sparse, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                for delete_iter in delete_iterations:
                    if delete_iter-num_input <= iteration < delete_iter:
                        gaussians.add_weight_stats(weight_imgs,mask_vis)

                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

            # Basis merge
            if iteration in merge_iterations:
                gaussians.merge_brdf(opt.basis_merge_threshold)

            # Basis removal
            if iteration in delete_iterations:
                gaussians.delete_brdf()
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        #args.model_path = os.path.join("./output/", f"{args.source_path.split('/')[-1]}_{unique_str[0:10]}")
        args.model_path = os.path.join("./output/", f"{args.source_path.split('/')[-1]}")
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, dist_loss, normal_loss, mask_loss, loss_weight_sparse, loss_weight_img_sparse, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/dist_loss', dist_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/mask_loss', mask_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/weight_sparse_loss', loss_weight_sparse.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/weight_img_sparse_loss', loss_weight_img_sparse.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)



    # Report test and samples of training set
    if iteration in testing_iterations:
        mask_file = os.path.join("data/ball", "mask.png")
        ball_mask = torch.from_numpy(cv2.imread(os.path.join(mask_file), 0).astype(np.float32) / 255.)
        gt_normal_file = os.path.join("data/ball", "Normal_gt.mat")
        ball_normal = torch.from_numpy(read_mat_file(gt_normal_file))
        valid_idx = torch.where(ball_mask > 0.5)
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, vis=True)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)        
                    if tb_writer and config['name']=='test' and (idx==13):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        rend_alpha = render_pkg['rend_alpha']
                        gt_mask = viewpoint.gt_alpha_mask
                        mask_vis = (rend_alpha.detach() > 1e-5)
                        error_map = weight2rgb(torch.mean(torch.abs(image - gt_image), dim=0),mask_vis)
                        theta_map = torch.pow(torch.clamp(render_pkg["theta_map"],0.0,1.0),10)
                        theta_h = weight2rgb(theta_map.detach(), mask_vis)
                        depth_wrt = depth2rgb(depth, mask_vis)
                        weight_imgs = torch.clamp(render_pkg["weight_imgs"],0.0,1.0)
                        weights_wrt = []
                        for i in range(weight_imgs.shape[0]):
                            weights_wrt.append(weight2rgb(weight_imgs[i],mask_vis)) 
                        rend_normal = render_pkg['rend_normal']
                        rend_normal = rend_normal * 0.5 + 0.5
                        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                        brdf_map = torch.clamp(gamma_correction(render_pkg["brdf_imgs"],1.0),0.0,1.0)
                        basis_imgs = sphere_basis(scene.gaussians.get_base_color.detach().cpu(),scene.gaussians.get_roughness.detach().cpu(),\
                                            scene.gaussians.get_metallic.detach().cpu(),ball_normal, valid_idx)
                        result_wrt = torch.cat([gt_image, image, rend_normal, depth_wrt, theta_h], 2)
                        basis_wrt = torch.cat([gamma_correction(x[:,:,50:-50],2.2) for x in basis_imgs], 2)
                        weights_wrt2 = torch.cat([x for x in weights_wrt], 2)
                        brdf_wrt = torch.cat([x for x in brdf_map], 2)
                        img_path = os.path.join(scene.model_path, str(iteration))
                        os.makedirs(img_path, exist_ok=True)
                        save_image(result_wrt.cpu(), os.path.join(img_path,f'{viewpoint.image_name}_rendered.png'))
                        save_image(basis_wrt.cpu(), os.path.join(img_path,f'{viewpoint.image_name}_basis.png'))
                        save_image(weights_wrt2.cpu(), os.path.join(img_path,f'{viewpoint.image_name}_weight.png'))
                        save_image(brdf_wrt.cpu(), os.path.join(img_path,f'{viewpoint.image_name}_brdf_rendered.png'))
                        np.save(os.path.join(img_path,f'{viewpoint.image_name}_weight_imgs'), weight_imgs.cpu().numpy())
                        
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth_wrt[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/theta_h".format(viewpoint.image_name), theta_h[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/error_map".format(viewpoint.image_name),  error_map[None], global_step=iteration)

                        rend_dist = render_pkg["rend_dist"]
                        rend_dist = colormap(rend_dist.cpu().numpy()[0])
                        tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        #for i, brdf_img in enumerate(brdf_map):
                        #    tb_writer.add_images(config['name'] + "_view_{}/basis_brdf{}".format(viewpoint.image_name,i),  brdf_img[None], global_step=iteration)
                        #    tb_writer.add_images(config['name'] + "_view_{}/basis_sphere{}".format(viewpoint.image_name,i),  gamma_correction(basis_imgs[i][:,:,50:-50][None],2.2), global_step=iteration)
                        #    tb_writer.add_images(config['name'] + "_view_{}/basis_weight{}".format(viewpoint.image_name,i),  weights_wrt[i][None], global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
            
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        for i in range(scene.gaussians.get_base_color.shape[0]):
            tb_writer.add_histogram("scene/weight{}_histogram".format(i), scene.gaussians.get_weight[:,i], iteration)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6008)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(1,20001,1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(100,20001,1000)))
    parser.add_argument("--merge_iterations", nargs="+", type=int, default=list(range(6200,14201,500))) # 7000
    parser.add_argument("--delete_iterations", nargs="+", type=int, default=list(range(7400,14401,500))) # 7000
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[5000,10000,15000,20000,25000,30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.merge_iterations, args.delete_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")