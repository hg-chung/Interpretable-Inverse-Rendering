#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include "cuda_rasterizer/auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void render_equation_backward_kernel(
    const int P,
    const glm::vec3* base_color,
    const float* roughness,
    const float* metallic,
    const glm::vec3* normals,
    const glm::vec3* viewdirs,
    const glm::vec3* incidents,
    const glm::vec3* dL_dpbrs,
    glm::vec3* dL_dbase_color,
    float* dL_droughness,
    float* dL_dmetallic,
    glm::vec3* dL_dnormals,
    glm::vec3* dL_dviewdirs,
    glm::vec3* dL_dincidents)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	const glm::vec3 normal = normals[idx];
    const glm::vec3 viewdir = viewdirs[idx];
    const glm::vec3 incident_dir = incidents[idx];
    const glm::vec3 base = base_color[idx];
    const glm::vec3 dL_dpbr = dL_dpbrs[idx];
    const float metal = metallic[idx];
    const float rough = roughness[idx];
    glm::vec3 half_dir = incident_dir + viewdir;
    float half_norm = fmaxf(glm::length(half_dir), 0.0000001f);
    glm::vec3 half_dir_normalize = half_dir / half_norm;
    float h_d_n = fmaxf(glm::dot(half_dir_normalize, normal), 0.0f);
    float h_d_o = fmaxf(glm::dot(half_dir_normalize, viewdir), 0.0f);
    float n_d_i = fmaxf(glm::dot(normal, incident_dir), 0.0f);
    float n_d_o = fmaxf(glm::dot(normal, viewdir), 0.0f);

    glm::vec3 f_d = (1 - metal) * base / 3.14159f;

    // D
    float r2 = fmaxf(rough * rough, 0.0000001f);
    float amp = 1.0f / (r2 * 3.14159f);
    float sharp = 2.0f / r2;
    float expf_amp = expf(sharp * (h_d_n - 1.0f));
    float D = amp * expf_amp;

    // F
    glm::vec3 F_0 = 0.04f * (1.0f - metal) + base * metal;
    glm::vec3 F = F_0 + (1.0f - F_0) * powf(1.0f - h_d_o, 5.0f);

    // V
    float r2v = powf(1.0f + rough, 2.0f) / 8.0f;
    float denom1 = fmaxf(n_d_i * (1 - r2v) + r2v, 0.0000001f);
    float denom2 = fmaxf(n_d_o * (1 - r2v) + r2v, 0.0000001f);
    float v_schlick_ggx1 = 0.5f / denom1;
    float v_schlick_ggx2 = 0.5f / denom2;
    float V = v_schlick_ggx1 * v_schlick_ggx2;

    glm::vec3 f_s = D*F*V;
    //pbr += (f_d+f_s) * incident_light * (2.0f * 3.14159f * n_d_i / (float)sample_num);

    glm::vec3 dL_dfd = dL_dpbr  * (2.0f * 3.14159f * n_d_i );
    glm::vec3 dL_dfs = dL_dpbr  * (2.0f * 3.14159f * n_d_i );
    float dL_dn_d_i = glm::dot(dL_dpbr, (f_d+f_s) ) * (2.0f * 3.14159f );

    // from dL_dfd
    glm::vec3 dL_dbase = dL_dfd * (1 - metal) / 3.14159f;
    float dL_dmetal = -glm::dot(dL_dfd, base) / 3.14159f;

    // from dL_dfs
    float dL_dD = glm::dot(dL_dfs*V, F);
    glm::vec3 dL_dF = dL_dfs*D*V;
    float dL_dV = glm::dot(dL_dfs*D,F);
    // from dL_dD
    float dL_damp = dL_dD * expf_amp;
    float dL_dexpf_amp = dL_dD * amp;
    float dL_dsharp = (h_d_n - 1.0f) * expf_amp * dL_dexpf_amp;
    float dL_dh_d_n = sharp * expf_amp * dL_dexpf_amp;
    float dL_dr2 = -2.0f/(r2*r2) * dL_dsharp - 1.0f / (r2 * r2 * 3.14159f) * dL_damp;
    float dL_drough = dL_dr2 * 2.0f * rough;
    // from dL_dF
    glm::vec3 dL_dF0 = (1.0f - powf(1.0f - h_d_o, 5.0f)) * dL_dF;
    float dL_dh_d_o = glm::dot((1.0f - F_0), dL_dF) * -5.0f * powf(1.0f - h_d_o, 4.0f);
    dL_dbase += metal * dL_dF0;
    dL_dmetal += glm::dot(base-0.04f, dL_dF0);
    // from dL_dV
    float dL_dv_schlick_ggx1 = dL_dV * v_schlick_ggx2;
    float dL_dv_schlick_ggx2 = dL_dV * v_schlick_ggx1;
    float dL_ddenom1 = -0.5f / (denom1 * denom1) * dL_dv_schlick_ggx1;
    float dL_ddenom2 = -0.5f / (denom2 * denom2) * dL_dv_schlick_ggx2;
    dL_dn_d_i += dL_ddenom1 * (1 - r2v);
    float dL_dn_d_o = dL_ddenom2 * (1 - r2v);
    float dL_dr2v = (1.0f - n_d_i) * dL_ddenom1 + (1.0f - n_d_o) * dL_ddenom2;
    dL_drough += (1.0f + rough) / 4.0f * dL_dr2v;

    glm::vec3 dL_dhalf_dir_normalize={0.0f,0.0f,0.0f};
    glm::vec3 dL_dnormal={0.0f,0.0f,0.0f};
    glm::vec3 dL_dviewdir={0.0f,0.0f,0.0f};
    glm::vec3 dL_dincident={0.0f,0.0f,0.0f};
    if (h_d_n>0.0f){
        dL_dhalf_dir_normalize += normal * dL_dh_d_n;
        dL_dnormal += half_dir_normalize * dL_dh_d_n;
    }
    if (h_d_o>0.0f){
        dL_dhalf_dir_normalize += viewdir * dL_dh_d_o;
        dL_dviewdir += half_dir_normalize * dL_dh_d_o;
    }
    if (n_d_i>0.0f){
        dL_dnormal += incident_dir* dL_dn_d_i;
        dL_dincident += normal * dL_dn_d_i;
    }

    if (n_d_o>0.0f){
        dL_dnormal += viewdir * dL_dn_d_o;
        dL_dviewdir += normal * dL_dn_d_o;
    }

    // half_dir TODO:consider norm
    dL_dviewdir += dL_dhalf_dir_normalize / half_norm;
    dL_dincident += dL_dhalf_dir_normalize / half_norm;

    dL_dviewdirs[idx]+=dL_dviewdir;
    dL_dincidents[idx]+=dL_dincident;
    dL_dnormals[idx]+=dL_dnormal;
    dL_dbase_color[idx]+=dL_dbase;
    dL_dmetallic[idx]+=dL_dmetal;
    dL_droughness[idx]+=dL_drough;
}


void render_equation_backward_cuda(
    const int P,
    const glm::vec3* base_color,
    const float* roughness,
    const float* metallic,
    const glm::vec3* normals,
    const glm::vec3* viewdirs,
    const glm::vec3* incidents,
    const glm::vec3* dL_dpbrs,
    glm::vec3* dL_dbase_color,
    float* dL_droughness,
    float* dL_dmetallic,
    glm::vec3* dL_dnormals,
    glm::vec3* dL_dviewdirs,
    glm::vec3* dL_dincidents
){
    render_equation_backward_kernel << <(P + 255) / 256, 256 >> > (
    P, base_color, roughness, metallic, normals, viewdirs, incidents, 
    dL_dpbrs, dL_dbase_color, dL_droughness, dL_dmetallic, dL_dnormals, dL_dviewdirs, dL_dincidents);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RenderEquationBackwardCUDA(
 	const torch::Tensor& base_color,
	const torch::Tensor& roughness,
	const torch::Tensor& metallic,
    const torch::Tensor& normals,
    const torch::Tensor& viewdirs,
	const torch::Tensor& incidents,
    const torch::Tensor& dL_dpbrs,
	const bool debug){
    const int P = base_color.size(0);
    auto float_opts = base_color.options().dtype(torch::kFloat32);
	torch::Tensor dL_dbase_color = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_droughness = torch::zeros({P, 1}, float_opts);
    torch::Tensor dL_dmetallic = torch::zeros({P, 1}, float_opts);
    torch::Tensor dL_dnormals = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dviewdirs = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dincidents = torch::zeros({P, 3}, float_opts);
    render_equation_backward_cuda(
            P,
            (glm::vec3*)base_color.contiguous().data_ptr<float>(),
            roughness.contiguous().data_ptr<float>(),
            metallic.contiguous().data_ptr<float>(),
            (glm::vec3*)normals.contiguous().data_ptr<float>(),
            (glm::vec3*)viewdirs.contiguous().data_ptr<float>(),
            (glm::vec3*)incidents.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_dpbrs.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_dbase_color.contiguous().data_ptr<float>(),
            dL_droughness.contiguous().data_ptr<float>(),
            dL_dmetallic.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_dnormals.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_dviewdirs.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_dincidents.contiguous().data_ptr<float>()
        );
    return std::make_tuple(dL_dbase_color, dL_droughness, dL_dmetallic, dL_dnormals,dL_dviewdirs, dL_dincidents);
}




__global__ void render_equation_forward_kernel(
    const int P,
    const glm::vec3* base_color,
    const float* roughness,
    const float* metallic,
    const glm::vec3* normals,
    const glm::vec3* viewdirs,
    const glm::vec3* incidents,
    glm::vec3* out_pbr)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
    glm::vec3 pbr = {0.0f, 0.0f, 0.0f};
    const glm::vec3 normal = normals[idx];
    const glm::vec3 viewdir = viewdirs[idx];
    const glm::vec3 incident_dir = incidents[idx];
    const glm::vec3 base = base_color[idx];
    const float metal = metallic[idx];
    const float rough = roughness[idx];
    
    glm::vec3 half_dir = incident_dir + viewdir;
    half_dir = half_dir / fmaxf(glm::length(half_dir), 0.0000001f);
    float h_d_n = fmaxf(glm::dot(half_dir, normal), 0.0f);
    float h_d_o = fmaxf(glm::dot(half_dir, viewdir), 0.0f);
    float n_d_i = fmaxf(glm::dot(normal, incident_dir), 0.0f);
    float n_d_o = fmaxf(glm::dot(normal, viewdir), 0.0f);

    glm::vec3 f_d = (1 - metal) * base / 3.14159f;

    // D
    float r2 = fmaxf(rough * rough, 0.0000001f);
    float amp = 1.0f / (r2 * 3.14159f);
    float sharp = 2.0f / r2;
    float D = amp * expf(sharp * (h_d_n - 1.0f));

    // F
    glm::vec3 F_0 = 0.04f * (1.0f - metal) + base * metal;
    glm::vec3 F = F_0 + (1.0f - F_0) * powf(1.0f - h_d_o, 5.0f);

    // V
    r2 = __powf(1.0f + rough, 2.0f) / 8.0f;
    float V = (0.5f / fmaxf(n_d_i * (1 - r2) + r2, 0.0000001f)) * (0.5f / fmaxf(n_d_o * (1 - r2) + r2, 0.0000001f));

    glm::vec3 f_s = D*F*V;
    //glm::vec3 transport = (2.0f * 3.14159f * n_d_i);
    pbr += (f_d+f_s) * (2.0f * 3.14159f * n_d_i);
	
	out_pbr[idx]=pbr;
}

void render_equation_forward_cuda(
    const int P,
    const glm::vec3* base_color,
    const float* roughness,
    const float* metallic,
    const glm::vec3* normals,
    const glm::vec3* viewdirs,
    const glm::vec3* incidents,
    glm::vec3* out_pbr
){
    render_equation_forward_kernel << <(P + 255) / 256, 256 >> > (
    P, base_color, roughness, metallic, normals, viewdirs, incidents, out_pbr);
}


std::tuple<torch::Tensor>
RenderEquationForwardCUDA(
	const torch::Tensor& base_color,
	const torch::Tensor& roughness,
	const torch::Tensor& metallic,
    const torch::Tensor& normals,
    const torch::Tensor& viewdirs,
	const torch::Tensor& incidents,
	const bool debug){
	const int P = base_color.size(0);
    auto float_opts = base_color.options().dtype(torch::kFloat32);
    torch::Tensor pbr = torch::full({P, 3}, 0.0, float_opts);
    render_equation_forward_cuda(
        P,
        (glm::vec3*)base_color.contiguous().data_ptr<float>(),
        roughness.contiguous().data_ptr<float>(),
        metallic.contiguous().data_ptr<float>(),
        (glm::vec3*)normals.contiguous().data_ptr<float>(),
        (glm::vec3*)viewdirs.contiguous().data_ptr<float>(),
        (glm::vec3*)incidents.contiguous().data_ptr<float>(),
        (glm::vec3*)pbr.contiguous().data_ptr<float>()
    );
    return std::make_tuple(pbr);
}