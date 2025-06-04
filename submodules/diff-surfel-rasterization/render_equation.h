#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor>
RenderEquationForwardCUDA(
	const torch::Tensor& base_color,
	const torch::Tensor& roughness,
	const torch::Tensor& metallic,
    const torch::Tensor& normals,
    const torch::Tensor& viewdirs,
	const torch::Tensor& incidents,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RenderEquationBackwardCUDA(
 	const torch::Tensor& base_color,
	const torch::Tensor& roughness,
	const torch::Tensor& metallic,
    const torch::Tensor& normals,
    const torch::Tensor& viewdirs,
	const torch::Tensor& incidents,
    const torch::Tensor& dL_drgb,
	const bool debug);