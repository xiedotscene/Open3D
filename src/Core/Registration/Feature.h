// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <Core/Geometry/KDTreeSearchParam.h>

namespace three {

class PointCloud;

class Feature
{
public:
	void Resize(int dim, int n) { data_.resize(dim, n); data_.setZero(); }
	size_t Dimension() const { return data_.rows(); }
	size_t Num() const { return data_.cols(); }

public:
	Eigen::MatrixXd data_;
};

/// Function to compute FPFH feature for a point cloud
std::shared_ptr<Feature> ComputeFPFHFeature(const PointCloud &input,
		const KDTreeSearchParam &search_param = KDTreeSearchParamKNN());


enum DepthDensifyMethod { depth_densify_nearest_neighbor,
		depth_densify_gaussian_kernel };
class PlanarParameterizationOption
{
public:
	PlanarParameterizationOption(double sigma = 0.1,
			int number_of_neighbors = 3, int half_patch_size_ = 5,
			DepthDensifyMethod depth_densify_method = depth_densify_gaussian_kernel):
			sigma_(sigma), number_of_neighbors_(number_of_neighbors),
			half_patch_size_(half_patch_size_),
			depth_densify_method_(depth_densify_method) {}

public:
	double sigma_;
	int number_of_neighbors_;
	int half_patch_size_; // patch_size = patch_half_size_ * 2 + 1
	DepthDensifyMethod depth_densify_method_;
};

class PlanarParameterizationOutput
{
public:
	Feature depth_;
	std::vector<Feature> weight_;
	std::vector<Eigen::MatrixXi> index_;
};

std::shared_ptr<PlanarParameterizationOutput> PlanarParameterization(
		const PointCloud &cloud,
		const KDTreeSearchParamHybrid &search_param,
		const PlanarParameterizationOption &option);

}	// namespace three
