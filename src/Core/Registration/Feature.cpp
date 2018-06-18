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

#include "Feature.h"

#include <Eigen/Dense>
#include <Core/Utility/Console.h>
#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/KDTreeFlann.h>

namespace three {

namespace {

Eigen::Vector4d ComputePairFeatures(const Eigen::Vector3d &p1,
		const Eigen::Vector3d &n1, const Eigen::Vector3d &p2,
		const Eigen::Vector3d &n2)
{
	Eigen::Vector4d result;
	Eigen::Vector3d dp2p1 = p2 - p1;
	result(3) = dp2p1.norm();
	if (result(3) == 0.0) {
		return Eigen::Vector4d::Zero();
	}
	auto n1_copy = n1;
	auto n2_copy = n2;
	double angle1 = n1_copy.dot(dp2p1) / result(3);
	double angle2 = n2_copy.dot(dp2p1) / result(3);
	if (acos(fabs(angle1)) > acos(fabs(angle2))) {
		n1_copy = n2;
		n2_copy = n1;
		dp2p1 *= -1.0;
		result(2) = -angle2;
	} else {
		result(2) = angle1;
	}
	auto v = dp2p1.cross(n1_copy);
	double v_norm = v.norm();
	if (v_norm == 0.0) {
		return Eigen::Vector4d::Zero();
	}
	v /= v_norm;
	auto w = n1_copy.cross(v);
	result(1) = v.dot(n2_copy);
	result(0) = atan2(w.dot(n2_copy), n1_copy.dot(n2_copy));
	return result;
}

std::shared_ptr<Feature> ComputeSPFHFeature(const PointCloud &input,
		const KDTreeFlann &kdtree, const KDTreeSearchParam &search_param)
{
	auto feature = std::make_shared<Feature>();
	feature->Resize(33, (int)input.points_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < (int)input.points_.size(); i++) {
		const auto &point = input.points_[i];
		const auto &normal = input.normals_[i];
		std::vector<int> indices;
		std::vector<double> distance2;
		if (kdtree.Search(point, search_param, indices, distance2) > 1) {
			// only compute SPFH feature when a point has neighbors
			double hist_incr = 100.0 / (double)(indices.size() - 1);
			for (size_t k = 1; k < indices.size(); k++) {
				// skip the point itself, compute histogram
				auto pf = ComputePairFeatures(point, normal,
						input.points_[indices[k]], input.normals_[indices[k]]);
				int h_index = (int)(floor(11 * (pf(0) + M_PI) / (2.0 * M_PI)));
				if (h_index < 0) h_index = 0;
				if (h_index >= 11) h_index = 10;
				feature->data_(h_index, i) += hist_incr;
				h_index = (int)(floor(11 * (pf(1) + 1.0) * 0.5));
				if (h_index < 0) h_index = 0;
				if (h_index >= 11) h_index = 10;
				feature->data_(h_index + 11, i) += hist_incr;
				h_index = (int)(floor(11 * (pf(2) + 1.0) * 0.5));
				if (h_index < 0) h_index = 0;
				if (h_index >= 11) h_index = 10;
				feature->data_(h_index + 22, i) += hist_incr;
			}
		}
	}
	return feature;
}

class ProjectedPoints
{
public:
	int id;		// global point ID
	double u;	// u coordinate on the projected patch
	double v;	// v coordinate on the projected patch
	double depth;
};

std::vector<ProjectedPoints> PlanarParameterizationForOnePoint(
		size_t point_id, const PointCloud &cloud, const KDTreeFlann &kdtree,
		const KDTreeSearchParamHybrid &search_param,
		const PlanarParameterizationOption &option)
{
	const int patch_half_size = option.half_patch_size_;
	const int patch_size = patch_half_size * 2 + 1;
	const double patch_half_diag_length = sqrt((2.0 * pow(patch_half_size, 2.0)));

	std::vector<int> indices;
	std::vector<double> distance2;
	Eigen::Matrix3d tangential_axis;
	Eigen::Vector3d vt, vt_n, vt_t1, vt_t2, vt_adj, vt_proj;
	Eigen::Vector2d vt_val = Eigen::Vector2d::Zero();
	Eigen::Vector2i vt_val_descrete = Eigen::Vector2i::Zero();
	Eigen::MatrixXd patch = Eigen::MatrixXd::Zero(patch_size, patch_size);
	double depth = -1.0;
	int adj_id = -1;
	vt = cloud.points_[point_id];

	// giving enough search radius to give enough space in the boundry pixels
	KDTreeSearchParamHybrid search_param_buffer = search_param;
	search_param_buffer.radius_ = search_param.radius_ * 2.0;
	int number_of_searched_neighbors =
			kdtree.Search(vt, search_param_buffer, indices, distance2);
	std::vector<ProjectedPoints> output;
	output.resize(number_of_searched_neighbors);

	if (number_of_searched_neighbors >= 1) {
		tangential_axis = ComputeTangentialAxis(cloud, indices);
		vt_n = tangential_axis.col(0);
		vt_t1 = tangential_axis.col(1);
		vt_t2 = tangential_axis.col(2);
		// project points onto tangential PlanarParameterization
		for (int j = 0; j < number_of_searched_neighbors; j++){
			adj_id = indices[j];
			vt_adj = cloud.points_[adj_id];
			depth = (vt_adj - vt).dot(vt_n);
			vt_proj = vt_adj - depth * vt_n;
			vt_val(0) = (vt_proj - vt).dot(vt_t1);
			vt_val(1) = (vt_proj - vt).dot(vt_t2);
			vt_val = vt_val / search_param.radius_ * patch_half_diag_length;
			output[j].u = vt_val(0);
			output[j].v = vt_val(1);
			output[j].depth = depth;
			output[j].id = adj_id;
		}
	}
	return std::move(output);
}

class ProcessedPatch
{
public:
	// kernel x kernel size, local depth map
	Eigen::MatrixXd depth_map;
	// kernel x kernel size, point cloud index
	std::vector<Eigen::MatrixXi> index_map;
	// kernel x kernel size, point cloud weight to adjacent index
	std::vector<Eigen::MatrixXd> weight_map;
};

std::shared_ptr<ProcessedPatch> PlanarParameterizationForOnePatch(
		const std::vector<ProjectedPoints> &projected_points,
		const PlanarParameterizationOption &option)
{
	const int number_of_neighbors = option.number_of_neighbors_;
	const int patch_half_size = option.half_patch_size_;
	const int patch_size = patch_half_size * 2 + 1;
	const double sigma_pow2 = option.sigma_ * option.sigma_;

	auto output = std::make_shared<ProcessedPatch>();
	output->index_map.resize(number_of_neighbors);
	output->weight_map.resize(number_of_neighbors);
	for (int i = 0; i < number_of_neighbors; i++){
		output->index_map[i].resize(patch_size, patch_size);
		output->weight_map[i].resize(patch_size, patch_size);
		output->index_map[i].setConstant(-1);
		output->weight_map[i].setZero();
	}
	output->depth_map.resize(patch_size, patch_size);

	const int number_of_points = int(projected_points.size());
	Eigen::MatrixXd data = Eigen::MatrixXd::Zero(2, number_of_points);
	int cnt = 0;
	for (int i = 0; i < (int)projected_points.size(); i++){
		data(0,i) = projected_points[i].u;
		data(1,i) = projected_points[i].v;
	}
	KDTreeFlann kdtree;
	kdtree.SetMatrixData(data);
	std::vector<int> indices;
	std::vector<double> distance2;
	for (int u = 0; u < (int)patch_size; u++){
		for (int v = 0; v < (int)patch_size; v++){
			Eigen::Vector2d query;
			query(0) = u - patch_half_size;
			query(1) = v - patch_half_size;
			// todo: should we use SearchKNN here? maybe fill in blank
			int number_of_searched_neighbors = kdtree.SearchKNN(
					query, number_of_neighbors, indices, distance2);
			double sum_weight = 0.0, sum_weighted_val = 0.0;
			for (int i = 0; i < number_of_searched_neighbors; i++) {
				int i_adj = indices[i];
				double dist_adj = distance2[i];
				double weight_i_th_adj = exp(-dist_adj/sigma_pow2);
				double depth_adj = projected_points[i_adj].depth;
				sum_weight += weight_i_th_adj;
				sum_weighted_val += weight_i_th_adj * depth_adj;
				output->index_map[i](v, u) = projected_points[i_adj].id;
				output->weight_map[i](v, u) = weight_i_th_adj;
			}
			// normalize weight_map
			if (sum_weight != 0.0){
				for (int i = 0; i < number_of_searched_neighbors; i++) {
					output->weight_map[i](v, u) /= sum_weight;
				}
			}
			if (number_of_searched_neighbors >= 1){
				if (option.depth_densify_method_ ==
					depth_densify_gaussian_kernel && sum_weight != 0.0)
					output->depth_map(v, u) = sum_weighted_val / sum_weight;
				else if (option.depth_densify_method_ ==
						depth_densify_nearest_neighbor)
					output->depth_map(v, u) =
					projected_points[indices[0]].depth;
			}
		}
	}
	return output;
}

}	// unnamed namespace

std::shared_ptr<Feature> ComputeFPFHFeature(const PointCloud &input,
		const KDTreeSearchParam &search_param/* = KDTreeSearchParamKNN()*/)
{
	auto feature = std::make_shared<Feature>();
	feature->Resize(33, (int)input.points_.size());
	if (input.HasNormals() == false) {
		PrintDebug("[ComputeFPFHFeature] Failed because input point cloud has no normal.\n");
		return feature;
	}
	KDTreeFlann kdtree(input);
	auto spfh = ComputeSPFHFeature(input, kdtree, search_param);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < (int)input.points_.size(); i++) {
		const auto &point = input.points_[i];
		std::vector<int> indices;
		std::vector<double> distance2;
		if (kdtree.Search(point, search_param, indices, distance2) > 1) {
			double sum[3] = {0.0, 0.0, 0.0};
			for (size_t k = 1; k < indices.size(); k++) {
				// skip the point itself
				double dist = distance2[k];
				if (dist == 0.0)
					continue;
				for (int j = 0; j < 33; j++) {
					double val = spfh->data_(j, indices[k]) / dist;
					sum[j / 11] += val;
					feature->data_(j, i) += val;
				}
			}
			for (int j = 0; j < 3; j++)
				if (sum[j] != 0.0) sum[j] = 100.0 / sum[j];
			for (int j = 0; j < 33; j++) {
				feature->data_(j, i) *= sum[j / 11];
				// The commented line is the fpfh function in the paper.
				// But according to PCL implementation, it is skipped.
				// Our initial test shows that the full fpfh function in the
				// paper seems to be better than PCL implementation. Further
				// test required.
				feature->data_(j, i) += spfh->data_(j, i);
			}
		}
	}
	return feature;
}

std::shared_ptr<PlanarParameterizationOutput> PlanarParameterization(
		const PointCloud &cloud,
		const KDTreeSearchParamHybrid &search_param,
		const PlanarParameterizationOption &option)
{
	// parameters
	const int patch_size = option.half_patch_size_ * 2 + 1;
	const int patch_size_pow2 = patch_size * patch_size;
	const int number_of_points = (int)cloud.points_.size();

	auto output = std::make_shared<PlanarParameterizationOutput>();
	output->depth_.Resize(patch_size_pow2, number_of_points);
	for (int i = 0; i < option.number_of_neighbors_; i++){
		Feature data_weight;
		data_weight.Resize(patch_size_pow2, number_of_points);
		output->weight_.push_back(data_weight);
		Eigen::MatrixXi data_index;
		data_index.resize(patch_size_pow2, number_of_points);
		output->index_.push_back(data_index);
	}
	KDTreeFlann kdtree;
	kdtree.SetGeometry(cloud);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < number_of_points; i++) {
		auto projected_points = PlanarParameterizationForOnePoint(
				i, cloud, kdtree, search_param, option);
		auto information_patch = PlanarParameterizationForOnePatch(
				projected_points, option);
		for (int j = 0; j < option.number_of_neighbors_; j++){
			for (int k = 0; k < patch_size_pow2; k++){
				output->weight_[j].data_(k, i) =
						information_patch->weight_map[j](k);
				output->index_[j](k, i) =
						information_patch->index_map[j](k);
			}
		}
		for (size_t k = 0; k < patch_size_pow2; k++){
			output->depth_.data_(k, i) = information_patch->depth_map(k);
		}
	}
	return output;
}

}	// namespace three
