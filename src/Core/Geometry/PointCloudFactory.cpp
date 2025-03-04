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

#include "PointCloud.h"

#include <Eigen/Dense>
#include <Core/Utility/Console.h>
#include <Core/Geometry/Image.h>
#include <Core/Geometry/RGBDImage.h>
#include <Core/Camera/PinholeCameraIntrinsic.h>

namespace three{

namespace {

std::shared_ptr<PointCloud> CreatePointCloudFromFloatDepthImage(
		const Image &depth, const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic, int stride)
{
	auto pointcloud = std::make_shared<PointCloud>();
	Eigen::Matrix4d camera_pose = extrinsic.inverse();
	auto focal_length = intrinsic.GetFocalLength();
	auto principal_point = intrinsic.GetPrincipalPoint();
	for (int i = 0; i < depth.height_; i += stride) {
		for (int j = 0; j < depth.width_; j += stride) {
			const float *p = PointerAt<float>(depth, j, i);
			if (*p > 0) {
				double z = (double)(*p);
				double x = (j - principal_point.first) * z /
						focal_length.first;
				double y = (i - principal_point.second) * z /
						focal_length.second;
				Eigen::Vector4d point = camera_pose *
						Eigen::Vector4d(x, y, z, 1.0);
				pointcloud->points_.push_back(point.block<3, 1>(0, 0));
			}
		}
	}
	return pointcloud;
}

template<typename TC, int NC>
std::shared_ptr<PointCloud> CreatePointCloudFromRGBDImageT(
		const RGBDImage &image, const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic)
{
	auto pointcloud = std::make_shared<PointCloud>();
	Eigen::Matrix4d camera_pose = extrinsic.inverse();
	auto focal_length = intrinsic.GetFocalLength();
	auto principal_point = intrinsic.GetPrincipalPoint();
	double scale = (sizeof(TC) == 1) ? 255.0 : 1.0;
	for (int i = 0; i < image.depth_.height_; i++) {
		float *p = (float *)(image.depth_.data_.data() +
				i * image.depth_.BytesPerLine());
		TC *pc = (TC *)(image.color_.data_.data() +
				i * image.color_.BytesPerLine());
		for (int j = 0; j < image.depth_.width_; j++, p++, pc += NC) {
			if (*p > 0) {
				double z = (double)(*p);
				double x = (j - principal_point.first) * z /
						focal_length.first;
				double y = (i - principal_point.second) * z /
						focal_length.second;
				Eigen::Vector4d point = camera_pose *
						Eigen::Vector4d(x, y, z, 1.0);
				pointcloud->points_.push_back(point.block<3, 1>(0, 0));
				pointcloud->colors_.push_back(Eigen::Vector3d(
						pc[0], pc[(NC - 1) / 2], pc[NC - 1]) / scale);
			}
		}
	}
	return pointcloud;
}

}	// unnamed namespace

std::shared_ptr<PointCloud> CreatePointCloudFromDepthImage(
		const Image &depth, const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic/* = Eigen::Matrix4d::Identity()*/,
		double depth_scale/* = 1000.0*/, double depth_trunc/* = 1000.0*/,
		int stride/* = 1*/)
{
	if (depth.num_of_channels_ == 1) {
		if (depth.bytes_per_channel_ == 2) {
			auto float_depth = ConvertDepthToFloatImage(depth, depth_scale,
					depth_trunc);
			return CreatePointCloudFromFloatDepthImage(*float_depth, intrinsic,
					extrinsic, stride);
		} else if (depth.bytes_per_channel_ == 4) {
			return CreatePointCloudFromFloatDepthImage(depth, intrinsic,
					extrinsic, stride);
		}
	}
	PrintDebug("[CreatePointCloudFromDepthImage] Unsupported image format.\n");
	return std::make_shared<PointCloud>();
}

std::shared_ptr<PointCloud> CreatePointCloudFromRGBDImage(
		const RGBDImage &image, const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic/* = Eigen::Matrix4d::Identity()*/)
{
	if (image.depth_.num_of_channels_ == 1 &&
			image.depth_.bytes_per_channel_ == 4) {
		if (image.color_.bytes_per_channel_ == 1 &&
				image.color_.num_of_channels_ == 3) {
			return CreatePointCloudFromRGBDImageT<uint8_t, 3>(
					image, intrinsic, extrinsic);
		} else if (image.color_.bytes_per_channel_ == 4 &&
				image.color_.num_of_channels_ == 1) {
			return CreatePointCloudFromRGBDImageT<float, 1>(
					image, intrinsic, extrinsic);
		}
	}
	PrintDebug("[CreatePointCloudFromRGBDImage] Unsupported image format.\n");
	return std::make_shared<PointCloud>();
}

}	// namespace three
