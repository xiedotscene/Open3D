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

#include "PinholeCameraIntrinsic.h"

#include <Eigen/Dense>
#include <json/json.h>
#include <Core/Utility/Console.h>

namespace three{

const PinholeCameraIntrinsic PinholeCameraIntrinsic::GetPrimeSenseDefault()
{
	return PinholeCameraIntrinsic(640, 480, 525.0, 525.0, 319.5, 239.5);
}

PinholeCameraIntrinsic::PinholeCameraIntrinsic()
{
}

PinholeCameraIntrinsic::PinholeCameraIntrinsic(int width, int height,
		double fx, double fy, double cx, double cy)
{
	SetIntrinsics(width, height, fx, fy, cx, cy);
}

PinholeCameraIntrinsic::~PinholeCameraIntrinsic()
{
}

bool PinholeCameraIntrinsic::ConvertToJsonValue(Json::Value &value) const
{
	value["width"] = width_;
	value["height"] = height_;
	if (EigenMatrix3dToJsonArray(intrinsic_matrix_,
			value["intrinsic_matrix"]) == false) {
		return false;
	}
	return true;
}

bool PinholeCameraIntrinsic::ConvertFromJsonValue(const Json::Value &value)
{
	if (value.isObject() == false) {
		PrintWarning("PinholeCameraParameters read JSON failed: unsupported json format.\n");
		return false;
	}
	width_ = value.get("width", -1).asInt();
	height_ = value.get("height", -1).asInt();
	if (EigenMatrix3dFromJsonArray(intrinsic_matrix_,
			value["intrinsic_matrix"]) == false) {
		PrintWarning("PinholeCameraParameters read JSON failed: wrong format.\n");
		return false;
	}
	return true;
}

}	// namespace three
