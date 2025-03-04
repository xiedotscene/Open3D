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

#include "GlobalOptimization.h"

#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <Core/Utility/Console.h>
#include <Core/Utility/Timer.h>
#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimizationMethod.h>
#include <Core/Registration/GlobalOptimizationConvergenceCriteria.h>

namespace three{

namespace {

/// Definition of linear operators used for computing Jacobian matrix.
/// If the relative transform of the two geometry is reasonably small,
/// they can be approximated as below linearized form
/// SE(3) \approx = |     1 -gamma   beta     a |
///                 | gamma      1 -alpha     b |
///                 | -beta  alpha      1     c |
///                 |     0      0      0     1 |
/// It is from sin(x) \approx x and cos(x) \approx 1 when x is almost zero.
/// See [Choi et al 2015] for more detail. Reference list in GlobalOptimization.h
const std::vector<Eigen::Matrix4d> jacobian_operator = {
	(Eigen::Matrix4d() << /* for alpha */
	0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0).finished(),
	(Eigen::Matrix4d() << /* for beta */
	0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0).finished(),
	(Eigen::Matrix4d() << /* for gamma */
	0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished(),
	(Eigen::Matrix4d() << /* for a */
	0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished(),
	(Eigen::Matrix4d() << /* for b */
	0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0).finished(),
	(Eigen::Matrix4d() << /* for c */
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0).finished() };

/// This function is intended for linearized form of SE(3).
/// It is an approximate form. See [Choi et al 2015] for derivation.
/// Alternatively, explicit representation that uses quaternion can be used
/// here to replace this function. Refer to linearizeOplus() in
/// https://github.com/RainerKuemmerle/g2o/blob/master/g2o/types/slam3d/edge_se3.cpp
inline Eigen::Vector6d GetLinearized6DVector(const Eigen::Matrix4d input)
{
	Eigen::Vector6d output;
	output(0) = (-input(1, 2) + input(2, 1)) / 2.0;
	output(1) = (-input(2, 0) + input(0, 2)) / 2.0;
	output(2) = (-input(0, 1) + input(1, 0)) / 2.0;
	output.block<3, 1>(3, 0) = input.block<3, 1>(0, 3);
	return std::move(output);
}

inline Eigen::Vector6d GetMisalignmentVector(const Eigen::Matrix4d &X_inv,
		const Eigen::Matrix4d &Ts, const Eigen::Matrix4d &Tt_inv)
{
	Eigen::Matrix4d temp;
	temp.noalias() = X_inv * Tt_inv * Ts;
	return GetLinearized6DVector(temp);
}

inline std::tuple<Eigen::Matrix4d, Eigen::Matrix4d, Eigen::Matrix4d>
		GetRelativePoses(const PoseGraph &pose_graph, int edge_id)
{
	const PoseGraphEdge &te = pose_graph.edges_[edge_id];
	const PoseGraphNode &ts = pose_graph.nodes_[te.source_node_id_];
	const PoseGraphNode &tt = pose_graph.nodes_[te.target_node_id_];
	Eigen::Matrix4d X_inv = te.transformation_.inverse();
	Eigen::Matrix4d Ts = ts.pose_;
	Eigen::Matrix4d Tt_inv = tt.pose_.inverse();
	return std::make_tuple(std::move(X_inv), std::move(Ts), std::move(Tt_inv));
}

std::tuple<Eigen::Matrix6d, Eigen::Matrix6d> GetJacobian(
		const Eigen::Matrix4d &X_inv, const Eigen::Matrix4d &Ts,
		const Eigen::Matrix4d &Tt_inv)
{
	Eigen::Matrix6d Js = Eigen::Matrix6d::Zero();
	for (int i = 0; i < 6; i++) {
		Eigen::Matrix4d temp = X_inv * Tt_inv *
				jacobian_operator[i] * Ts;
		Js.block<6, 1>(0, i) = GetLinearized6DVector(temp);
	}
	Eigen::Matrix6d Jt = Eigen::Matrix6d::Zero();
	for (int i = 0; i < 6; i++) {
		Eigen::Matrix4d temp = X_inv * Tt_inv *
				-jacobian_operator[i] * Ts;
		Jt.block<6, 1>(0, i) = GetLinearized6DVector(temp);
	}
	return std::make_tuple(std::move(Js), std::move(Jt));
}

/// Function to update line_process value defined in [Choi et al 2015]
/// See Eq (2). temp2 value in this function is derived from dE/dl = 0
int UpdateConfidence(
		PoseGraph &pose_graph, const Eigen::VectorXd &zeta,
		const double line_process_weight,
		const GlobalOptimizationOption &option)
{
	int n_edges = (int)pose_graph.edges_.size();
	int valid_edges_num = 0;
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		PoseGraphEdge &t = pose_graph.edges_[iter_edge];
		Eigen::Vector6d e = zeta.block<6, 1>(iter_edge * 6, 0);
		double residual_square = e.transpose() * t.information_ * e;
		double temp = line_process_weight /
				(line_process_weight + residual_square);
		double temp2 = temp * temp;
		t.confidence_ = temp2;
		if (temp2 > option.edge_prune_threshold_)
			valid_edges_num++;
	}
	return valid_edges_num;
}

/// Function to compute residual defined in [Choi et al 2015] See Eq (9).
double ComputeResidual(const PoseGraph &pose_graph, const Eigen::VectorXd &zeta,
		const double line_process_weight,
		const GlobalOptimizationOption &option)
{
	int n_edges = (int)pose_graph.edges_.size();
	double residual = 0.0;
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &te = pose_graph.edges_[iter_edge];
		double line_process_iter = 1.0;
		if (te.uncertain_)
			line_process_iter = te.confidence_;
		Eigen::Vector6d e = zeta.block<6, 1>(iter_edge * 6, 0);
		residual += line_process_iter * e.transpose() * te.information_ * e +
				line_process_weight * pow(sqrt(line_process_iter) - 1, 2.0);
	}
	return residual;
}

/// Function to compute residual defined in [Choi et al 2015] See Eq (6).
Eigen::VectorXd ComputeZeta(const PoseGraph &pose_graph)
{
	int n_edges = (int)pose_graph.edges_.size();
	Eigen::VectorXd output(n_edges * 6);
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		Eigen::Matrix4d X_inv, Ts, Tt_inv;
		std::tie(X_inv, Ts, Tt_inv) = GetRelativePoses(pose_graph, iter_edge);
		Eigen::Vector6d e = GetMisalignmentVector(X_inv, Ts, Tt_inv);
		output.block<6, 1>(iter_edge * 6, 0) = e;
	}
	return std::move(output);
}

/// The information matrix used here is consistent with [Choi et al 2015].
/// It is [p_x | I]^T[p_x | I]. \zeta is [\alpha \beta \gamma a b c]
/// Another definition of information matrix used for [Kümmerle et al 2011] is
/// [I | p_x] ^ T[I | p_x]  so \zeta is [a b c \alpha \beta \gamma].
///
/// To see how H can be derived see [Kümmerle et al 2011].
/// Eq (9) for definition of H and b for k-th constraint.
/// To see how the covariance matrix forms H, check g2o technical note:
/// https ://github.com/RainerKuemmerle/g2o/blob/master/doc/g2o.pdf
/// Eq (20) and Eq (21). (There is a typo in the equation though. B should be J)
///
/// This function focuses the case that every edge has two nodes (not hyper graph)
/// so we have two Jacobian matrices from one constraint.
std::tuple<Eigen::MatrixXd, Eigen::VectorXd> ComputeLinearSystem(
		const PoseGraph &pose_graph, const Eigen::VectorXd &zeta)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();
	Eigen::MatrixXd H(n_nodes * 6, n_nodes * 6);
	Eigen::VectorXd b(n_nodes * 6);
	H.setZero();
	b.setZero();

	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
		Eigen::Vector6d e = zeta.block<6, 1>(iter_edge * 6, 0);

		Eigen::Matrix4d X_inv, Ts, Tt_inv;
		std::tie(X_inv, Ts, Tt_inv) = GetRelativePoses(pose_graph, iter_edge);

		Eigen::Matrix6d Js, Jt;
		std::tie(Js, Jt) = GetJacobian(X_inv, Ts, Tt_inv);
		Eigen::Matrix6d JsT_Info =
				Js.transpose() * t.information_;
		Eigen::Matrix6d JtT_Info =
				Jt.transpose() * t.information_;
		Eigen::Vector6d eT_Info = e.transpose() * t.information_;

		double line_process_iter = 1.0;
		if (t.uncertain_)
			line_process_iter = t.confidence_;

		int id_i = t.source_node_id_ * 6;
		int id_j = t.target_node_id_ * 6;
		H.block<6, 6>(id_i, id_i).noalias() +=
				line_process_iter * JsT_Info * Js;
		H.block<6, 6>(id_i, id_j).noalias() +=
				line_process_iter * JsT_Info * Jt;
		H.block<6, 6>(id_j, id_i).noalias() +=
				line_process_iter * JtT_Info * Js;
		H.block<6, 6>(id_j, id_j).noalias() +=
				line_process_iter * JtT_Info * Jt;
		b.block<6, 1>(id_i, 0).noalias() -=
				line_process_iter * eT_Info.transpose() * Js;
		b.block<6, 1>(id_j, 0).noalias() -=
				line_process_iter * eT_Info.transpose() * Jt;
	}
	return std::make_tuple(std::move(H), std::move(b));
}

Eigen::VectorXd UpdatePoseVector(const PoseGraph &pose_graph)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	Eigen::VectorXd output(n_nodes * 6);
	for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
		Eigen::Vector6d output_iter = TransformMatrix4dToVector6d(
				pose_graph.nodes_[iter_node].pose_);
		output.block<6, 1>(iter_node * 6, 0) = output_iter;
	}
	return std::move(output);
}

std::shared_ptr<PoseGraph> UpdatePoseGraph(const PoseGraph &pose_graph,
		const Eigen::VectorXd delta)
{
	std::shared_ptr<PoseGraph> pose_graph_updated =
		std::make_shared<PoseGraph>();
	*pose_graph_updated = pose_graph;
	int n_nodes = (int)pose_graph.nodes_.size();
	for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
		Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
		pose_graph_updated->nodes_[iter_node].pose_ =
				TransformVector6dToMatrix4d(delta_iter) *
				pose_graph_updated->nodes_[iter_node].pose_;
	}
	return pose_graph_updated;
}

bool CheckRightTerm(const Eigen::VectorXd &right_term,
		const GlobalOptimizationConvergenceCriteria &criteria)
{
	if (right_term.maxCoeff() < criteria.min_right_term_) {
		PrintDebug("Maximum coefficient of right term < %e\n",
				criteria.min_right_term_);
		return true;
	}
	return false;
}

bool CheckRelativeIncrement(
		const Eigen::VectorXd &delta, const Eigen::VectorXd &x,
		const GlobalOptimizationConvergenceCriteria &criteria)
{
	if (delta.norm() < criteria.min_relative_increment_ *
			(x.norm() + criteria.min_relative_increment_)) {
		PrintDebug("Delta.norm() < %e * (x.norm() + %e)\n",
				criteria.min_relative_increment_,
				criteria.min_relative_increment_);
		return true;
	}
	return false;
}

bool CheckRelativeResidualIncrement(
		double current_residual, double new_residual,
		const GlobalOptimizationConvergenceCriteria &criteria)
{
	if (current_residual - new_residual <
		criteria.min_relative_residual_increment_ * current_residual) {
		PrintDebug("Current_residual - new_residual < %e * current_residual\n",
				criteria.min_relative_residual_increment_);
		return true;
	}
	return false;
}

bool CheckResidual(double residual,
		const GlobalOptimizationConvergenceCriteria &criteria)
{
	if (residual < criteria.min_residual_) {
		PrintDebug("Current_residual < %e\n", criteria.min_residual_);
		return true;
	}
	return false;
}

bool CheckMaxIteration(int iteration,
		const GlobalOptimizationConvergenceCriteria &criteria)
{
	if (iteration >= criteria.max_iteration_) {
		PrintDebug("Reached maximum number of iterations (%d)\n",
				criteria.max_iteration_);
		return true;
	}
	return false;
}

bool CheckMaxIterationLM(int iteration,
		const GlobalOptimizationConvergenceCriteria &criteria)
{
	if (iteration >= criteria.max_iteration_lm_) {
		PrintDebug("Reached maximum number of iterations (%d)\n",
				criteria.max_iteration_lm_);
		return true;
	}
	return false;
}

double ComputeLineProcessWeight(const PoseGraph &pose_graph,
		const GlobalOptimizationOption &option)
{
	int n_edges = (int)pose_graph.edges_.size();
	double average_number_of_correspondences = 0.0;
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		double number_of_correspondences =
				pose_graph.edges_[iter_edge].information_(0,0);
		average_number_of_correspondences += number_of_correspondences;
	}
	if (n_edges > 0) {
		// see Section 5 in [Choi et al 2015]
		average_number_of_correspondences /= (double)n_edges;
		double line_process_weight = 2.0 *
				pow(option.max_correspondence_distance_, 2) *
				average_number_of_correspondences;
		return line_process_weight;
	}
	else {
		return 0.0;
	}
}

void CompensateReferencePoseGraphNode(PoseGraph &pose_graph_new,
		const PoseGraph &pose_graph_orig, int reference_node)
{
	PrintDebug("CompensateReferencePoseGraphNode : reference : %d\n",
			reference_node);
	int n_nodes = (int)pose_graph_new.nodes_.size();
	if (reference_node < 0 || reference_node >= n_nodes) {
		return;
	} else {
		Eigen::Matrix4d compensation =
				pose_graph_orig.nodes_[reference_node].pose_ *
				pose_graph_new.nodes_[reference_node].pose_.inverse();
		for (int i = 0; i < n_nodes; i++)
		{
			pose_graph_new.nodes_[i].pose_ = compensation *
					pose_graph_new.nodes_[i].pose_;
		}
	}
}

}	// unnamed namespace

std::shared_ptr<PoseGraph> CreatePoseGraphWithoutInvalidEdges(
		const PoseGraph &pose_graph,
		const GlobalOptimizationOption &option)
{
	std::shared_ptr<PoseGraph> pose_graph_pruned =
			std::make_shared<PoseGraph>();

	int n_nodes = (int)pose_graph.nodes_.size();
	for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
		const PoseGraphNode &t = pose_graph.nodes_[iter_node];
		pose_graph_pruned->nodes_.push_back(t);
	}
	int n_edges = (int)pose_graph.edges_.size();
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
		if (t.uncertain_) {
			if (t.confidence_ > option.edge_prune_threshold_) {
				pose_graph_pruned->edges_.push_back(t);
			}
		} else {
			pose_graph_pruned->edges_.push_back(t);
		}
	}
	return pose_graph_pruned;
}

void GlobalOptimizationGaussNewton::
		OptimizePoseGraph(PoseGraph &pose_graph,
		const GlobalOptimizationConvergenceCriteria &criteria,
		const GlobalOptimizationOption &option) const
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();
	double line_process_weight = ComputeLineProcessWeight(pose_graph, option);

	PrintDebug("[GlobalOptimizationGaussNewton] Optimizing PoseGraph having %d nodes and %d edges. \n",
			n_nodes, n_edges);
	PrintDebug("Line process weight : %f\n", line_process_weight);

	Eigen::VectorXd zeta = ComputeZeta(pose_graph);
	double current_residual, new_residual;
	new_residual = ComputeResidual(pose_graph, zeta,
			line_process_weight, option);
	current_residual = new_residual;

	int valid_edges_num;
	valid_edges_num = UpdateConfidence(pose_graph, zeta,
			line_process_weight, option);

	Eigen::MatrixXd H;
	Eigen::VectorXd b;
	Eigen::VectorXd x = UpdatePoseVector(pose_graph);

	std::tie(H, b) = ComputeLinearSystem(pose_graph, zeta);

	PrintDebug("[Initial     ] residual : %e\n", current_residual);

	bool stop = false;
	if (stop || CheckRightTerm(b, criteria))
		return;

	Timer timer_overall;
	timer_overall.Start();
	int iter;
	for (iter = 0; !stop; iter++) {
		Timer timer_iter;
		timer_iter.Start();

		Eigen::VectorXd delta = H.ldlt().solve(b);

		stop = stop || CheckRelativeIncrement(delta, x, criteria);
		if (stop) {
			break;
		} else {
			std::shared_ptr<PoseGraph> pose_graph_new =
				UpdatePoseGraph(pose_graph, delta);

			Eigen::VectorXd zeta_new;
			zeta_new = ComputeZeta(*pose_graph_new);
			new_residual = ComputeResidual(pose_graph, zeta_new,
					line_process_weight, option);
			stop = stop || CheckRelativeResidualIncrement(
					current_residual, new_residual, criteria);
			if (stop)
				break;
			current_residual = new_residual;

			zeta = zeta_new;
			pose_graph = *pose_graph_new;
			x = UpdatePoseVector(pose_graph);
			valid_edges_num = UpdateConfidence(pose_graph, zeta,
					line_process_weight, option);
			std::tie(H, b) = ComputeLinearSystem(pose_graph, zeta);

			stop = stop || CheckRightTerm(b, criteria);
			if (stop)
				break;
		}
		timer_iter.Stop();
		PrintDebug("[Iteration %02d] residual : %e, valid edges : %d, time : %.3f sec.\n",
				iter, current_residual, valid_edges_num,
				timer_iter.GetDuration() / 1000.0);
		stop = stop || CheckResidual(current_residual, criteria)
				|| CheckMaxIteration(iter, criteria);
	}	// end for
	timer_overall.Stop();
	PrintDebug("[GlobalOptimizationGaussNewton] total time : %.3f sec.\n",
			timer_overall.GetDuration() / 1000.0);
}

void GlobalOptimizationLevenbergMarquardt::
		OptimizePoseGraph(PoseGraph &pose_graph,
		const GlobalOptimizationConvergenceCriteria &criteria,
		const GlobalOptimizationOption &option) const
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();
	double line_process_weight = ComputeLineProcessWeight(pose_graph, option);

	PrintDebug("[GlobalOptimizationLM] Optimizing PoseGraph having %d nodes and %d edges. \n",
			n_nodes, n_edges);
	PrintDebug("Line process weight : %f\n", line_process_weight);

	Eigen::VectorXd zeta = ComputeZeta(pose_graph);
	double current_residual, new_residual;
	new_residual = ComputeResidual(pose_graph, zeta,
			line_process_weight, option);
	current_residual = new_residual;

	int valid_edges_num = UpdateConfidence(pose_graph, zeta,
			line_process_weight, option);

	Eigen::MatrixXd H_I = Eigen::MatrixXd::Identity(n_nodes * 6, n_nodes * 6);
	Eigen::MatrixXd H;
	Eigen::VectorXd b;
	Eigen::VectorXd x = UpdatePoseVector(pose_graph);

	std::tie(H, b) = ComputeLinearSystem(pose_graph, zeta);

	Eigen::VectorXd H_diag = H.diagonal();
	double tau = 1e-5;
	double current_lambda = tau * H_diag.maxCoeff();
	double ni = 2.0;
	double rho = 0.0;

	PrintDebug("[Initial     ] residual : %e, lambda : %e\n",
			current_residual, current_lambda);

	bool stop = false;
	stop = stop || CheckRightTerm(b, criteria);
	if (stop)
		return;

	Timer timer_overall;
	timer_overall.Start();
	for (int iter = 0; !stop; iter++) {
		Timer timer_iter;
		timer_iter.Start();
		int lm_count = 0;
		do {
			Eigen::MatrixXd H_LM = H + current_lambda * H_I;
			Eigen::VectorXd delta = H_LM.ldlt().solve(b);

			stop = stop || CheckRelativeIncrement(delta, x, criteria);
			if (!stop) {
				std::shared_ptr<PoseGraph> pose_graph_new =
						UpdatePoseGraph(pose_graph, delta);

				Eigen::VectorXd zeta_new;
				zeta_new = ComputeZeta(*pose_graph_new);
				new_residual = ComputeResidual(pose_graph, zeta_new,
						line_process_weight, option);
				rho = (current_residual - new_residual) /
						(delta.dot(current_lambda * delta + b) + 1e-3);
				if (rho > 0) {
					stop = stop || CheckRelativeResidualIncrement(
							current_residual, new_residual, criteria);
					if (stop)
						break;
					double alpha = 1. - pow((2 * rho - 1), 3);
					alpha = (std::min)(alpha, criteria.upper_scale_factor_);
					double scaleFactor = (std::max)
							(criteria.lower_scale_factor_, alpha);
					current_lambda *= scaleFactor;
					ni = 2;
					current_residual = new_residual;

					zeta = zeta_new;
					pose_graph = *pose_graph_new;
					x = UpdatePoseVector(pose_graph);
					valid_edges_num = UpdateConfidence(pose_graph, zeta,
							line_process_weight, option);
					std::tie(H, b) = ComputeLinearSystem(pose_graph, zeta);

					stop = stop || CheckRightTerm(b, criteria);
					if (stop)
						break;
				} else {
					current_lambda *= ni;
					ni *= 2;
				}
			}
			lm_count++;
			stop = stop || CheckMaxIterationLM(lm_count, criteria);
		} while (!((rho > 0) || stop));
		timer_iter.Stop();
		if (!stop) {
			PrintDebug("[Iteration %02d] residual : %e, valid edges : %d, time : %.3f sec.\n",
					iter, current_residual, valid_edges_num,
					timer_iter.GetDuration() / 1000.0);
		}
		stop = stop || CheckResidual(current_residual, criteria)
				|| CheckMaxIteration(iter, criteria);
	}	// end for
	timer_overall.Stop();
	PrintDebug("[GlobalOptimizationLM] total time : %.3f sec.\n",
			timer_overall.GetDuration() / 1000.0);
}

void GlobalOptimization(
		PoseGraph &pose_graph,
		const GlobalOptimizationMethod &method
		/* = GlobalOptimizationLevenbergMarquardt() */,
		const GlobalOptimizationConvergenceCriteria &criteria
		/* = GlobalOptimizationConvergenceCriteria() */,
		const GlobalOptimizationOption &option
		/* = GlobalOptimizationOption() */)
{
	std::shared_ptr<PoseGraph> pose_graph_pre =
			std::make_shared<PoseGraph>();
	*pose_graph_pre = pose_graph;
	method.OptimizePoseGraph(*pose_graph_pre, criteria, option);
	auto pose_graph_pruned = CreatePoseGraphWithoutInvalidEdges(
			*pose_graph_pre, option);
	method.OptimizePoseGraph(*pose_graph_pruned, criteria, option);
	CompensateReferencePoseGraphNode(*pose_graph_pruned,
			pose_graph, option.reference_node_);
	pose_graph = *pose_graph_pruned;
}

}	// namespace three
