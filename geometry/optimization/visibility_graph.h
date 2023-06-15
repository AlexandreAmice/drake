#ifndef VISIBILITY_GRAPH_H
#define VISIBILITY_GRAPH_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/rational_forward_kinematics.h"

using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SparseMatrix;

SparseMatrix ComputeVisibilityGraph(
    const Eigen::Ref<const MatrixXd>& points,
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::systems::Context<double>& plant_context,
    const drake::multibody::RationalForwardKinematics* rat_fk,
    const Eigen::Ref<const MatrixXd>& q_star, int num_samples, double tol);

#endif // VISIBILITY_GRAPH_H