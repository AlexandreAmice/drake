#pragma once
// #include <Eigen/Sparse>
#include <Eigen/Dense>
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/rational/rational_forward_kinematics.h"
#include "drake/geometry/optimization/convex_set.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
//typedef Eigen::SparseMatrix<int, Eigen::RowMajor> SparseMatrix;

/** Configuration options for the IRIS algorithm.

@ingroup geometry_optimization
*/

/** Scutnum @p points : are vertices in configurationspace that 

@ingroup geometry_optimization
*/

// Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> 
Eigen::SparseMatrix<int, Eigen::RowMajor> ComputeVisibilityGraph(
    const Eigen::Ref<const MatrixXd>& points,
    const drake::geometry::optimization::ConvexSets& c_obstacles,
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>& plant_context,
    //const drake::multibody::RationalForwardKinematics* rat_fk,
    //const Eigen::Ref<const VectorXd>& q_star, 
    int num_samples, double tol);
