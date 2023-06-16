#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/rational/rational_forward_kinematics.h"
#include "drake/geometry/optimization/visibility_graph.h"
#include "drake/geometry/optimization/convex_set.h"
#include <iostream>
#include <string>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace drake::multibody;
using namespace drake::systems;
using namespace drake::geometry;

// typedef Eigen::SparseMatrix<int, Eigen::RowMajor> SparseMatrix;

// Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> 
Eigen::SparseMatrix<int, Eigen::RowMajor> ComputeVisibilityGraph(
    const Eigen::Ref<const MatrixXd>& points,
    const drake::geometry::optimization::ConvexSets& c_obstacles,
    const MultibodyPlant<double>& plant,
    Context<double>& plant_context,
    //const RationalForwardKinematics* rat_fk,
    //const Eigen::Ref<const VectorXd>& q_star, 
    int num_samples, double tolerence) {

  const int numPoints = points.rows();
  Eigen::SparseMatrix<int, Eigen::RowMajor> adjacencyMatrix(numPoints, numPoints); //= Eigen::MatrixXi::Zero(numPoints, numPoints);
 
  adjacencyMatrix.reserve(Eigen::VectorXi::Constant(numPoints, 4)); //
  // Reserve space for 4 non-zero elements per row
  const int nc = static_cast<int>(c_obstacles.size());
  const auto& query_port = plant.get_geometry_query_input_port();
  
  auto pointInCollision =
      [&]() {
        const auto& query_object =
            query_port.Eval<QueryObject<double>>(plant_context);
        double max_distance =
            0.25;  // Drake will ignore the geometry pair if their
                  // distance is surely above this max_distance.
        const std::vector<SignedDistancePair<double>>
            signed_distance_pairs =
                query_object.ComputeSignedDistancePairwiseClosestPoints(
                    max_distance);

        for (const auto& signed_distance_pair : signed_distance_pairs) {
          if (signed_distance_pair.distance <= tolerence) {
            return true;
          }
        }

        return false;
      };

  auto isVisible =
      [&](const Eigen::Ref<const Eigen::VectorXd>& point1, 
          const Eigen::Ref<const Eigen::VectorXd>& point2) {

        for (int i = 0; i < num_samples; ++i) {
          double t = static_cast<double>(i) /
                     (num_samples - 1);  // Parameter value between 0 and 1
          Eigen::VectorXd qInterpPoint = point1 + t * (point2 - point1);
          //auto qInterpPoint = rat_fk->ComputeQValue(interpPoint, q_star);
          plant.SetPositions(&plant_context, qInterpPoint);
          for (int j = 0; j < nc; ++i) {
            if (c_obstacles[j]->PointInSet(qInterpPoint)) return false;
          }
          if (pointInCollision()) {
            // The interpolated point is in collision, return false
            return false;
          }
        }
        // All evenly spaced points are not in collision, return true
        return true;
      };

  for (int i = 0; i < numPoints; ++i) {
    if (i%50==0){
        drake::log()->info(fmt::format(
          "Remaining Nodes to check {}/{}", i, numPoints));
      }
    for (int j = i + 1; j < numPoints; ++j) {
      
      if (isVisible(points.row(i), 
                    points.row(j))) {
        adjacencyMatrix.insert(i, j) = 1;
        adjacencyMatrix.insert(j, i) = 1;
      }
    }
  }
  // adjacencyMatrix.makeCompressed();
  return adjacencyMatrix;
}