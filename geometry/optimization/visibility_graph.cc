#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/rational_forward_kinematics.h"
#include "drake/geometry/optimization/visibility_graph.h"

using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<int, 
                            Eigen::Dynamic, 
                            Eigen::Dynamic,
                            Eigen::RowMajor> SparseMatrix;

SparseMatrix ComputeVisibilityGraph(
    const Ref<const MatrixXd>& points,
    const drake::multibody::MultibodyPlant<double>& plant,
    const systems::Context<double>& plant_context,
    const drake::multibody::RationalForwardKinematics* rat_fk,
    const Ref<const MatrixXd>& q_star, int num_samples, double tol) {
  const int numPoints = points.rows();
  SparseMatrix adjacencyMatrix(numPoints, numPoints);
  // adjacencyMatrix.reserve(Eigen::VectorXi::Constant(numPoints, 4)); //
  // Reserve space for 4 non-zero elements per row (assuming most points are
  // visible)

  bool pointInCollision =
      [&](const MultibodyPlant<double>& plant,
          const systems::Context<double>& plant_context, double tol) {

        const auto& query_port = plant.get_geometry_query_input_port();
        const auto& query_object =
            query_port.Eval<geometry::QueryObject<double>>(plant_context);
        double max_distance =
            0.5;  // Drake will ignore the geometry pair if their
                  // distance is surely above this max_distance.
        const std::vector<geometry::SignedDistancePair<double>>
            signed_distance_pairs =
                query_object.ComputeSignedDistancePairwiseClosestPoints(
                    max_distance);

        for (const auto& signed_distance_pair : signed_distance_pairs) {
          if (signed_distance_pair.distance <= tol) {
            return true;
          }
        }

        return false;
      }

  bool isVisible =
      [&](const Eigen::VectorXd& point1, const Eigen::VectorXd& point2,
          const MultibodyPlant<double>& plant,
          const systems::Context<double>& plant_context,
          const drake::multibody::RationalForwardKinematics* rat_fk,
          const Ref<const MatrixXd>& q_star, double tol, int N) {
        for (int i = 0; i < N; ++i) {
          double t = static_cast<double>(i) /
                     (N - 1);  // Parameter value between 0 and 1
          Eigen::VectorXd interpPoint = point1 + t * (point2 - point1);
          auto qInterPoint = rat_fk->ComputeQValue(cspace_variable, q_star);
          plant.SetPositions(&plant_context, qInterPoint);
          if (pointInCollision(plant, plant_context, tol)) {
            // The interpolated point is in collision, return false
            return false;
          }
        }
        // All evenly spaced points are not in collision, return true
        return true;
      };

  for (int i = 0; i < numPoints; ++i) {
    for (int j = i + 1; j < numPoints; ++j) {
      if (isVisible(points.row(i), points.row(j), plant, plant_context, rat_fk,
                    q_star)) {
        adjacencyMatrix.coeffRef(i, j) = 1;
        adjacencyMatrix.coeffRef(j, i) = 1;
      }
    }
  }

  adjacencyMatrix.makeCompressed();
  return adjacencyMatrix;
}