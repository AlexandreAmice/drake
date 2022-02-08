#pragma once
#include "drake/geometry/optimization/hpolyhedron.h"
#include <unordered_set>


namespace drake {
namespace multibody {
  using HPolyhedron = geometry::optimization::HPolyhedron;

  /**
   * Generates a symmetric HPolyhedron centered around the point @p seed_point. The polytope has
   * 2*dim(seed_point)*num_perm_dim*num_rot faces. This is achieved by rotating the unit dim(seed_point) unit box
   * centered at seed_point around the first perm_dim axis (starting at 0) at num_rot evenly space angles
   */
   HPolyhedron GenerateSeedingPolytope(const Eigen::VectorXd seed_point, const int num_perm_dim, const int num_rot);


  /**
   * compute the set {Cx + d | x in P}. Assumes that C is left invertible
  */
  HPolyhedron SameDimensionalAffineTransform(const Eigen::MatrixXd C, const Eigen::VectorXd d, const HPolyhedron P);

  /**
   * Generates k members of SO(n). This is done by evenly spacing k*n points on the unit circle. Every member of
   * SO(n) can be represented as similarity transform diag([[a_{i},b_{i}],[-a_{i},b_{i}], 1]) for i = 1, ..., n//2
   * with ceil(n/2) ones at the end
   */
   std::vector<Eigen::MatrixXd> MakeKCanonicalSOnMembers(const int k, const int n);

   /**
    * Generates k permutation matrices of dimension n exchanging the first dimension with the next k dimensions
    */
    std::vector<Eigen::MatrixXd> MakeKFirstDimSwapsOfDimN(const int k, const int n);

} // name multibody
} // namespace drake