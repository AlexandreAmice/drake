#include "drake/multibody/rational_forward_kinematics_old/redundant_inequality_pruning.h"

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace {

GTEST_TEST(RedundantHyperplaneInequalities, PruneSimpleBoundingBox) {
  Eigen::Matrix<double, 12, 3> A_redundant;
  Eigen::Matrix<double, 12, 1> b_redundant;

  Eigen::MatrixXd I3 = Eigen::MatrixXd::Identity(3, 3);
  Eigen::MatrixXd one3 = Eigen::MatrixXd::Ones(3, 1);
  const float scale = 0.25;

  // clang-format off
  A_redundant << I3, -I3, I3, -I3;
  b_redundant << one3, one3, scale*one3, scale*one3;
  // clang-format on

  std::vector<int> redundant_indices =
      FindRedundantInequalitiesInHPolyhedronByIndex(A_redundant, b_redundant);
  std::vector<int> redundant_indices_expected = {0, 1, 2, 3, 4, 5};

  EXPECT_TRUE(redundant_indices == redundant_indices_expected);
}

GTEST_TEST(RedundantHyperplaneInequalities, PruneSimpleBoundingBox2) {
  Eigen::Matrix<double, 5, 2> A_redundant;
  Eigen::Matrix<double, 5, 1> b_redundant = Eigen::MatrixXd::Ones(5, 1);
  b_redundant(4) = 0;

  // clang-format off
  A_redundant <<  1, -1,  // redundant
                 -1,  1,
                 -1, -1,
                  1,  1,  // redundant
                  1,  0;
  // clang-format on

  std::vector<int> redundant_indices =
      FindRedundantInequalitiesInHPolyhedronByIndex(A_redundant, b_redundant);
  std::vector<int> redundant_indices_expected = {0, 3};

  EXPECT_TRUE(redundant_indices == redundant_indices_expected);
}

GTEST_TEST(RedundantHyperplaneInequalities, RemoveWithTighten) {
  Eigen::Matrix<double, 5, 2> A;
  // clang-format off
  A << 1, 0,
       0, 1,
       -1, 0,
       0, -1,
       1, 0;
  // clang-format on
  Eigen::Matrix<double, 5, 1> b;
  b << 1, 1, 1, 1, 0.5;
  // With tighten=0, the first constraint x(0) <= 1 is redundant.
  EXPECT_EQ(FindRedundantInequalitiesInHPolyhedronByIndex(A, b, 0.),
            std::vector<int>({0}));
  // with tighten = 0.6, none of the constraint is redundant.
  EXPECT_EQ(FindRedundantInequalitiesInHPolyhedronByIndex(A, b, 0.6).size(), 0);
}

}  // namespace
}  // namespace rational_old
}  // namespace multibody
}  // namespace drake
