#include "drake/geometry/optimization/polytope_cover.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace geometry {
namespace optimization {
namespace {
GTEST_TEST(AxisAlignedBox, Test) {
  const Eigen::Vector2d lo(1, -3);
  const Eigen::Vector2d up(2, -1);
  AxisAlignedBox dut(lo, up);
  EXPECT_TRUE(CompareMatrices(dut.lo(), lo));
  EXPECT_TRUE(CompareMatrices(dut.up(), up));
  AxisAlignedBox scaled = dut.Scale(0.5);
  EXPECT_TRUE(CompareMatrices(scaled.lo(), Eigen::Vector2d(1.25, -2.5)));
  EXPECT_TRUE(CompareMatrices(scaled.up(), Eigen::Vector2d(1.75, -1.5)));

  // Compute the outer box of the 1-norm countour |x| + |y|<=1
  Eigen::Matrix<double, 4, 2> C;
  // clang-format off
  C <<  1, 1,
        1, -1,
        -1, 1,
        -1, -1;
  // clang-format on
  Eigen::Vector4d d(1, 1, 1, 1);
  const AxisAlignedBox outer = AxisAlignedBox::OuterBox(C, d);
  const double tol = 1E-6;
  EXPECT_TRUE(CompareMatrices(outer.lo(), Eigen::Vector2d(-1, -1), tol));
  EXPECT_TRUE(CompareMatrices(outer.up(), Eigen::Vector2d(1, 1), tol));
}

GTEST_TEST(FindInscribedBox, Test1) {
  // Find the largest box in the region |x|+|y|<=1.
  Eigen::Matrix<double, 4, 2> C;
  // clang-format off
  C <<  1, 1,
        1, -1,
        -1, 1,
        -1, -1;
  // clang-format on
  Eigen::Vector4d d(1, 1, 1, 1);
  FindInscribedBox dut(C, d, {}, std::nullopt);
  // Maximize the box volume as product(box_up(i) - box_lo(i))
  Eigen::Matrix<double, 2, 4> A;
  // clang-format off
  A << 1, 0, -1, 0,
       0, 1, 0, -1;
  // clang-format on
  dut.mutable_prog()->AddMaximizeGeometricMeanCost(
      A, Eigen::Vector2d::Zero(),
      Vector4<symbolic::Variable>(dut.box_up()(0), dut.box_up()(1),
                                  dut.box_lo()(0), dut.box_lo()(1)));
  const auto result = solvers::Solve(dut.prog());
  EXPECT_TRUE(result.is_success());
  const double tol{1E-4};
  EXPECT_TRUE(CompareMatrices(result.GetSolution(dut.box_lo()),
                              Eigen::Vector2d(-0.5, -0.5), tol));
  EXPECT_TRUE(CompareMatrices(result.GetSolution(dut.box_up()),
                              Eigen::Vector2d(0.5, 0.5), tol));
}

GTEST_TEST(FindInscribedBox, Test2) {
  // Find a box in this region
  // ________
  // |XX XX|
  // |     |
  // |XX XX|
  // --------
  Eigen::Matrix<double, 4, 2> C;
  C << Eigen::Matrix2d::Identity(), -Eigen::Matrix2d::Identity();
  Eigen::Vector4d d(1, 1, 1, 1);
  std::vector<AxisAlignedBox> obstacles;
  obstacles.emplace_back(Eigen::Vector2d(-1, 0.5), Eigen::Vector2d(-0.2, 1));
  obstacles.emplace_back(Eigen::Vector2d(0.2, 0.5), Eigen::Vector2d(1, 1));
  obstacles.emplace_back(Eigen::Vector2d(-1, -1), Eigen::Vector2d(-0.2, -0.5));
  obstacles.emplace_back(Eigen::Vector2d(0.2, -1), Eigen::Vector2d(1, -0.5));
  FindInscribedBox dut(C, d, obstacles, std::nullopt);
  dut.MaximizeBoxVolume();
  const auto result = solvers::Solve(dut.prog());
  EXPECT_TRUE(result.is_success());
  const double tol{1E-6};
  EXPECT_TRUE(CompareMatrices(result.GetSolution(dut.box_lo()),
                              Eigen::Vector2d(-1, -0.5), tol));
  EXPECT_TRUE(CompareMatrices(result.GetSolution(dut.box_up()),
                              Eigen::Vector2d(1, 0.5), tol));
}

GTEST_TEST(FindInscribedBox, AddObstacle) {
  Eigen::Matrix<double, 4, 2> C;
  // clang-format on
  C << 1, 1, 1, -1, -1, 1, -1, -1;
  // clang-format off
  const Eigen::Vector4d d(1, 1, 1, 1);
  FindInscribedBox dut(C, d, {}, std::nullopt);
  const auto b = dut.AddObstacle(AxisAlignedBox(Eigen::Vector2d(-0.5, -0.5),
                                                Eigen::Vector2d(0.5, 0.5)));
  // box on the left of the obstacle.
  dut.mutable_prog()->AddBoundingBoxConstraint(1, 1, b(1));
  dut.MaximizeBoxVolume();
  const auto result = solvers::Solve(dut.prog());
  EXPECT_TRUE(result.is_success());
  const double tol{1E-4};
  EXPECT_TRUE(CompareMatrices(result.GetSolution(dut.box_lo()),
                              Eigen::Vector2d(-0.75, -0.25), tol));
  EXPECT_TRUE(CompareMatrices(result.GetSolution(dut.box_up()),
                              Eigen::Vector2d(-0.5, 0.25), tol));
}
}  // namespace
}  // namespace optimization
}  // namespace geometry
}  // namespace drake
