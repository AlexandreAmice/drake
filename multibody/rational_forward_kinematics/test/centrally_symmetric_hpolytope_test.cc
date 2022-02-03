#include "drake/multibody/rational_forward_kinematics/centrally_symmetric_hpolytope.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include <vector>
#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace {
using HPolyhedron = geometry::optimization::HPolyhedron;
GTEST_TEST(AffineTransformationTest, RotateLinf3Ball) {
  HPolyhedron box = HPolyhedron::MakeUnitBox(3);

  Eigen::Matrix3d X_rot = math::RotationMatrixd::MakeXRotation(-M_PI_2).matrix();
  Eigen::Matrix3d Y_rot = math::RotationMatrixd::MakeYRotation(-M_PI_2).matrix();
  Eigen::Matrix3d Z_rot = math::RotationMatrixd::MakeZRotation(-M_PI_2).matrix();
  Eigen::Matrix3d tot_rot = Z_rot * Y_rot * X_rot;
  Eigen::Vector3d translation;
  translation << 1, 0, 0;
  HPolyhedron transformed_box = SameDimensionalAffineTransform(tot_rot, translation, box);

  Eigen::Matrix3d X_rot_inv = math::RotationMatrixd::MakeXRotation(M_PI_2).matrix();
  Eigen::Matrix3d Y_rot_inv = math::RotationMatrixd::MakeYRotation(M_PI_2).matrix();
  Eigen::Matrix3d Z_rot_inv = math::RotationMatrixd::MakeZRotation(M_PI_2).matrix();
  Eigen::Matrix3d tot_rot_inv = X_rot_inv * Y_rot_inv * Z_rot_inv;

  Eigen::MatrixXd rotated_box_A_expected = box.A() * tot_rot_inv;
  Eigen::VectorXd rotated_box_b_expected = box.b() + rotated_box_A_expected * translation;

  EXPECT_TRUE(CompareMatrices(transformed_box.A(), rotated_box_A_expected));
  EXPECT_TRUE(CompareMatrices(transformed_box.b(), rotated_box_b_expected));
}

GTEST_TEST(SOnMembersTest, Generate3_2D_rotations){
  const int n = 2;
  const int k = 3;
  std::vector<Eigen::MatrixXd> members = MakeKCanonicalSOnMembers(k, n);
  // TODO(Alex.Amice) make formal test
  for (Eigen::MatrixXd m : members) {
    std::cout << m << std::endl << std::endl;
  }
}

GTEST_TEST(SOnMembersTest, Generate5_3D_rotations){
  const int n = 3;
  const int k = 5;
  std::vector<Eigen::MatrixXd> members = MakeKCanonicalSOnMembers(k, n);
  // TODO(Alex.Amice) make formal test
  for (Eigen::MatrixXd m : members) {
    std::cout << m << std::endl << std::endl;
  }
}

}  // namespace
}  // namespace multibody
}  // namespace drake
