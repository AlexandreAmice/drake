#include <gtest/gtest.h>

#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/geometry/optimization/dev/cspace_free_path_separating_plane.h"

namespace drake {
namespace geometry {
namespace optimization {

GTEST_TEST(CalcPlane, TestAllSymbolic) {
  symbolic::Variable mu("mu");
  for (int plane_degree = 1; plane_degree < 3; ++plane_degree) {
    const int num_coeffs_per_poly = plane_degree + 1;
    const int num_decision_vars = 4 * num_coeffs_per_poly;
    Eigen::Matrix<symbolic::Variable, Eigen::Dynamic, 1> decision_vars{
        num_decision_vars};
    for (int i = 0; i < num_decision_vars; ++i) {
      decision_vars(i) = symbolic::Variable("plane_var" + std::to_string(i));
    }
    Vector3<symbolic::Polynomial> a;
    symbolic::Polynomial b;
    CalcPathPlane<symbolic::Variable, symbolic::Variable, symbolic::Polynomial>(
        decision_vars, mu, plane_degree, &a, &b);
    int decision_var_ctr = 0;
    for (int i = 0; i < 3; ++i) {
      symbolic::Polynomial::MapType expected_poly_map;
      for(int j = 0; j < num_coeffs_per_poly; ++j) {
        expected_poly_map.insert({symbolic::Monomial{mu, plane_degree-j},
                                 decision_vars(decision_var_ctr)});
        ++decision_var_ctr;
      }
      EXPECT_PRED2(
          symbolic::test::PolyEqual, a(i),
          symbolic::Polynomial(expected_poly_map));
    }
    symbolic::Polynomial::MapType expected_poly_map;
      for(int j = 0; j < num_coeffs_per_poly; ++j) {
        expected_poly_map.insert({symbolic::Monomial{mu, plane_degree-j},
                                 decision_vars(decision_var_ctr)});
        ++decision_var_ctr;
      }
    EXPECT_PRED2(symbolic::test::PolyEqual, b,
                 symbolic::Polynomial(expected_poly_map));
  }
}
//
//// Test decision_vars taking double values and s takes symbolic values.
//GTEST_TEST(CalcPlane, TestDoubleDecisionVariableSymbolicS) {
//  symbolic::Variable s("s");
//  Eigen::Matrix<double, 8, 1> decision_var_vals;
//  for (int i = 0; i < 8; ++i) {
//    decision_var_vals(i) = i + 1;
//  }
//  Vector3<symbolic::Polynomial> a;
//  symbolic::Polynomial b;
//  CalcPlane<double, symbolic::Variable, symbolic::Polynomial>(
//      decision_var_vals, Vector1<symbolic::Variable>(s),
//      SeparatingPlaneOrder::kAffine, &a, &b);
//  for (int i = 0; i < 3; ++i) {
//    EXPECT_PRED2(symbolic::test::PolyEqual, a(i),
//                 symbolic::Polynomial(decision_var_vals(i) * s +
//                                      decision_var_vals(3 + i)));
//  }
//  EXPECT_PRED2(
//      symbolic::test::PolyEqual, b,
//      symbolic::Polynomial(decision_var_vals(6) * s + decision_var_vals(7)));
//}
//
//// Test with both decision variable and s taking double values.
//GTEST_TEST(CalcPlane, TestDoubleDecisionVariableDoubleS) {
//  const double s = 2;
//  Eigen::Matrix<double, 8, 1> decision_var_vals;
//  for (int i = 0; i < 8; ++i) {
//    decision_var_vals(i) = i + 1;
//  }
//  Eigen::Vector3d a;
//  double b;
//  CalcPlane<double, double, double>(decision_var_vals, Vector1d(s),
//                                    SeparatingPlaneOrder::kAffine, &a, &b);
//  for (int i = 0; i < 3; ++i) {
//    EXPECT_EQ(a(i), decision_var_vals(i) * s + decision_var_vals(3 + i));
//  }
//  EXPECT_EQ(b, decision_var_vals(6) * s + decision_var_vals(7));
//}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake