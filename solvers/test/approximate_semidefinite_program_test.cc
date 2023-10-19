#include "drake/solvers/approximate_semidefinite_program.h"

#include <gtest/gtest.h>

#include "drake/common/ssize.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/solvers/test/mathematical_program_test_util.h"

namespace drake {
namespace solvers {
namespace {

void CheckDiagonallyDominantDualConeOuterApproximation(
    std::unique_ptr<MathematicalProgram> sdp_prog) {
  MathematicalProgram prog = sdp_prog->Clone();
  const int num_psd_constraints =
      ssize(sdp_prog->positive_semidefinite_constraints());
  int num_new_linear_constraints_expected = 0;
  for (const auto& constraint : prog.positive_semidefinite_constraints()) {
    num_new_linear_constraints_expected +=
        std::pow(constraint.evaluator()->matrix_rows(), 2);
  }
  // Approximated program should have the same costs.
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog.generic_costs(), sdp_prog->generic_costs()));
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog.linear_costs(), sdp_prog->linear_costs()));
  EXPECT_TRUE(IsVectorOfBindingEqual(prog.quadratic_costs(),
                                     sdp_prog->quadratic_costs()));

  // Approximated program should have most of the same constraints.
  EXPECT_TRUE(IsVectorOfBindingEqual(prog.generic_constraints(),
                                     sdp_prog->generic_constraints()));

  // Diagonally Dominant Dual Cone program will different linear constraints
  // which are tested later.

  EXPECT_TRUE(IsVectorOfBindingEqual(prog.linear_equality_constraints(),
                                     sdp_prog->linear_equality_constraints()));
  EXPECT_TRUE(IsVectorOfBindingEqual(prog.bounding_box_constraints(),
                                     sdp_prog->bounding_box_constraints()));
  EXPECT_TRUE(IsVectorOfBindingEqual(prog.lorentz_cone_constraints(),
                                     sdp_prog->lorentz_cone_constraints()));
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog.rotated_lorentz_cone_constraints(),
                             sdp_prog->rotated_lorentz_cone_constraints()));
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog.positive_semidefinite_constraints(),
                             sdp_prog->positive_semidefinite_constraints()));
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog.linear_matrix_inequality_constraints(),
                             sdp_prog->linear_matrix_inequality_constraints()));
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog.linear_matrix_inequality_constraints(),
                             sdp_prog->linear_matrix_inequality_constraints()));
  EXPECT_TRUE(IsVectorOfBindingEqual(prog.exponential_cone_constraints(),
                                     sdp_prog->exponential_cone_constraints()));
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog.linear_complementarity_constraints(),
                             sdp_prog->linear_complementarity_constraints()));

  EXPECT_TRUE(CompareMatrices(sdp_prog->initial_guess(), prog.initial_guess()));

  // Now we test the linear constraints of the approximated program
  EXPECT_EQ(ssize(prog.linear_constraints()),
            ssize(sdp_prog->linear_constraints()) +
                num_new_linear_constraints_expected);
  // Ensure that all the linear constraints of the original program are in the
  // approximating program.
  for (const auto& original_linear_constraint :
       sdp_prog->linear_constraints()) {
    bool constraint_found = false;
    for(const auto&)
  }
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, NoCostsNorConstraints) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  const auto relaxation = MakeSemidefiniteRelaxation(prog);

  // X is 3x3 symmetric.
  EXPECT_EQ(relaxation->num_vars(), 6);
  // X ≽ 0.
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  // X(-1,-1) = 1.
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
}

}  // namespace
}  // namespace solvers
}  // namespace drake