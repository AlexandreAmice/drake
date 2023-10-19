#include "drake/solvers/approximate_semidefinite_program.h"

#include <gtest/gtest.h>
#include <iostream>

#include "drake/common/ssize.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/solvers/test/mathematical_program_test_util.h"

namespace drake {
namespace solvers {
namespace test {

void CheckDiagonallyDominantDualConeOuterApproximation(
    std::unique_ptr<MathematicalProgram> sdp_prog) {
  // Copy the psd constraints so that we can check that sdp_prog isn't modified.
  std::vector<Binding<PositiveSemidefiniteConstraint>>original_psd_constraints;
  std::copy(sdp_prog->positive_semidefinite_constraints().begin(),
            sdp_prog->positive_semidefinite_constraints().end(),
            std::back_inserter(original_psd_constraints));

  auto prog = sdp_prog->Clone();
  MakeDiagonallyDominantDualConeOuterApproximation(prog);

  // Approximated program should have the same costs.
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog->generic_costs(), sdp_prog->generic_costs()));
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog->linear_costs(), sdp_prog->linear_costs()));
  EXPECT_TRUE(IsVectorOfBindingEqual(prog->quadratic_costs(),
                                     sdp_prog->quadratic_costs()));
  // Approximated program should have most of the same constraints.
  EXPECT_TRUE(IsVectorOfBindingEqual(prog->generic_constraints(),
                                     sdp_prog->generic_constraints()));

  // Diagonally Dominant Dual Cone program will different linear constraints
  // which are tested later.

  EXPECT_TRUE(IsVectorOfBindingEqual(prog->linear_equality_constraints(),
                                     sdp_prog->linear_equality_constraints()));
  EXPECT_TRUE(IsVectorOfBindingEqual(prog->bounding_box_constraints(),
                                     sdp_prog->bounding_box_constraints()));
  EXPECT_TRUE(IsVectorOfBindingEqual(prog->lorentz_cone_constraints(),
                                     sdp_prog->lorentz_cone_constraints()));
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog->rotated_lorentz_cone_constraints(),
                             sdp_prog->rotated_lorentz_cone_constraints()));

  // Prog should not have any semidefinite constraints, but the semidefinite
  // constraints of sdp_prog should be untouched.
  EXPECT_TRUE(
      IsVectorOfBindingEqual(original_psd_constraints,
                             sdp_prog->positive_semidefinite_constraints()));
  EXPECT_EQ(ssize(prog->positive_semidefinite_constraints()), 0);

  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog->linear_matrix_inequality_constraints(),
                             sdp_prog->linear_matrix_inequality_constraints()));
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog->linear_matrix_inequality_constraints(),
                             sdp_prog->linear_matrix_inequality_constraints()));
  EXPECT_TRUE(IsVectorOfBindingEqual(prog->exponential_cone_constraints(),
                                     sdp_prog->exponential_cone_constraints()));
  EXPECT_TRUE(
      IsVectorOfBindingEqual(prog->linear_complementarity_constraints(),
                             sdp_prog->linear_complementarity_constraints()));

  EXPECT_TRUE(
      CompareMatrices(sdp_prog->initial_guess(), prog->initial_guess()));
    std::cout << "passed up to linear equality test" << std::endl;

  // Now we test the linear constraints of the approximated program
  int num_new_linear_constraints_expected = 0;
  for (const auto& constraint : sdp_prog->positive_semidefinite_constraints()) {
    num_new_linear_constraints_expected +=
        std::pow(constraint.evaluator()->matrix_rows(), 2);
  }
  EXPECT_EQ(ssize(prog->linear_constraints()),
            ssize(sdp_prog->linear_constraints()) +
                num_new_linear_constraints_expected);
std:: cout << "tested constraint size" << std::endl;
std:: cout << "num linear constraints sdp " << ssize(sdp_prog->linear_constraints()) << std::endl;
std:: cout << "num linear constraints new " << ssize(prog->linear_constraints()) << std::endl;
  // Ensure that all the linear constraints of the original program are in the
  // approximating program.
  for (const auto& original_linear_constraint :
       sdp_prog->linear_constraints()) {
    std:: cout << "loop entered" << std::endl;
    bool constraint_found = false;
    for (const auto& new_linear_constraint : prog->linear_constraints()) {
      std:: cout << "testing binding equality" << std::endl;
      ::testing::AssertionResult assertion_result{
          IsBindingEqual(original_linear_constraint, new_linear_constraint)};
      if (assertion_result) {
        constraint_found = true;
        break;
      }
    }
    if (!constraint_found) {
      EXPECT_TRUE(::testing::AssertionFailure()
                  << fmt::format("Constraint {} not found.",
                                 original_linear_constraint));
    }
  }
}

GTEST_TEST(MakeSemidefiniteRelaxationTest,
           DiagonallyDominantDualConeApproximation) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");

  CheckDiagonallyDominantDualConeOuterApproximation(
      std::unique_ptr<MathematicalProgram>(&prog));
}

}  // namespace test
}  // namespace solvers
}  // namespace drake