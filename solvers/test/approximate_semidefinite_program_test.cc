#include "drake/solvers/approximate_semidefinite_program.h"

#include <gtest/gtest.h>

#include "drake/common/ssize.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/test/mathematical_program_test_util.h"
#include "drake/solvers/test/semidefinite_program_examples.h"

namespace drake {
namespace solvers {
namespace test {

namespace {
void SdpsToTest(std::vector<std::unique_ptr<MathematicalProgram>>* sdp_progs) {
  ScsSolver scs_solver;
  sdp_progs->push_back(std::move(TestTrivialSDP(
      scs_solver, 0,
      false /* don't test the results of the constructed program */)));
  sdp_progs->push_back(FindCommonLyapunov(
      scs_solver, {}, 0,
      false /* don't test the results of the constructed program */));
  sdp_progs->push_back(FindOuterEllipsoid(
      scs_solver, {}, 0,
      false /* don't test the results of the constructed program */));
  sdp_progs->push_back(SolveEigenvalueProblem(
      scs_solver, {}, 0,
      false /* don't test the results of the constructed program */));
  sdp_progs->push_back(SolveSDPwithSecondOrderConeExample1(
      scs_solver, 0,
      false /* don't test the results of the constructed program */));
  sdp_progs->push_back(SolveSDPwithOverlappingVariables(
      scs_solver, 0,
      false /* don't test the results of the constructed program */));
}

template <typename T>
void CheckVectorOfBindingsContainedInOther(std::vector<Binding<T>> subset,
                                           std::vector<Binding<T>> superset) {
  // Ensure that all the rotated_lorentz_cone constraints of
  // the original program are in the approximating program.
  for (const auto& b1 : subset) {
    bool constraint_found = false;
    for (const auto& b2 : superset) {
      ::testing::AssertionResult assertion_result{IsBindingEqual(b1, b2)};
      if (assertion_result) {
        constraint_found = true;
        break;
      }
    }
    if (!constraint_found) {
      EXPECT_TRUE(::testing::AssertionFailure()
                  << fmt::format("Constraint {} not found.", b1));
    }
  }
}

}  // namespace

GTEST_TEST(MakeSemidefiniteApproximationTest, DiagonallyDominantApproximation) {
  std::vector<std::unique_ptr<MathematicalProgram>> test_progs;
  SdpsToTest(&test_progs);
  for (const auto& sdp_prog : test_progs) {
    // Copy the psd constraints so that we can check that sdp_prog isn't
    // modified.

    std::vector<Binding<PositiveSemidefiniteConstraint>>
        original_psd_constraints;
    std::copy(sdp_prog->positive_semidefinite_constraints().begin(),
              sdp_prog->positive_semidefinite_constraints().end(),
              std::back_inserter(original_psd_constraints));
    auto prog = sdp_prog->Clone();
    MakeDiagonallyDominantInnerApproximation(prog.get());
    // Prog should not have any semidefinite constraints, but the semidefinite
    // constraints of sdp_prog should be untouched.
    EXPECT_TRUE(
        IsVectorOfBindingEqual(original_psd_constraints,
                               sdp_prog->positive_semidefinite_constraints()));
    EXPECT_EQ(ssize(prog->positive_semidefinite_constraints()), 0);

    // Approximated program should have the same costs.
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->generic_costs(),
                                       sdp_prog->generic_costs()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->linear_costs(), sdp_prog->linear_costs()));
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->quadratic_costs(),
                                       sdp_prog->quadratic_costs()));
    // Approximated program should have most of the same constraints.
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->generic_constraints(),
                                       sdp_prog->generic_constraints()));

    // The diagonally dominant program will have different linear
    // constraints which are tested later.
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->linear_equality_constraints(),
                               sdp_prog->linear_equality_constraints()));
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->bounding_box_constraints(),
                                       sdp_prog->bounding_box_constraints()));
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->lorentz_cone_constraints(),
                                       sdp_prog->lorentz_cone_constraints()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->rotated_lorentz_cone_constraints(),
                               sdp_prog->rotated_lorentz_cone_constraints()));

    EXPECT_TRUE(IsVectorOfBindingEqual(
        prog->linear_matrix_inequality_constraints(),
        sdp_prog->linear_matrix_inequality_constraints()));
    EXPECT_TRUE(IsVectorOfBindingEqual(
        prog->linear_matrix_inequality_constraints(),
        sdp_prog->linear_matrix_inequality_constraints()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->exponential_cone_constraints(),
                               sdp_prog->exponential_cone_constraints()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->linear_complementarity_constraints(),
                               sdp_prog->linear_complementarity_constraints()));

    // We have a different number of variables in sdp_prog and prog due to the
    // DD relaxation, and so we do NOT expect the initial guess to have been
    // preserved. The initial guess for the original variables in sdp_prog
    // should be preserved.
    for (int i = 0; i < sdp_prog->decision_variables().size(); ++i) {
      const symbolic::Variable var = sdp_prog->decision_variables()(i);

      EXPECT_TRUE((std::isnan(sdp_prog->GetInitialGuess(var)) &&
                   std::isnan(prog->GetInitialGuess(var))) ||
                  sdp_prog->GetInitialGuess(var) == prog->GetInitialGuess(var));
    }

    // Now we test the linear constraints of the approximated program
    int num_new_linear_constraints_expected = 0;
    for (const auto& constraint :
         sdp_prog->positive_semidefinite_constraints()) {
      num_new_linear_constraints_expected +=
          std::pow(constraint.evaluator()->matrix_rows(), 2);
    }
    EXPECT_EQ(ssize(prog->linear_constraints()),
              ssize(sdp_prog->linear_constraints()) +
                  num_new_linear_constraints_expected);
    // Ensure that all the linear constraints of the original program are in the
    // approximating program.
    CheckVectorOfBindingsContainedInOther<LinearConstraint>(
        sdp_prog->linear_constraints(), prog->linear_constraints());
  }
}

GTEST_TEST(MakeSemidefiniteApproximationTest,
           ScaledDiagonallyDominantApproximation) {
  std::vector<std::unique_ptr<MathematicalProgram>> test_progs;
  SdpsToTest(&test_progs);
  for (const auto& sdp_prog : test_progs) {
    // Copy the psd constraints so that we can check that sdp_prog isn't
    // modified.

    std::vector<Binding<PositiveSemidefiniteConstraint>>
        original_psd_constraints;
    std::copy(sdp_prog->positive_semidefinite_constraints().begin(),
              sdp_prog->positive_semidefinite_constraints().end(),
              std::back_inserter(original_psd_constraints));
    auto prog = sdp_prog->Clone();
    MakeScaledDiagonallyDominantInnerApproximation(prog.get());
    // Prog should not have any semidefinite constraints, but the semidefinite
    // constraints of sdp_prog should be untouched.
    EXPECT_TRUE(
        IsVectorOfBindingEqual(original_psd_constraints,
                               sdp_prog->positive_semidefinite_constraints()));
    EXPECT_EQ(ssize(prog->positive_semidefinite_constraints()), 0);

    // Approximated program should have the same costs.
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->generic_costs(),
                                       sdp_prog->generic_costs()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->linear_costs(), sdp_prog->linear_costs()));
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->quadratic_costs(),
                                       sdp_prog->quadratic_costs()));
    // Approximated program should have most of the same constraints.
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->generic_constraints(),
                                       sdp_prog->generic_constraints()));

    EXPECT_TRUE(IsVectorOfBindingEqual(prog->linear_constraints(),
                                       sdp_prog->linear_constraints()));
    // The approximated program will have additional linear equality constraints
    // due to the introduction of slack variables.

    EXPECT_TRUE(IsVectorOfBindingEqual(prog->bounding_box_constraints(),
                                       sdp_prog->bounding_box_constraints()));

    EXPECT_TRUE(IsVectorOfBindingEqual(prog->lorentz_cone_constraints(),
                                       sdp_prog->lorentz_cone_constraints()));
    // The scaled diagonally dominant dual cone program will have different
    // RotatedLorentzConeConstraints which are tested later.

    EXPECT_TRUE(IsVectorOfBindingEqual(
        prog->linear_matrix_inequality_constraints(),
        sdp_prog->linear_matrix_inequality_constraints()));
    EXPECT_TRUE(IsVectorOfBindingEqual(
        prog->linear_matrix_inequality_constraints(),
        sdp_prog->linear_matrix_inequality_constraints()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->exponential_cone_constraints(),
                               sdp_prog->exponential_cone_constraints()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->linear_complementarity_constraints(),
                               sdp_prog->linear_complementarity_constraints()));

    // We have a different number of variables in sdp_prog and prog due to the
    // DD relaxation, and so we do NOT expect the initial guess to have been
    // preserved. The initial guess for the original variables in sdp_prog
    // should be preserved.
    for (int i = 0; i < sdp_prog->decision_variables().size(); ++i) {
      const symbolic::Variable var = sdp_prog->decision_variables()(i);

      EXPECT_TRUE((std::isnan(sdp_prog->GetInitialGuess(var)) &&
                   std::isnan(prog->GetInitialGuess(var))) ||
                  sdp_prog->GetInitialGuess(var) == prog->GetInitialGuess(var));
    }

    // Now we test the rotated lorentz cone constraints of the approximated
    // program.
    int num_new_rotate_lorentz_cone_expected = 0;
    for (const auto& constraint :
         sdp_prog->positive_semidefinite_constraints()) {
      num_new_rotate_lorentz_cone_expected +=
          constraint.evaluator()->matrix_rows() *
          (constraint.evaluator()->matrix_rows() - 1) / 2;
    }
    EXPECT_EQ(ssize(prog->rotated_lorentz_cone_constraints()),
              ssize(sdp_prog->rotated_lorentz_cone_constraints()) +
                  num_new_rotate_lorentz_cone_expected);
    // Ensure that all the rotated_lorentz_cone constraints of
    // the original program are in the approximating program.
    CheckVectorOfBindingsContainedInOther<RotatedLorentzConeConstraint>(
        sdp_prog->rotated_lorentz_cone_constraints(),
        prog->rotated_lorentz_cone_constraints());

    const int num_additional_equality_constraints =
        ssize(sdp_prog->positive_semidefinite_constraints());
    EXPECT_EQ(ssize(prog->linear_equality_constraints()),
              ssize(sdp_prog->linear_equality_constraints()) +
                  num_additional_equality_constraints);
    // Ensure that all the rotated_lorentz_cone constraints of
    // the original program are in the approximating program.
    CheckVectorOfBindingsContainedInOther<LinearEqualityConstraint>(
        sdp_prog->linear_equality_constraints(),
        prog->linear_equality_constraints());
  }
}

GTEST_TEST(MakeSemidefiniteApproximationTest,
           DiagonallyDominantDualConeApproximation) {
  std::vector<std::unique_ptr<MathematicalProgram>> test_progs;
  SdpsToTest(&test_progs);
  for (const auto& sdp_prog : test_progs) {
    // Copy the psd constraints so that we can check that sdp_prog isn't
    // modified.

    std::vector<Binding<PositiveSemidefiniteConstraint>>
        original_psd_constraints;
    std::copy(sdp_prog->positive_semidefinite_constraints().begin(),
              sdp_prog->positive_semidefinite_constraints().end(),
              std::back_inserter(original_psd_constraints));
    auto prog = sdp_prog->Clone();
    MakeDiagonallyDominantDualConeOuterApproximation(prog.get());
    // Prog should not have any semidefinite constraints, but the semidefinite
    // constraints of sdp_prog should be untouched.
    EXPECT_TRUE(
        IsVectorOfBindingEqual(original_psd_constraints,
                               sdp_prog->positive_semidefinite_constraints()));
    EXPECT_EQ(ssize(prog->positive_semidefinite_constraints()), 0);

    // Approximated program should have the same costs.
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->generic_costs(),
                                       sdp_prog->generic_costs()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->linear_costs(), sdp_prog->linear_costs()));
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->quadratic_costs(),
                                       sdp_prog->quadratic_costs()));
    // Approximated program should have most of the same constraints.
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->generic_constraints(),
                                       sdp_prog->generic_constraints()));

    // The diagonally dominant dual cone program will have different linear
    // constraints which are tested later.
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->linear_equality_constraints(),
                               sdp_prog->linear_equality_constraints()));
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->bounding_box_constraints(),
                                       sdp_prog->bounding_box_constraints()));
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->lorentz_cone_constraints(),
                                       sdp_prog->lorentz_cone_constraints()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->rotated_lorentz_cone_constraints(),
                               sdp_prog->rotated_lorentz_cone_constraints()));

    EXPECT_TRUE(IsVectorOfBindingEqual(
        prog->linear_matrix_inequality_constraints(),
        sdp_prog->linear_matrix_inequality_constraints()));
    EXPECT_TRUE(IsVectorOfBindingEqual(
        prog->linear_matrix_inequality_constraints(),
        sdp_prog->linear_matrix_inequality_constraints()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->exponential_cone_constraints(),
                               sdp_prog->exponential_cone_constraints()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->linear_complementarity_constraints(),
                               sdp_prog->linear_complementarity_constraints()));

    EXPECT_TRUE(
        CompareMatrices(sdp_prog->initial_guess(), prog->initial_guess()));

    // Now we test the linear constraints of the approximated program.
    int num_new_linear_constraints_expected =
        ssize(sdp_prog->positive_semidefinite_constraints());
    EXPECT_EQ(ssize(prog->linear_constraints()),
              ssize(sdp_prog->linear_constraints()) +
                  num_new_linear_constraints_expected);
    // Ensure that all the linear constraints of the original program are in the
    // approximating program.
    CheckVectorOfBindingsContainedInOther<LinearConstraint>(
        sdp_prog->linear_constraints(), prog->linear_constraints());
  }
}

GTEST_TEST(MakeSemidefiniteApproximationTest,
           ScaledDiagonallyDominantDualConeApproximation) {
  std::vector<std::unique_ptr<MathematicalProgram>> test_progs;
  SdpsToTest(&test_progs);
  for (const auto& sdp_prog : test_progs) {
    // Copy the psd constraints so that we can check that sdp_prog isn't
    // modified.

    std::vector<Binding<PositiveSemidefiniteConstraint>>
        original_psd_constraints;
    std::copy(sdp_prog->positive_semidefinite_constraints().begin(),
              sdp_prog->positive_semidefinite_constraints().end(),
              std::back_inserter(original_psd_constraints));
    auto prog = sdp_prog->Clone();
    MakeScaledDiagonallyDominantDualConeOuterApproximation(prog.get());
    // Prog should not have any semidefinite constraints, but the semidefinite
    // constraints of sdp_prog should be untouched.
    EXPECT_TRUE(
        IsVectorOfBindingEqual(original_psd_constraints,
                               sdp_prog->positive_semidefinite_constraints()));
    EXPECT_EQ(ssize(prog->positive_semidefinite_constraints()), 0);

    // Approximated program should have the same costs.
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->generic_costs(),
                                       sdp_prog->generic_costs()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->linear_costs(), sdp_prog->linear_costs()));
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->quadratic_costs(),
                                       sdp_prog->quadratic_costs()));
    // Approximated program should have most of the same constraints.
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->generic_constraints(),
                                       sdp_prog->generic_constraints()));

    EXPECT_TRUE(IsVectorOfBindingEqual(prog->linear_constraints(),
                                       sdp_prog->linear_constraints()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->linear_equality_constraints(),
                               sdp_prog->linear_equality_constraints()));
    EXPECT_TRUE(IsVectorOfBindingEqual(prog->bounding_box_constraints(),
                                       sdp_prog->bounding_box_constraints()));

    EXPECT_TRUE(IsVectorOfBindingEqual(prog->lorentz_cone_constraints(),
                                       sdp_prog->lorentz_cone_constraints()));
    // The scaled diagonally dominant dual cone program will have different
    // RotatedLorentzConeConstraints which are tested later.

    EXPECT_TRUE(IsVectorOfBindingEqual(
        prog->linear_matrix_inequality_constraints(),
        sdp_prog->linear_matrix_inequality_constraints()));
    EXPECT_TRUE(IsVectorOfBindingEqual(
        prog->linear_matrix_inequality_constraints(),
        sdp_prog->linear_matrix_inequality_constraints()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->exponential_cone_constraints(),
                               sdp_prog->exponential_cone_constraints()));
    EXPECT_TRUE(
        IsVectorOfBindingEqual(prog->linear_complementarity_constraints(),
                               sdp_prog->linear_complementarity_constraints()));

    EXPECT_TRUE(
        CompareMatrices(sdp_prog->initial_guess(), prog->initial_guess()));

    // We have a different number of variables in sdp_prog and prog due to the
    // SDD relaxation, and so we do NOT expect the initial guess to have been
    // preserved. The initial guess for the original variables in sdp_prog
    // should be preserved.
    for (int i = 0; i < sdp_prog->decision_variables().size(); ++i) {
      const symbolic::Variable var = sdp_prog->decision_variables()(i);

      EXPECT_TRUE((std::isnan(sdp_prog->GetInitialGuess(var)) &&
                   std::isnan(prog->GetInitialGuess(var))) ||
                  sdp_prog->GetInitialGuess(var) == prog->GetInitialGuess(var));
    }

    // Now we test the rotated lorentz cone constraints of the approximated
    // program.
    int num_new_rotate_lorentz_cone_expected = 0;
    for (const auto& constraint :
         sdp_prog->positive_semidefinite_constraints()) {
      num_new_rotate_lorentz_cone_expected +=
          constraint.evaluator()->matrix_rows() *
          (constraint.evaluator()->matrix_rows() - 1) / 2;
    }
    EXPECT_EQ(ssize(prog->rotated_lorentz_cone_constraints()),
              ssize(sdp_prog->rotated_lorentz_cone_constraints()) +
                  num_new_rotate_lorentz_cone_expected);
    // Ensure that all the rotated_lorentz_cone constraints of
    // the original program are in the approximating program.
    CheckVectorOfBindingsContainedInOther<RotatedLorentzConeConstraint>(
        sdp_prog->rotated_lorentz_cone_constraints(),
        prog->rotated_lorentz_cone_constraints());
  }
}

}  // namespace test
}  // namespace solvers
}  // namespace drake
