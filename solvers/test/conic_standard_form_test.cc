#include "drake/solvers/conic_standard_form.h"

#include <gtest/gtest.h>

#include "drake/common/ssize.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/aggregate_costs_constraints.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/clarabel_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"
#include "drake/solvers/test/linear_program_examples.h"
#include "drake/solvers/test/second_order_cone_program_examples.h"
#include "drake/solvers/test/semidefinite_program_examples.h"
#include "drake/solvers/test/sos_examples.h"

namespace drake {
namespace solvers {
namespace test {

namespace {
const double kInf = std::numeric_limits<double>::infinity();
void CheckParseToConicStandardForm(const MathematicalProgram& prog) {
  ConicStandardForm standard_form{prog};
  std::unique_ptr<MathematicalProgram> prog_standard_form =
      standard_form.MakeProgram();

  ProgramAttributes expected_output_constraint_attributes(
      std::initializer_list<ProgramAttribute>{
          // Output cones.
          ProgramAttribute::kLinearEqualityConstraint,
          ProgramAttribute::kLinearConstraint,
          ProgramAttribute::kLorentzConeConstraint,
          ProgramAttribute::kPositiveSemidefiniteConstraint,
      });

  auto original_result = Solve(prog);
  auto standard_form_result = Solve(*prog_standard_form);
  EXPECT_EQ(original_result.get_solution_result(),
            standard_form_result.get_solution_result());
  if (original_result.get_solution_result() == SolutionResult::kSolutionFound) {
    const double kTol =
        prog.positive_semidefinite_constraints().size() +
                    prog.linear_matrix_inequality_constraints().size() ==
                0
            ? 1e-10
            : 1e-6;
    EXPECT_NEAR(original_result.get_optimal_cost(),
                standard_form_result.get_optimal_cost(), kTol);
    EXPECT_TRUE(CompareMatrices(
        original_result.get_x_val(),
        standard_form_result.get_x_val().head(prog.decision_variables().size()),
        kTol, MatrixCompareType::relative));
  }
}
}  // namespace

TEST_P(LinearProgramTest, TestLP) {
  CheckParseToConicStandardForm(*prob()->prog());
}

INSTANTIATE_TEST_SUITE_P(
    ConicStandardFormTest, LinearProgramTest,
    ::testing::Combine(::testing::ValuesIn(linear_cost_form()),
                       ::testing::ValuesIn(linear_constraint_form()),
                       ::testing::ValuesIn(linear_problems())));

TEST_F(InfeasibleLinearProgramTest0, TestInfeasible) {
  CheckParseToConicStandardForm(*prog_);
}

TEST_F(UnboundedLinearProgramTest0, TestUnbounded) {
  CheckParseToConicStandardForm(*prog_);
}

TEST_P(TestEllipsoidsSeparation, TestSOCP) {
  CheckParseToConicStandardForm(prog_);
}
INSTANTIATE_TEST_SUITE_P(
    ConicStandardFormTest, TestEllipsoidsSeparation,
    ::testing::ValuesIn(GetEllipsoidsSeparationProblems()));

TEST_P(TestQPasSOCP, TestSOCP) {
  CheckParseToConicStandardForm(prog_socp_);
}
INSTANTIATE_TEST_SUITE_P(ConicStandardFormTest, TestQPasSOCP,
                         ::testing::ValuesIn(GetQPasSOCPProblems()));

TEST_P(TestFindSpringEquilibrium, TestSOCP) {
  CheckParseToConicStandardForm(prog_);
}
INSTANTIATE_TEST_SUITE_P(
    ConicStandardFormTest, TestFindSpringEquilibrium,
    ::testing::ValuesIn(GetFindSpringEquilibriumProblems()));

GTEST_TEST(TestSos, SimpleSos1) {
  SimpleSos1 dut;
  CheckParseToConicStandardForm(dut.prog());
}

GTEST_TEST(TestSos, UnivariateNonnegative1) {
  UnivariateNonnegative1 dut;
  CheckParseToConicStandardForm(dut.prog());
}

GTEST_TEST(TestSdp, TestTrivialSDP) {
  // TODO(Alexandre.Amice) get from semidefinite_program_example.h
  MathematicalProgram prog;

  auto S = prog.NewSymmetricContinuousVariables<2>("S");

  // S is p.s.d
  prog.AddPositiveSemidefiniteConstraint(S);

  // S(1, 0) = 1
  prog.AddBoundingBoxConstraint(1, 1, S(1, 0));

  // Min S.trace()
  prog.AddLinearCost(S.cast<symbolic::Expression>().trace());
  CheckParseToConicStandardForm(prog);
}

// This tests LMI constraints.
GTEST_TEST(TestSdp, SolveEigenvalueProblem) {
  // TODO(Alexandre.Amice) get from semidefinite_program_example.h
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>("x");
  Eigen::Matrix3d F1;
  // clang-format off
  F1 << 1, 0.2, 0.3,
        0.2, 2, -0.1,
        0.3, -0.1, 4;
  Eigen::Matrix3d F2;
  F2 << 2, 0.4, 0.7,
        0.4, -1, 0.1,
        0.7, 0.1, 5;
  // clang-format on
  auto z = prog.NewContinuousVariables<1>("z");
  prog.AddLinearMatrixInequalityConstraint(
      {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity(), -F1, -F2}, {z, x});

  const Eigen::Vector2d x_lb(0.1, 1);
  const Eigen::Vector2d x_ub(2, 3);
  prog.AddBoundingBoxConstraint(x_lb, x_ub, x);

  prog.AddLinearCost(z(0));

  CheckParseToConicStandardForm(prog);
}

// GTEST_TEST(TestSdp, SolveSDPwithSecondOrderConeExample1) {
//  // TODO(Alexandre.Amice) get from semidefinite_program_example.h
//  MathematicalProgram prog;
//  auto X = prog.NewSymmetricContinuousVariables<3>();
//  auto x = prog.NewContinuousVariables<3>();
//  Eigen::Matrix3d C0;
//  // clang-format off
//  C0 << 2, 1, 0,
//        1, 2, 1,
//        0, 1, 2;
//  // clang-format on
//  prog.AddLinearCost((C0 * X.cast<symbolic::Expression>()).trace() + x(0));
//  prog.AddLinearConstraint(
//      (Eigen::Matrix3d::Identity() * X.cast<symbolic::Expression>()).trace() +
//          x(0) ==
//      1);
//  prog.AddLinearConstraint(
//      (Eigen::Matrix3d::Ones() * X.cast<symbolic::Expression>()).trace() +
//          x(1) + x(2) ==
//      0.5);
//  prog.AddPositiveSemidefiniteConstraint(X);
//  prog.AddLorentzConeConstraint(x.cast<symbolic::Expression>());
//
//  std::unordered_map<Binding<Constraint>, MatrixX<symbolic::Expression>>
//      constraint_to_dual_variable_map;
//  auto dual_prog =
//      CreateDualConvexProgram(prog, &constraint_to_dual_variable_map);
//  CheckPrimalDualSolution(prog, *dual_prog, constraint_to_dual_variable_map);
//}

// GTEST_TEST(TestSdp, SolveSDPwithSecondOrderConeExample2) {
//  // TODO(Alexandre.Amice) get from semidefinite_program_example.h
//  MathematicalProgram prog;
//  const auto X = prog.NewSymmetricContinuousVariables<3>();
//  const auto x = prog.NewContinuousVariables<1>()(0);
//  prog.AddLinearCost(X(0, 0) + X(1, 1) + x);
//  prog.AddBoundingBoxConstraint(0, kInf, x);
//  prog.AddLinearConstraint(X(0, 0) + 2 * X(1, 1) + X(2, 2) + 3 * x == 3);
//  Vector3<symbolic::Expression> lorentz_cone_expr;
//  lorentz_cone_expr << X(0, 0), X(1, 1) + x, X(1, 1) + X(2, 2);
//  prog.AddLorentzConeConstraint(lorentz_cone_expr);
//  prog.AddLinearConstraint(X(1, 0) + X(2, 1) == 1);
//  prog.AddPositiveSemidefiniteConstraint(X);
//
//  std::unordered_map<Binding<Constraint>, MatrixX<symbolic::Expression>>
//      constraint_to_dual_variable_map;
//  auto dual_prog =
//      CreateDualConvexProgram(prog, &constraint_to_dual_variable_map);
//  CheckPrimalDualSolution(prog, *dual_prog, constraint_to_dual_variable_map);
//}

}  // namespace test
}  // namespace solvers
}  // namespace drake
