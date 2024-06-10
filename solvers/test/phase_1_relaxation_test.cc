#include "drake/solvers/phase_1_relaxation.h"

#include <iostream>

#include <gtest/gtest.h>

#include "drake/common/ssize.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/solvers/test/linear_program_examples.h"
#include "drake/solvers/test/second_order_cone_program_examples.h"

namespace drake {
namespace solvers {
namespace test {

void TestPhase1Relaxation(const MathematicalProgram& prog,
                          bool expect_feasible) {
  std::unordered_map<Binding<Constraint>, VectorXDecisionVariable>
      constraint_to_slack_variables{};
  auto relaxed_prog = RelaxToPhase1Prog(prog, &constraint_to_slack_variables);
//  std::cout << *relaxed_prog << std::endl;
  SolverOptions options;
  options.SetOption(CommonSolverOption::kPrintToConsole, 1);
  const MathematicalProgramResult result =
      Solve(*relaxed_prog, std::nullopt, options);
  EXPECT_TRUE(result.is_success());
//  std::cout << fmt::format("result = {}", result.get_solution_result())
//            << std::endl;
  double s_max = 0;
  for (const auto& [constraint, slack_variables] :
       constraint_to_slack_variables) {
    unused(constraint);
    s_max = std::max(s_max, result.GetSolution(slack_variables).maxCoeff());
  }
  const double kTol = 1e-6;
  if (expect_feasible) {
    EXPECT_LE(s_max, kTol);
  } else {
    EXPECT_GT(s_max, kTol);
  }
}

TEST_P(LinearProgramTest, TestLP) {
  TestPhase1Relaxation(*prob()->prog(), true);
}

// We exclude kLinearFeasibilityProgram and kLinearProgram2 due to their duals
// being non-unique.
INSTANTIATE_TEST_SUITE_P(
    DualConvexProgramTest, LinearProgramTest,
    ::testing::Combine(::testing::ValuesIn(linear_cost_form()),
                       ::testing::ValuesIn(linear_constraint_form()),
                       ::testing::ValuesIn(linear_problems())));

TEST_F(InfeasibleLinearProgramTest0, TestInfeasible) {
  TestPhase1Relaxation(*prog_, false);
}

TEST_F(UnboundedLinearProgramTest0, TestUnbounded) {
  TestPhase1Relaxation(*prog_, true);
}

}  // namespace test
}  // namespace solvers
}  // namespace drake