#include <memory>
#include <type_traits>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/scs_solver.h"

namespace drake {
namespace solvers {
namespace {

template <typename Solver>
class SolverCacheTest : public ::testing::Test {};
using CacheSolvers = ::testing::Types<OsqpSolver, ScsSolver>;
TYPED_TEST_SUITE(SolverCacheTest, CacheSolvers);

TYPED_TEST(SolverCacheTest, LargeCoefficientChanges) {
  TypeParam solver;
  if (!solver.available()) GTEST_SKIP();
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<1>();
  auto cost = prog.AddQuadraticCost(Eigen::MatrixXd::Identity(1, 1),
                                    Eigen::VectorXd::Zero(1), 1e20, x);
  prog.AddBoundingBoxConstraint(0, 2, x);
  SolverOptions options;
  options.SetOption(solver.id(), "retain_solver_cache", 1);
  prog.SetSolverOption(solver.id(), "eps_abs", 1e-8);
  prog.SetSolverOption(solver.id(), "eps_rel", 1e-8);
  auto result = solver.Solve(prog, {}, options);
  for (double coefficient : {1e16, -1.0}) {
    cost.evaluator()->UpdateCoefficients(
        Eigen::MatrixXd::Identity(1, 1),
        Eigen::VectorXd::Constant(1, coefficient), 7);
    solver.Solve(prog, {}, options, &result);
  }
  const auto fresh = solver.Solve(prog);
  ASSERT_TRUE(result.is_success());
  EXPECT_TRUE(CompareMatrices(result.get_x_val(), fresh.get_x_val(), 1e-4));
  EXPECT_NEAR(result.get_optimal_cost(), fresh.get_optimal_cost(), 1e-4);
}

TYPED_TEST(SolverCacheTest, RecoveryAndIndependentCaches) {
  TypeParam solver;
  if (!solver.available()) GTEST_SKIP();
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<2>();
  auto cost = prog.AddQuadraticCost(Eigen::Matrix2d::Identity(),
                                    Eigen::Vector2d::Zero(), x);
  auto box = prog.AddBoundingBoxConstraint(0, 1, x);
  SolverOptions options;
  options.SetOption(solver.id(), "retain_solver_cache", 1);
  auto first = solver.Solve(prog, {}, options);
  auto second = solver.Solve(prog, {}, options);
  ASSERT_NE(first.get_solver_cache(), second.get_solver_cache());
  // Constants change reported objectives, without changing the optimizer.
  cost.evaluator()->UpdateCoefficients(Eigen::Matrix2d::Identity(),
                                       Eigen::Vector2d::Zero(), 7);
  solver.Solve(prog, {}, options, &first);
  EXPECT_NEAR(first.get_optimal_cost(), 7, 1e-4);
  EXPECT_NEAR(second.get_optimal_cost(), 0, 1e-4);
  solver.Solve(prog, {}, options, &second);
  EXPECT_NEAR(second.get_optimal_cost(), 7, 1e-4);
  box.evaluator()->UpdateLowerBound(Eigen::Vector2d::Constant(2));
  solver.Solve(prog, {}, options, &first);
  EXPECT_FALSE(first.is_success());
  box.evaluator()->UpdateLowerBound(Eigen::Vector2d::Constant(0.5));
  solver.Solve(prog, {}, options, &first);
  ASSERT_TRUE(first.is_success());
  EXPECT_TRUE(
      CompareMatrices(first.get_x_val(), Eigen::Vector2d::Constant(0.5), 1e-4));
}

TYPED_TEST(SolverCacheTest, SettingsAndInitialGuesses) {
  TypeParam solver;
  if (!solver.available()) GTEST_SKIP();
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<2>();
  prog.AddQuadraticCost(Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero(),
                        x);
  prog.AddBoundingBoxConstraint(1, 2, x);
  SolverOptions options;
  options.SetOption(solver.id(), "retain_solver_cache", 1);
  options.SetOption(solver.id(), "solver_cache_rebuild_policy", "error");
  auto result = solver.Solve(prog, {}, options);
  const auto* cache = result.get_solver_cache();
  options.SetOption(solver.id(), "eps_abs", 1e-7);
  EXPECT_THROW(solver.Solve(prog, {}, options, &result), std::runtime_error);
  EXPECT_EQ(result.get_solver_cache(), cache);
  options.SetOption(solver.id(), "solver_cache_rebuild_policy", "allow");
  solver.Solve(prog, Eigen::Vector2d(1, 1), options, &result);
  ASSERT_TRUE(result.is_success());
  EXPECT_EQ(result.template get_solver_details<TypeParam>().cache.status,
            SolverCacheStatus::kRebuilt);
  options.SetOption(solver.id(), "warm_start_from_cache", 0);
  solver.Solve(prog, Eigen::Vector2d(2, 2), options, &result);
  ASSERT_TRUE(result.is_success());
  EXPECT_EQ(result.template get_solver_details<TypeParam>().cache.status,
            SolverCacheStatus::kReused);
  for (const auto& key : {"retain_solver_cache", "warm_start_from_cache"}) {
    auto bad = options;
    bad.SetOption(solver.id(), key, 2);
    EXPECT_THROW(solver.Solve(prog, {}, bad, &result), std::exception);
    bad.SetOption(solver.id(), key, "yes");
    EXPECT_THROW(solver.Solve(prog, {}, bad, &result), std::exception);
  }
  auto bad = options;
  bad.SetOption(solver.id(), "solver_cache_rebuild_policy", "sometimes");
  EXPECT_THROW(solver.Solve(prog, {}, bad, &result), std::exception);
  EXPECT_TRUE(result.has_solver_cache());
}

TYPED_TEST(SolverCacheTest, LifetimesAndStructure) {
  auto prog = std::make_unique<MathematicalProgram>();
  const auto x = prog->NewContinuousVariables<2>();
  prog->AddQuadraticCost(Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero(),
                         x);
  auto box = prog->AddBoundingBoxConstraint(1, 2, x);
  MathematicalProgramResult result;
  SolverOptions options;
  {
    TypeParam original_solver;
    if (!original_solver.available()) GTEST_SKIP();
    options.SetOption(original_solver.id(), "retain_solver_cache", 1);
    original_solver.Solve(*prog, {}, options, &result);
  }
  TypeParam solver;
  solver.Solve(*prog, {}, options, &result);
  EXPECT_EQ(result.template get_solver_details<TypeParam>().cache.status,
            SolverCacheStatus::kReused);
  prog->RemoveConstraint(box);
  solver.Solve(*prog, {}, options, &result);
  EXPECT_EQ(result.template get_solver_details<TypeParam>().cache.status,
            SolverCacheStatus::kRebuilt);
  prog.reset();
  EXPECT_TRUE(result.has_solver_cache());
  EXPECT_TRUE(result.is_success());
}

GTEST_TEST(OsqpSolverCacheTest, LargeMatrixChanges) {
  OsqpSolver solver;
  if (!solver.available()) GTEST_SKIP();
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<1>();
  auto cost = prog.AddQuadraticCost(Eigen::MatrixXd::Identity(1, 1),
                                    -Eigen::VectorXd::Ones(1), x);
  auto constraint = prog.AddLinearConstraint(
      Eigen::MatrixXd::Identity(1, 1), Eigen::VectorXd::Zero(1),
      Eigen::VectorXd::Constant(1, 2), x);
  prog.SetSolverOption(solver.id(), "eps_abs", 1e-8);
  prog.SetSolverOption(solver.id(), "eps_rel", 1e-8);
  SolverOptions options;
  options.SetOption(solver.id(), "retain_solver_cache", 1);
  options.SetOption(solver.id(), "solver_cache_rebuild_policy", "error");
  auto result = solver.Solve(prog, {}, options);
  for (double coefficient : {1e16, 1.0}) {
    cost.evaluator()->UpdateCoefficients(
        coefficient * Eigen::MatrixXd::Identity(1, 1),
        -Eigen::VectorXd::Ones(1));
    constraint.evaluator()->UpdateCoefficients(
        coefficient * Eigen::MatrixXd::Identity(1, 1), Eigen::VectorXd::Zero(1),
        Eigen::VectorXd::Constant(1, 2 * coefficient));
    solver.Solve(prog, {}, options, &result);
  }
  const auto fresh = solver.Solve(prog);
  ASSERT_TRUE(result.is_success());
  EXPECT_TRUE(CompareMatrices(result.get_x_val(), fresh.get_x_val(), 1e-5));
  EXPECT_NEAR(result.get_optimal_cost(), fresh.get_optimal_cost(), 1e-5);
}

GTEST_TEST(OsqpSolverCacheTest, RepeatedVariablesInQuadraticCost) {
  OsqpSolver solver;
  if (!solver.available()) GTEST_SKIP();
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<2>();
  VectorXDecisionVariable vars(2);
  vars << x(0), x(0);
  Eigen::Matrix2d Q;
  Q << 2, 0.2, 0.2, 2;
  auto cost = prog.AddQuadraticCost(Q, Eigen::Vector2d(-1, -1), vars);
  prog.AddQuadraticCost(x(1) * x(1));
  prog.AddBoundingBoxConstraint(0, 1, x);
  SolverOptions options;
  options.SetOption(solver.id(), "retain_solver_cache", 1);
  options.SetOption(solver.id(), "eps_abs", 1e-8);
  options.SetOption(solver.id(), "eps_rel", 1e-8);
  auto result = solver.Solve(prog, {}, options);
  Q(0, 1) = Q(1, 0) = 0.8;
  cost.evaluator()->UpdateCoefficients(Q, Eigen::Vector2d(-1, -1));
  solver.Solve(prog, {}, options, &result);
  EXPECT_NEAR(result.GetSolution(x(0)), 1.0 / 2.8, 1e-5);
}

}  // namespace
}  // namespace solvers
}  // namespace drake
