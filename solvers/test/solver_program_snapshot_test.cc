#include "drake/solvers/solver_program_snapshot.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/limit_malloc.h"

namespace drake {
namespace solvers {
namespace internal {
namespace {

GTEST_TEST(SolverProgramSnapshotTest, ChangesAndSharedBindings) {
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<2>();
  auto cost = prog.AddQuadraticCost(Eigen::Matrix2d::Identity(),
                                    Eigen::Vector2d::Zero(), x);
  prog.AddCost(cost);
  auto bounds = prog.AddBoundingBoxConstraint(-1, 1, x);
  SolverProgramSnapshot snapshot(prog);
  EXPECT_TRUE(snapshot.CheckStructure(prog).empty());
  EXPECT_TRUE(snapshot.IsUnchanged());
  bounds.evaluator()->UpdateLowerBound(Eigen::Vector2d::Constant(-2));
  EXPECT_TRUE(snapshot.CheckStructure(prog).empty());
  EXPECT_FALSE(snapshot.IsUnchanged());
  EXPECT_TRUE(snapshot.constraints[0].MatrixMatches());
  snapshot.constraints[0].Refresh();
  EXPECT_TRUE(snapshot.IsUnchanged());
  cost.evaluator()->UpdateCoefficients(2 * Eigen::Matrix2d::Identity(),
                                       Eigen::Vector2d::Ones());
  EXPECT_FALSE(snapshot.costs[0].MatrixMatches());
  EXPECT_FALSE(snapshot.costs[1].VectorsMatch());
  prog.SetVariableScaling(x(0), 2);
  EXPECT_EQ(snapshot.CheckStructure(prog), "variable scaling changed");
}

GTEST_TEST(SolverProgramSnapshotTest, StructureAndSparseCoefficients) {
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<2>();
  auto c = prog.AddLinearConstraint(Eigen::Matrix2d::Identity(),
                                    Eigen::Vector2d::Zero(),
                                    Eigen::Vector2d::Ones(), x);
  SolverProgramSnapshot snapshot(prog);
  Eigen::Matrix2d A = Eigen::Matrix2d::Identity();
  A(0, 1) = 1e-20;
  c.evaluator()->UpdateCoefficients(A, Eigen::Vector2d::Zero(),
                                    Eigen::Vector2d::Ones());
  EXPECT_TRUE(snapshot.CheckStructure(prog).empty());
  EXPECT_FALSE(snapshot.IsUnchanged());
  snapshot.constraints[0].Refresh();
  EXPECT_TRUE(snapshot.IsUnchanged());
  prog.RemoveConstraint(c);
  EXPECT_EQ(snapshot.CheckStructure(prog), "bindings changed");
}

GTEST_TEST(SolverProgramSnapshotTest, EvaluatorStorageAndAllocation) {
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<2>();
  auto constraint = prog.AddLinearConstraint(Eigen::Matrix2d::Identity(),
                                             Eigen::Vector2d::Zero(),
                                             Eigen::Vector2d::Ones(), x);
  SolverProgramSnapshot snapshot(prog);
  {
    test::LimitMalloc guard;
    EXPECT_TRUE(snapshot.CheckStructure(prog).empty());
    EXPECT_TRUE(snapshot.IsUnchanged());
  }
  // Updating a retained evaluator may replace its sparse coefficient storage.
  constraint.evaluator()->UpdateCoefficients(Eigen::MatrixXd::Ones(3, 2),
                                             Eigen::VectorXd::Zero(3),
                                             Eigen::VectorXd::Ones(3));
  EXPECT_EQ(snapshot.CheckStructure(prog), "bindings changed");
  constraint.evaluator()->UpdateCoefficients(2 * Eigen::Matrix2d::Identity(),
                                             Eigen::Vector2d::Zero(),
                                             Eigen::Vector2d::Ones());
  EXPECT_TRUE(snapshot.CheckStructure(prog).empty());
  EXPECT_FALSE(snapshot.IsUnchanged());
  snapshot.constraints[0].Refresh();
  {
    test::LimitMalloc guard;
    EXPECT_TRUE(snapshot.IsUnchanged());
  }
}

}  // namespace
}  // namespace internal
}  // namespace solvers
}  // namespace drake
