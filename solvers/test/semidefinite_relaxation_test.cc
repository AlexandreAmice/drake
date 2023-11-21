#include "drake/solvers/semidefinite_relaxation.h"

#include <gtest/gtest.h>

#include "drake/common/ssize.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"

namespace drake {
namespace solvers {
namespace internal {

using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using symbolic::Variable;
using symbolic::Variables;

void SetRelaxationInitialGuess(const Eigen::Ref<const VectorXd>& y_expected,
                               MathematicalProgram* relaxation) {
  const int N = y_expected.size() + 1;
  MatrixX<Variable> X(N, N);
  // relaxation->decision_variables = [y, flat_lower(Y), 1]. Therefore
  // X = [Y  y]
  //     [yᵀ 1]
  int count = 0;
  while (count < y_expected.size()) {
    X(N - 1, count) = relaxation->decision_variable(count);
    X(count, N - 1) = relaxation->decision_variable(count);
    ++count;
  }
  for (int r = 0; r < N - 1; ++r) {
    for (int c = r; c < N - 1; ++c) {
      X(r, c) = relaxation->decision_variable(count);
      X(c, r) = relaxation->decision_variable(count);
      ++count;
    }
  }
  X(N-1, N-1) = relaxation->decision_variable(count);
  ++count;
  DRAKE_ASSERT(count == (N*(N+1))/2);

  VectorXd x_expected(N);
  x_expected << y_expected, 1;
  const MatrixXd X_expected = x_expected * x_expected.transpose();
  relaxation->SetInitialGuess(X, X_expected);
}

Variables GetMinorVariablesFromMakeSemidefiniteRelaxationVars(
    const VectorX<Variable>& semidefinite_relaxation_vars,
    const int original_prog_vars_size, const std::set<int>& minor_indexes) {
  Variables ret;
  int rows_in_X = (-1 + sqrt(1 + 8 * semidefinite_relaxation_vars.rows())) / 2;
  int count = original_prog_vars_size;
  // If X = [Y | y
  //        [- | -]
  //        [yᵀ| 1],
  // then semidefinite_relaxation_vars = [y, flat_lower(Y), 1] with
  // flat_lower(Y) being the columns of the lower triangular part of Y stacked
  // column-wise.
  for (int r = 0; r < rows_in_X - 1; ++r) {
    for (int c = r; c < rows_in_X - 1; ++c) {
      if (minor_indexes.contains(c) && minor_indexes.contains(r)) {
        ret.insert(semidefinite_relaxation_vars(count));
      }
      ++count;
    }
  }
  if (minor_indexes.contains(original_prog_vars_size)) {
    for (const auto& ind : minor_indexes) {
      DRAKE_ASSERT(ind <= original_prog_vars_size);
      // Only insert the variables of the last column if this variable group has
      // the constant monomial.
      if (ind < original_prog_vars_size) {
        ret.insert(semidefinite_relaxation_vars(ind));
      } else {
        ret.insert(semidefinite_relaxation_vars.tail<1>()(0));
      }
    }
  }
  return ret;
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, NoCostsNorConstraints) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  const auto relaxation = MakeSemidefiniteRelaxation(prog, false);

  // X is 3x3 symmetric.
  EXPECT_EQ(relaxation->num_vars(), 6);
  // X ≽ 0.
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  // X(-1,-1) = 1.
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  const Variable one{relaxation->decision_variables().tail<1>()(0)};
  EXPECT_EQ(Variables(relaxation->bounding_box_constraints().at(0).variables()),
            Variables{one});
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, NoCostsNorConstraintsAutoSparse) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  const auto relaxation = MakeSemidefiniteRelaxation(prog, true);

  // X is 3x3 symmetric.
  EXPECT_EQ(relaxation->num_vars(), 6);

  // X(-1,-1) = 1.
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  const Variable one{relaxation->decision_variables().tail<1>()(0)};
  EXPECT_EQ(Variables(relaxation->bounding_box_constraints().at(0).variables()),
            Variables{one});

  // X ≽ 0 is not enforced as there is no term sparsity.
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 0);
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, NoCostsNorConstraintsManualSparse) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  std::map<Variables, bool> variables_to_enforce_sparsity{
      {Variables{y(0)}, true}, {Variables{y(1)}, false}};
  const auto relaxation =
      MakeSemidefiniteRelaxation(prog, variables_to_enforce_sparsity);

  // X is 3x3 symmetric.
  EXPECT_EQ(relaxation->num_vars(), 6);
  const Variable one{relaxation->decision_variables().tail<1>()(0)};

  // X(-1,-1) = 1.
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(Variables(relaxation->bounding_box_constraints().at(0).variables()),
            Variables{one});

  // X([0,2], [0,2]) ≽ 0 and X[1,1] ≽ 0
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(),
            variables_to_enforce_sparsity.size());
  EXPECT_EQ(
      Variables(
          relaxation->positive_semidefinite_constraints().at(0).variables()),
      GetMinorVariablesFromMakeSemidefiniteRelaxationVars(
          relaxation->decision_variables(), prog.num_vars(), {0, 2}));
  EXPECT_EQ(
      Variables(
          relaxation->positive_semidefinite_constraints().at(1).variables()),
      GetMinorVariablesFromMakeSemidefiniteRelaxationVars(
          relaxation->decision_variables(), prog.num_vars(), {1}));
}

GTEST_TEST(MakeSemidefiniteRelaxationTest,
           NoCostsNorConstraintsManualSparseEmptyMap) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  const auto relaxation = MakeSemidefiniteRelaxation(prog, true);

  std::map<Variables, bool> variables_to_enforce_sparsity{
      {Variables{y(0)}, true}, {Variables{y(1)}, false}};

  // X is 3x3 symmetric.
  EXPECT_EQ(relaxation->num_vars(), 6);
  const Variable one{relaxation->decision_variables().tail<1>()(0)};

  // X(-1,-1) = 1.
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(Variables(relaxation->bounding_box_constraints().at(0).variables()),
            Variables{one});

  // X ≽ 0 is not enforced as there is nothing in the map.
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 0);
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, UnsupportedCost) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  prog.AddCost(sin(y[0]));
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, true),
      ".*GenericCost was declared but is not supported.");
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, UnsupportedCostAutoSparse) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  prog.AddCost(sin(y[0]));
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, false),
      ".*GenericCost was declared but is not supported.");
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, UnsupportedCostManualSparse) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  std::map<Variables, bool> variables_to_enforce_sparsity{
      {Variables{y(0)}, true}, {Variables{y(1)}, false}};
  prog.AddCost(sin(y[0]));
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, variables_to_enforce_sparsity),
      ".*GenericCost was declared but is not supported.");
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, UnsupportedConstraint) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  prog.AddConstraint(sin(y[0]) >= 0.2);
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, false),
      ".*GenericConstraint was declared but is not supported.");
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, UnsupportedConstraintAutoSparse) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  prog.AddConstraint(sin(y[0]) >= 0.2);
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, true),
      ".*GenericConstraint was declared but is not supported.");
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, UnsupportedConstraintManualSparse) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  std::map<Variables, bool> variables_to_enforce_sparsity{
      {Variables{y(0)}, true}, {Variables{y(1)}, false}};
  prog.AddConstraint(sin(y[0]) >= 0.2);
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, variables_to_enforce_sparsity),
      ".*GenericConstraint was declared but is not supported.");
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, LinearCost) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  const Vector2d a(0.5, 0.7);
  const double b = 1.3;
  prog.AddLinearCost(a, b, y);
  auto relaxation = MakeSemidefiniteRelaxation(prog, false);

  EXPECT_EQ(relaxation->num_vars(), 6);
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(relaxation->linear_costs().size(), 1);

  const Vector2d y_test(1.3, 0.24);
  SetRelaxationInitialGuess(y_test, relaxation.get());
  EXPECT_NEAR(
      relaxation->EvalBindingAtInitialGuess(relaxation->linear_costs()[0])[0],
      a.transpose() * y_test + b, 1e-12);

  // Confirm that the decision variables of prog are also decision variables
  // of the relaxation.
  std::vector<int> indices = relaxation->FindDecisionVariableIndices(y);
  EXPECT_EQ(indices.size(), 2);
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, LinearCostAutoSparseWithOffset) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<3>("y");
  const Vector3d a(0.5, 0, 0.7);
  const double b = 1.3;
  prog.AddLinearCost(a, b, y);
  auto relaxation = MakeSemidefiniteRelaxation(prog, true);
  const Variable one{relaxation->decision_variables().tail<1>()(0)};

  EXPECT_EQ(relaxation->num_vars(), 10);
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  // The minor corresponding to the outer product of the variables (y(0),
  // y(2), 1) should be psd
  EXPECT_EQ(
      Variables(
          relaxation->positive_semidefinite_constraints().at(0).variables()),
      GetMinorVariablesFromMakeSemidefiniteRelaxationVars(
          relaxation->decision_variables(), prog.num_vars(), {0, 2, 3}));

  // X(-1,-1) = 1.
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(Variables(relaxation->bounding_box_constraints().at(0).variables()),
            Variables{one});

  EXPECT_EQ(relaxation->linear_costs().size(), 1);

  const Vector3d y_test(1.3, 0.24, -2.7);
  SetRelaxationInitialGuess(y_test, relaxation.get());
  EXPECT_NEAR(
      relaxation->EvalBindingAtInitialGuess(relaxation->linear_costs()[0])[0],
      a.transpose() * y_test + b, 1e-12);

  // Confirm that the decision variables of prog are also decision variables
  // of the relaxation.
  std::vector<int> indices = relaxation->FindDecisionVariableIndices(y);
  EXPECT_EQ(indices.size(), 3);
}

 GTEST_TEST(MakeSemidefiniteRelaxationTest, LinearCostAutoSparseNoOffset) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<3>("y");
  const Vector3d a(0.5, 0, 0.7);
  const double b = 0;
  prog.AddLinearCost(a, b, y);
  auto relaxation = MakeSemidefiniteRelaxation(prog, true);
  const Variable one{relaxation->decision_variables().tail<1>()(0)};

  EXPECT_EQ(relaxation->num_vars(), 10);
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  // The minor corresponding to the outer product of the variables (y(0),
  // y(2)) should be psd
  EXPECT_EQ(
      Variables(
          relaxation->positive_semidefinite_constraints().at(0).variables()),
      GetMinorVariablesFromMakeSemidefiniteRelaxationVars(
          relaxation->decision_variables(), prog.num_vars(), {0, 2}));

  // X(-1,-1) = 1.
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(Variables(relaxation->bounding_box_constraints().at(0).variables()),
            Variables{one});

  EXPECT_EQ(relaxation->linear_costs().size(), 1);

  const Vector3d y_test(1.3, 0.24, -2.7);
  SetRelaxationInitialGuess(y_test, relaxation.get());
  EXPECT_NEAR(
      relaxation->EvalBindingAtInitialGuess(relaxation->linear_costs()[0])[0],
      a.transpose() * y_test + b, 1e-12);

  // Confirm that the decision variables of prog are also decision variables
  // of the relaxation.
  std::vector<int> indices = relaxation->FindDecisionVariableIndices(y);
  EXPECT_EQ(indices.size(), 3);
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, QuadraticCost) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  const Vector2d yd(0.5, 0.7);
  prog.AddQuadraticErrorCost(Matrix2d::Identity(), yd, y);
  auto relaxation = MakeSemidefiniteRelaxation(prog);

  EXPECT_EQ(relaxation->num_vars(), 6);
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(relaxation->linear_costs().size(), 1);

  SetRelaxationInitialGuess(yd, relaxation.get());
  EXPECT_NEAR(
      relaxation->EvalBindingAtInitialGuess(relaxation->linear_costs()[0])[0],
      0, 1e-12);
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, BoundingBoxConstraint) {
  MathematicalProgram prog;
  const int N_VARS = 2;
  const auto y = prog.NewContinuousVariables<2>("y");

  VectorXd lb(N_VARS);
  lb << -1.5, -2.0;

  const double kInf = std::numeric_limits<double>::infinity();
  VectorXd ub(N_VARS);
  ub << kInf, 2.3;

  prog.AddBoundingBoxConstraint(lb, ub, y);

  auto relaxation = MakeSemidefiniteRelaxation(prog);

  // First bounding box constraint is X(-1,-1) = 1, and we add one
  // constraint, so we expect there to be two bounding box constraints
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 2);

  auto bbox_evaluator = relaxation->bounding_box_constraints()[1].evaluator();

  EXPECT_TRUE(CompareMatrices(lb, bbox_evaluator->lower_bound()));
  EXPECT_TRUE(CompareMatrices(ub, bbox_evaluator->upper_bound()));

  const int N_CONSTRAINTS = 3;
  VectorXd b(N_CONSTRAINTS);
  b << -lb[0], -lb[1], ub[1];  // all of the finite lower/upper bounds.

  MatrixXd A(N_CONSTRAINTS, 2);
  // Rows of A:
  // 1. Lower bound y[0]
  // 2. Lower bound y[1]
  // 3. Upper bound y[1]
  A << -1, 0, 0, -1, 0, 1;

  const Vector2d y_test(1.3, 0.24);
  SetRelaxationInitialGuess(y_test, relaxation.get());

  // First linear constraint (in the new decision variables) is 0 ≤
  // (Ay-b)(Ay-b)ᵀ, where A and b represent all of the constraints stacked.
  auto linear_constraint = relaxation->linear_constraints()[0];
  VectorXd value = relaxation->EvalBindingAtInitialGuess(linear_constraint);
  MatrixXd expected =
      (A * y_test - b) * (A * y_test - b).transpose() - b * b.transpose();
  EXPECT_TRUE(CompareMatrices(
      Eigen::Map<MatrixXd>(value.data(), N_CONSTRAINTS, N_CONSTRAINTS),
      expected, 1e-12));
  value = linear_constraint.evaluator()->lower_bound();
  expected = -b * b.transpose();
  EXPECT_TRUE(CompareMatrices(
      Eigen::Map<MatrixXd>(value.data(), N_CONSTRAINTS, N_CONSTRAINTS),
      expected, 1e-12));
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, LinearConstraint) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  MatrixXd A0(3, 2);
  A0 << 0.5, 0.7, -0.2, 0.4, -2.3, -4.5;
  const double kInf = std::numeric_limits<double>::infinity();
  const Vector3d lb0(1.3, -kInf, 0.25);
  const Vector3d ub0(5.6, 0.1, kInf);
  prog.AddLinearConstraint(A0, lb0, ub0, y);
  Matrix2d A1;
  A1 << 0.2, 1.2, 0.24, -0.1;
  const Vector2d lb1(-0.74, -0.3);
  const Vector2d ub1(-0.75, 0.9);
  prog.AddLinearConstraint(A1, lb1, ub1, Vector2<Variable>(y[1], y[0]));
  Matrix2d A1_reordered;
  A1_reordered.col(0) = A1.col(1);
  A1_reordered.col(1) = A1.col(0);

  auto relaxation = MakeSemidefiniteRelaxation(prog);

  EXPECT_EQ(relaxation->num_vars(), 6);
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(relaxation->linear_constraints().size(), 3);

  const Vector2d y_test(1.3, 0.24);
  SetRelaxationInitialGuess(y_test, relaxation.get());

  // First linear constraint is lb0 ≤ A0y ≤ ub0.
  EXPECT_TRUE(CompareMatrices(
      A0, relaxation->linear_constraints()[0].evaluator()->GetDenseA()));
  EXPECT_TRUE(CompareMatrices(
      lb0, relaxation->linear_constraints()[0].evaluator()->lower_bound()));
  EXPECT_TRUE(CompareMatrices(
      ub0, relaxation->linear_constraints()[0].evaluator()->upper_bound()));

  // Second linear constraint is lb1 ≤ A1 y ≤ ub1.
  EXPECT_TRUE(CompareMatrices(
      A1, relaxation->linear_constraints()[1].evaluator()->GetDenseA()));
  EXPECT_TRUE(CompareMatrices(
      lb1, relaxation->linear_constraints()[1].evaluator()->lower_bound()));
  EXPECT_TRUE(CompareMatrices(
      ub1, relaxation->linear_constraints()[1].evaluator()->upper_bound()));

  // Third linear (in the new decision variables) constraint is 0 ≤
  // (Ay-b)(Ay-b)ᵀ, where A and b represent all of the constraints stacked.
  VectorXd b(8);  // all of the finite lower/upper bounds.
  b << -lb0[0], ub0[0], ub0[1], -lb0[2], -lb1[0], ub1[0], -lb1[1], ub1[1];
  MatrixXd A(8, 2);
  A << -A0.row(0), A0.row(0), A0.row(1), -A0.row(2), -A1_reordered.row(0),
      A1_reordered.row(0), -A1_reordered.row(1), A1_reordered.row(1);
  EXPECT_EQ(relaxation->linear_constraints()[2].evaluator()->num_constraints(),
            b.size() * b.size());
  VectorXd value = relaxation->EvalBindingAtInitialGuess(
      relaxation->linear_constraints()[2]);
  MatrixXd expected =
      (A * y_test - b) * (A * y_test - b).transpose() - b * b.transpose();
  EXPECT_TRUE(CompareMatrices(Eigen::Map<MatrixXd>(value.data(), 8, 8),
                              expected, 1e-12));
  value = relaxation->linear_constraints()[2].evaluator()->lower_bound();
  expected = -b * b.transpose();
  EXPECT_TRUE(CompareMatrices(Eigen::Map<MatrixXd>(value.data(), 8, 8),
                              expected, 1e-12));
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, LinearEqualityConstraint) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  MatrixXd A(3, 2);
  A << 0.5, 0.7, -0.2, 0.4, -2.3, -4.5;
  const Vector3d b(1.3, -0.24, 0.25);
  prog.AddLinearEqualityConstraint(A, b, y);
  auto relaxation = MakeSemidefiniteRelaxation(prog);

  EXPECT_EQ(relaxation->num_vars(), 6);
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(relaxation->linear_equality_constraints().size(), 3);

  const Vector2d y_test(1.3, 0.24);
  SetRelaxationInitialGuess(y_test, relaxation.get());

  for (int i = 0; i < 2; ++i) {
    // Linear constraints are (Ay - b)*y_i = 0.
    MatrixXd expected = (A * y_test - b) * y_test[i];
    VectorXd value = relaxation->EvalBindingAtInitialGuess(
        relaxation->linear_equality_constraints()[i]);
    EXPECT_TRUE(CompareMatrices(value, expected, 1e-12));
  }
  // Last constraint is (Ay-b)=0.
  MatrixXd expected = A * y_test - b;
  VectorXd value = relaxation->EvalBindingAtInitialGuess(
      relaxation->linear_equality_constraints()[2]);
  EXPECT_TRUE(CompareMatrices(value, expected, 1e-12));
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, QuadraticConstraint) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  Matrix2d Q;
  Q << 1, 2, 3, 4;
  const Vector2d b(0.2, 0.4);
  const double lb = -.4, ub = 0.5;
  prog.AddQuadraticConstraint(Q, b, lb, ub, y);
  auto relaxation = MakeSemidefiniteRelaxation(prog);

  EXPECT_EQ(relaxation->num_vars(), 6);
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(relaxation->linear_constraints().size(), 1);

  const Vector2d y_test(1.3, 0.24);
  SetRelaxationInitialGuess(y_test, relaxation.get());
  EXPECT_NEAR(
      relaxation->EvalBindingAtInitialGuess(
          relaxation->linear_constraints()[0])[0],
      (0.5 * y_test.transpose() * Q * y_test + b.transpose() * y_test)[0],
      1e-12);

  EXPECT_EQ(relaxation->linear_constraints()[0].evaluator()->lower_bound()[0],
            lb);
  EXPECT_EQ(relaxation->linear_constraints()[0].evaluator()->upper_bound()[0],
            ub);
}

// This test checks that repeated variables in a quadratic constraint are
// handled correctly.
GTEST_TEST(MakeSemidefiniteRelaxationTest, QuadraticConstraint2) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<1>("y");
  prog.AddQuadraticConstraint(Eigen::Matrix2d::Ones(), Eigen::Vector2d::Zero(),
                              0, 1, Vector2<Variable>(y(0), y(0)));
  auto relaxation = MakeSemidefiniteRelaxation(prog);

  EXPECT_EQ(relaxation->num_vars(), 3);
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(relaxation->linear_constraints().size(), 1);

  const Vector1d y_test(1.3);
  SetRelaxationInitialGuess(y_test, relaxation.get());
  EXPECT_NEAR(relaxation->EvalBindingAtInitialGuess(
                  relaxation->linear_constraints()[0])[0],
              2 * y_test(0) * y_test(0), 1e-12);
  EXPECT_EQ(relaxation->linear_constraints()[0].evaluator()->lower_bound()[0],
            0.0);
  EXPECT_EQ(relaxation->linear_constraints()[0].evaluator()->upper_bound()[0],
            1.0);
}

}  // namespace internal
}  // namespace solvers
}  // namespace drake
