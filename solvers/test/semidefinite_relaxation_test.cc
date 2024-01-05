#include "drake/solvers/semidefinite_relaxation.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/test/mathematical_program_test_util.h"

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
using test::IsVectorOfBindingEqual;
const double kInf = std::numeric_limits<double>::infinity();

namespace {
void SetRelaxationInitialGuess(const Eigen::Ref<const VectorXd>& y_expected,
                               MathematicalProgram* relaxation) {
  const int N = y_expected.size() + 1;
  MatrixX<Variable> X = Eigen::Map<const MatrixX<Variable>>(
      relaxation->positive_semidefinite_constraints()[0].variables().data(), N,
      N);
  VectorXd x_expected(N);
  x_expected << y_expected, 1;
  const MatrixXd X_expected = x_expected * x_expected.transpose();
  relaxation->SetInitialGuess(X, X_expected);
}

// Check that two MathematicalPrograms are exactly the same by checking whether
// they have the same decision variables, indeterminates, and same set of
// constraints. If exclude_linear_constraints is true, then the linear
// inequality and equality constraints will not be tested. This option is useful
// when testing the variable groups argument in MakeSemidefiniteProgram.
// TODO(Alexandre.Amice) move this to mathematical_program_test_util.
void ProgramsEqual(const MathematicalProgram& prog1,
                   const MathematicalProgram& prog2,
                   bool exclude_linear_constraints = false) {
  // Check that the programs have the same decision variables and
  // indeterminates.
  EXPECT_EQ(prog1.decision_variables().size(),
            prog2.decision_variables().size());
  for (const auto& [var_id, _] : prog1.decision_variable_index()) {
    EXPECT_GE(prog2.decision_variable_index().count(var_id), 0);
  }
  EXPECT_EQ(prog1.indeterminates().size(), prog2.indeterminates().size());
  for (const auto& [var_id, _] : prog1.indeterminates_index()) {
    EXPECT_GE(prog2.indeterminates_index().count(var_id), 0);
  }

  // Check that the programs have the same costs.
  IsVectorOfBindingEqual(prog1.generic_costs(), prog2.generic_costs());
  IsVectorOfBindingEqual(prog1.quadratic_costs(), prog2.quadratic_costs());
  IsVectorOfBindingEqual(prog1.linear_costs(), prog2.linear_costs());
  IsVectorOfBindingEqual(prog1.l2norm_costs(), prog2.l2norm_costs());

  // Check that the programs have the same constraints.
  IsVectorOfBindingEqual(prog1.generic_constraints(),
                         prog2.generic_constraints());
  if (!exclude_linear_constraints) {
    IsVectorOfBindingEqual(prog1.linear_constraints(),
                           prog2.linear_constraints());
    IsVectorOfBindingEqual(prog1.linear_equality_constraints(),
                           prog2.linear_equality_constraints());
  }

  IsVectorOfBindingEqual(prog1.bounding_box_constraints(),
                         prog2.bounding_box_constraints());
  IsVectorOfBindingEqual(prog1.quadratic_constraints(),
                         prog2.quadratic_constraints());
  IsVectorOfBindingEqual(prog1.lorentz_cone_constraints(),
                         prog2.lorentz_cone_constraints());
  IsVectorOfBindingEqual(prog1.rotated_lorentz_cone_constraints(),
                         prog2.rotated_lorentz_cone_constraints());
  IsVectorOfBindingEqual(prog1.positive_semidefinite_constraints(),
                         prog2.positive_semidefinite_constraints());
  IsVectorOfBindingEqual(prog1.linear_matrix_inequality_constraints(),
                         prog2.linear_matrix_inequality_constraints());
  IsVectorOfBindingEqual(prog1.exponential_cone_constraints(),
                         prog2.exponential_cone_constraints());
  IsVectorOfBindingEqual(prog1.linear_complementarity_constraints(),
                         prog2.linear_complementarity_constraints());
}
}  // namespace

GTEST_TEST(MakeSemidefiniteRelaxationTest, NoCostsNorConstraints) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  const auto relaxation = MakeSemidefiniteRelaxation(prog, std::nullopt);
  const auto relaxation_empty_vector =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>());
  const auto relaxation_with_groups =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>{Variables(y)});
  // All relaxations should be exactly equal.
  ProgramsEqual(*relaxation, *relaxation_empty_vector);
  ProgramsEqual(*relaxation, *relaxation_with_groups);

  // X is 3x3 symmetric.
  EXPECT_EQ(relaxation->num_vars(), 6);
  // X ≽ 0.
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  // X(-1,-1) = 1.
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, UnsupportedCost) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  prog.AddCost(sin(y[0]));
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, std::nullopt),
      ".*GenericCost was declared but is not supported.");
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>()),
      ".*GenericCost was declared but is not supported.");
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>{Variables(y)}),
      ".*GenericCost was declared but is not supported.");
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, UnsupportedConstraint) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  prog.AddConstraint(sin(y[0]) >= 0.2);
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, std::nullopt),
      ".*GenericConstraint was declared but is not supported.");
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>()),
      ".*GenericConstraint was declared but is not supported.");
  DRAKE_EXPECT_THROWS_MESSAGE(
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>{Variables(y)}),
      ".*GenericConstraint was declared but is not supported.");
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, LinearCost) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  const Vector2d a(0.5, 0.7);
  const double b = 1.3;
  prog.AddLinearCost(a, b, y);
  auto relaxation = MakeSemidefiniteRelaxation(prog, std::nullopt);
  const auto relaxation_empty_vector =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>());
  const auto relaxation_with_groups =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>{Variables(y)});
  // All relaxations should be exactly equal.
  ProgramsEqual(*relaxation, *relaxation_empty_vector);
  ProgramsEqual(*relaxation, *relaxation_with_groups);

  EXPECT_EQ(relaxation->num_vars(), 6);
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(relaxation->linear_costs().size(), 1);

  const Vector2d y_test(1.3, 0.24);
  SetRelaxationInitialGuess(y_test, relaxation.get());
  EXPECT_NEAR(
      relaxation->EvalBindingAtInitialGuess(relaxation->linear_costs()[0])[0],
      a.transpose() * y_test + b, 1e-12);

  // Confirm that the decision variables of prog are also decision variables of
  // the relaxation.
  std::vector<int> indices = relaxation->FindDecisionVariableIndices(y);
  std::vector<int> indices_empty_vector =
      relaxation_empty_vector->FindDecisionVariableIndices(y);
  std::vector<int> indices_with_groups =
      relaxation_with_groups->FindDecisionVariableIndices(y);
  EXPECT_EQ(indices.size(), 2);
  EXPECT_EQ(indices_empty_vector.size(), 2);
  EXPECT_EQ(indices_with_groups.size(), 2);
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, QuadraticCost) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  const Vector2d yd(0.5, 0.7);
  prog.AddQuadraticErrorCost(Matrix2d::Identity(), yd, y);
  const auto relaxation = MakeSemidefiniteRelaxation(prog, std::nullopt);
  const auto relaxation_empty_vector =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>());
  const auto relaxation_with_groups =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>{Variables(y)});
  // All relaxations should be exactly equal.
  ProgramsEqual(*relaxation, *relaxation_empty_vector);
  ProgramsEqual(*relaxation, *relaxation_with_groups);

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

  VectorXd ub(N_VARS);
  ub << kInf, 2.3;

  prog.AddBoundingBoxConstraint(lb, ub, y);

  const auto relaxation = MakeSemidefiniteRelaxation(prog, std::nullopt);
  const auto relaxation_empty_vector =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>());
  const auto relaxation_with_groups =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>{Variables(y)});
  // All relaxations should be exactly equal.
  ProgramsEqual(*relaxation, *relaxation_empty_vector);
  ProgramsEqual(*relaxation, *relaxation_with_groups);

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
  MatrixXd expected_mat =
      (A * y_test - b) * (A * y_test - b).transpose() - b * b.transpose();
  VectorXd expected = math::ToLowerTriangularColumnsFromMatrix(expected_mat);
  EXPECT_TRUE(CompareMatrices(value, expected, 1e-12));
  value = linear_constraint.evaluator()->lower_bound();
  expected = math::ToLowerTriangularColumnsFromMatrix(-b * b.transpose());
  EXPECT_TRUE(CompareMatrices(value, expected, 1e-12));
}

class LinearConstraintTest : public ::testing::Test {
 public:
  LinearConstraintTest()
      : prog(),
        x(prog.NewContinuousVariables<3>("x")),
        Ax(3, 3),
        lbx(3),
        ubx(3),
        y(prog.NewContinuousVariables<2>("y")),
        Ay(4, 2),
        lby(4),
        uby(4),
        Ay_permuted(4, 2),
        z(prog.NewContinuousVariables<4>("z")),
        Az(1, 4),
        lbz(1),
        ubz(1),
        xy(x.size() + y.size()),
        yz(y.size() + z.size()),
        xz(x.size() + z.size()),
        xyz(x.size() + y.size() + z.size()) {
    // clang-format off
    Ax << 0.5, 0.7, -0.2,
          0.4, -2.3, -4.5,
          1.7, 1.3, -0.75;
    // clang-format on
    lbx << 1.3, -kInf, 0.25;
    ubx << 5.6, 0.1, kInf;
    prog.AddLinearConstraint(Ax, lbx, ubx, x);

    // clang-format off
    Ay <<  0.2,   1.2,
           0.24, -0.1,
          -0.35,  2.1,
           1.1,   0.7;
    // clang-format on
    Ay_permuted.col(0) = Ay.col(1);
    Ay_permuted.col(1) = Ay.col(0);
    lby << -0.74, -0.3, -0.5, 8.9;
    uby << -0.75, 0.9, 0.1, kInf;
    prog.AddLinearConstraint(Ay, lby, uby, Vector2<Variable>(y[1], y[0]));
    Az << 0.5, 0.7, -0.2, 0.4;
    lbz << -kInf;
    ubz << 2.1;
    prog.AddLinearConstraint(Az, lbz, ubz, z);

    xy << x, y;
    yz << y, z;
    xz << x, z;
    xyz << x, y, z;

    x_vars = Variables(x);
    y_vars = Variables(y);
    z_vars = Variables(z);
    xy_vars = Variables(xy);
    yz_vars = Variables(yz);
    xz_vars = Variables(xz);
    xyz_vars = Variables(xyz);
  }

  // Test the parts of the relaxation that will be the same regardless of the
  // value of the variables group argument. Also check that we have the proper
  // amount of linear constraints which should be the original number of linear
  // constraints + the number of expected product constraints.
  void TestCommonPartsOfRelaxation(const MathematicalProgram& relaxation,
                                   int expected_number_of_product_constraints) {
    // The semidefinite relaxation matrix variable should have 3+2+4+1 = 10
    // rows. This corresponds to 10*11/2 = 55 variables.
    EXPECT_EQ(relaxation.num_vars(), 55);
    EXPECT_EQ(relaxation.positive_semidefinite_constraints().size(), 1);
    EXPECT_EQ(relaxation.bounding_box_constraints().size(), 1);

    EXPECT_EQ(relaxation.linear_constraints().size(),
              prog.linear_constraints().size() +
                  expected_number_of_product_constraints);

    // The first linear constraint is lbx ≤ Ax*x ≤ ubx.
    EXPECT_TRUE(CompareMatrices(
        Ax, relaxation.linear_constraints()[0].evaluator()->GetDenseA()));
    EXPECT_TRUE(CompareMatrices(
        lbx, relaxation.linear_constraints()[0].evaluator()->lower_bound()));
    EXPECT_TRUE(CompareMatrices(
        ubx, relaxation.linear_constraints()[0].evaluator()->upper_bound()));
    EXPECT_EQ(Variables(relaxation.linear_constraints()[0].variables()),
              x_vars);

    // The second linear constraint is lby ≤ Ay*y ≤ uby.
    EXPECT_TRUE(CompareMatrices(
        Ay, relaxation.linear_constraints()[1].evaluator()->GetDenseA()));
    EXPECT_TRUE(CompareMatrices(
        lby, relaxation.linear_constraints()[1].evaluator()->lower_bound()));
    EXPECT_TRUE(CompareMatrices(
        uby, relaxation.linear_constraints()[1].evaluator()->upper_bound()));
    EXPECT_EQ(Variables(relaxation.linear_constraints()[1].variables()),
              y_vars);

    // The third linear constraint is lbz ≤ Az*z ≤ ubz.
    EXPECT_TRUE(CompareMatrices(
        Az, relaxation.linear_constraints()[2].evaluator()->GetDenseA()));
    EXPECT_TRUE(CompareMatrices(
        lbz, relaxation.linear_constraints()[2].evaluator()->lower_bound()));
    EXPECT_TRUE(CompareMatrices(
        ubz, relaxation.linear_constraints()[2].evaluator()->upper_bound()));
    EXPECT_EQ(Variables(relaxation.linear_constraints()[2].variables()),
              z_vars);
  }

 protected:
  MathematicalProgram prog;
  VectorXDecisionVariable x;
  MatrixXd Ax;
  VectorXd lbx;
  VectorXd ubx;

  VectorXDecisionVariable y;
  MatrixXd Ay;
  VectorXd lby;
  VectorXd uby;
  MatrixXd Ay_permuted;

  VectorXDecisionVariable z;
  MatrixXd Az;
  VectorXd lbz;
  VectorXd ubz;

  VectorXDecisionVariable xy;
  VectorXDecisionVariable yz;
  VectorXDecisionVariable xz;
  VectorXDecisionVariable xyz;

  Variables x_vars;
  Variables y_vars;
  Variables z_vars;
  Variables xy_vars;
  Variables yz_vars;
  Variables xz_vars;
  Variables xyz_vars;
};

TEST_F(LinearConstraintTest, RelaxationOnlyChangesLinearConstraints) {
  // Check that all the relaxations are the same except for the linear
  // constraints.
  auto relaxation = MakeSemidefiniteRelaxation(prog, std::nullopt);
  auto relaxation_empty_vector =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>());
  auto relaxation_xy =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>({xy_vars}));
  auto relaxation_xz =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>({xz_vars}));
  auto relaxation_xyz =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>({xyz_vars}));

  auto relaxation_xy_yz = MakeSemidefiniteRelaxation(
      prog, std::vector<Variables>({xy_vars, yz_vars}));

  ProgramsEqual(*relaxation, *relaxation_empty_vector, true);
  ProgramsEqual(*relaxation, *relaxation_xy, true);
  ProgramsEqual(*relaxation, *relaxation_xz, true);
  ProgramsEqual(*relaxation, *relaxation_xyz, true);
  ProgramsEqual(*relaxation, *relaxation_xy_yz, true);
}

TEST_F(LinearConstraintTest, NoneMultiplied) {
  auto relaxation = MakeSemidefiniteRelaxation(prog, std::vector<Variables>());
  TestCommonPartsOfRelaxation(*relaxation, 0 /* no extra products */);
}

TEST_F(LinearConstraintTest, AllMultiplied) {
  auto relaxation = MakeSemidefiniteRelaxation(prog, std::nullopt);
  // 1 big linear constraint encoding the product of all constraints.
  TestCommonPartsOfRelaxation(*relaxation, 1);

  const Vector3d x_test{1.3, 0.24, -0.11};
  const Vector2d y_test{0.2, 1.2};
  VectorXd z_test(4);
  z_test << 0.5, 0.7, -0.2, 0.4;
  VectorXd initial_guess(3 + 2 + 4);
  initial_guess << x_test, y_test, z_test;

  SetRelaxationInitialGuess(initial_guess, relaxation.get());

  // The fourth linear (in the new decision variables) constraint is 0 ≤
  // (Av-b)(Av-b)ᵀ, where A and b represent all the constraints representing
  // finite bounds and v = [x,y,z].
  VectorXd b(12);  // all the finite lower/upper bounds.
  b << -lbx[0], ubx[0], ubx[1], -lbx[2], -lby[0], uby[0], -lby[1], uby[1],
      -lby[2], uby[2], -lby[3], ubz[0];
  MatrixXd A = Eigen::MatrixXd::Zero(12, 9);
  A.block<1, 3>(0, 0) = -Ax.row(0);
  A.block<1, 3>(1, 0) = Ax.row(0);
  A.block<1, 3>(2, 0) = Ax.row(1);
  A.block<1, 3>(3, 0) = -Ax.row(2);
  A.block<1, 2>(4, 3) = -Ay_permuted.row(0);
  A.block<1, 2>(5, 3) = Ay_permuted.row(0);
  A.block<1, 2>(6, 3) = -Ay_permuted.row(1);
  A.block<1, 2>(7, 3) = Ay_permuted.row(1);
  A.block<1, 2>(8, 3) = -Ay_permuted.row(2);
  A.block<1, 2>(9, 3) = Ay_permuted.row(2);
  A.block<1, 2>(10, 3) = -Ay_permuted.row(3);
  A.block<1, 4>(11, 5) = Az.row(0);

  int expected_size = (b.size() * (b.size() + 1)) / 2;
  EXPECT_EQ(relaxation->linear_constraints()[3].evaluator()->num_constraints(),
            expected_size);

  VectorXd value = relaxation->EvalBindingAtInitialGuess(
      relaxation->linear_constraints()[3]);
  MatrixXd expected_mat =
      (A * initial_guess - b) * (A * initial_guess - b).transpose() -
      b * b.transpose();
  VectorXd expected = math::ToLowerTriangularColumnsFromMatrix(expected_mat);
  EXPECT_TRUE(CompareMatrices(value, expected, 1e-12));
}

TEST_F(LinearConstraintTest, XZMultiplied) {
  // A non-contiguous block of variables get multiplied
  auto relaxation =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>({xz_vars}));
  // 1 linear constraint encoding the product of the x and z constraints.
  TestCommonPartsOfRelaxation(*relaxation, 1);

  const Vector3d x_test{1.3, 0.24, -0.11};
  const Vector2d y_test{0.2, 1.2};
  VectorXd z_test(4);
  z_test << 0.5, 0.7, -0.2, 0.4;
  VectorXd initial_guess(3 + 2 + 4);
  initial_guess << x_test, y_test, z_test;

  SetRelaxationInitialGuess(initial_guess, relaxation.get());

  // The fourth linear (in the new decision variables) constraint is 0 ≤
  // (Av-b)(Av-b)ᵀ, where A and b represent all the constraints representing
  // finite bounds on x and z and v = [x,z].
  VectorXd b(5);  // all the finite lower/upper bounds.
  b << -lbx[0], ubx[0], ubx[1], -lbx[2], ubz[0];
  MatrixXd A = Eigen::MatrixXd::Zero(5, 7);
  A.block<1, 3>(0, 0) = -Ax.row(0);
  A.block<1, 3>(1, 0) = Ax.row(0);
  A.block<1, 3>(2, 0) = Ax.row(1);
  A.block<1, 3>(3, 0) = -Ax.row(2);
  A.block<1, 4>(4, 3) = Az.row(0);

  int expected_size = (b.size() * (b.size() + 1)) / 2;
  EXPECT_EQ(relaxation->linear_constraints()[3].evaluator()->num_constraints(),
            expected_size);

  VectorXd value = relaxation->EvalBindingAtInitialGuess(
      relaxation->linear_constraints()[3]);
  VectorXd xz_initial(7);
  xz_initial << x_test, z_test;
  MatrixXd expected_mat =
      (A * xz_initial - b) * (A * xz_initial - b).transpose() -
      b * b.transpose();
  VectorXd expected = math::ToLowerTriangularColumnsFromMatrix(expected_mat);
  EXPECT_TRUE(CompareMatrices(value, expected, 1e-12));
}

TEST_F(LinearConstraintTest, XY_and_YZ_Multiplied) {
  // Two sets of linear constraints get multiplied.
  auto relaxation = MakeSemidefiniteRelaxation(
      prog, std::vector<Variables>({xy_vars, yz_vars}));
  // 1 linear constraint encoding the product of the Ax and Ay constraints, and
  // 1 linear constraint encoding the product of the Ay and Az constraints.
  TestCommonPartsOfRelaxation(*relaxation, 2);

  const Vector3d x_test{1.3, 0.24, -0.11};
  const Vector2d y_test{0.2, 1.2};
  VectorXd z_test(4);
  z_test << 0.5, 0.7, -0.2, 0.4;
  VectorXd initial_guess(3 + 2 + 4);
  initial_guess << x_test, y_test, z_test;

  SetRelaxationInitialGuess(initial_guess, relaxation.get());

  // The fourth linear (in the new decision variables) constraint is 0 ≤
  // (Av-b)(Av-b)ᵀ, where A and b represent all the constraints representing
  // finite bounds on x and y and v = [x,y].
  VectorXd b_xy(11);  // all the finite lower/upper bounds.
  b_xy << -lbx[0], ubx[0], ubx[1], -lbx[2], -lby[0], uby[0], -lby[1], uby[1],
      -lby[2], uby[2], -lby[3];
  MatrixXd A_xy = Eigen::MatrixXd::Zero(11, 5);
  A_xy.block<1, 3>(0, 0) = -Ax.row(0);
  A_xy.block<1, 3>(1, 0) = Ax.row(0);
  A_xy.block<1, 3>(2, 0) = Ax.row(1);
  A_xy.block<1, 3>(3, 0) = -Ax.row(2);
  A_xy.block<1, 2>(4, 3) = -Ay_permuted.row(0);
  A_xy.block<1, 2>(5, 3) = Ay_permuted.row(0);
  A_xy.block<1, 2>(6, 3) = -Ay_permuted.row(1);
  A_xy.block<1, 2>(7, 3) = Ay_permuted.row(1);
  A_xy.block<1, 2>(8, 3) = -Ay_permuted.row(2);
  A_xy.block<1, 2>(9, 3) = Ay_permuted.row(2);
  A_xy.block<1, 2>(10, 3) = -Ay_permuted.row(3);

  int expected_size = (b_xy.size() * (b_xy.size() + 1)) / 2;
  EXPECT_EQ(relaxation->linear_constraints()[3].evaluator()->num_constraints(),
            expected_size);

  VectorXd value = relaxation->EvalBindingAtInitialGuess(
      relaxation->linear_constraints()[3]);
  VectorXd xy_initial(5);
  xy_initial << x_test, y_test;
  MatrixXd expected_mat =
      (A_xy * xy_initial - b_xy) * (A_xy * xy_initial - b_xy).transpose() -
      b_xy * b_xy.transpose();
  VectorXd expected = math::ToLowerTriangularColumnsFromMatrix(expected_mat);
  EXPECT_TRUE(CompareMatrices(value, expected, 1e-12));

  // The fifth linear (in the new decision variables) constraint is 0 ≤
  // (Av-b)(Av-b)ᵀ, where A and b represent all the constraints representing
  // finite bounds on y and z and v = [y,z].
  VectorXd b_yz(8);  // all the finite lower/upper bounds.
  b_yz << -lby[0], uby[0], -lby[1], uby[1], -lby[2], uby[2], -lby[3], ubz[0];
  MatrixXd A_yz = Eigen::MatrixXd::Zero(8, 6);
  A_yz.block<1, 2>(0, 0) = -Ay_permuted.row(0);
  A_yz.block<1, 2>(1, 0) = Ay_permuted.row(0);
  A_yz.block<1, 2>(2, 0) = -Ay_permuted.row(1);
  A_yz.block<1, 2>(3, 0) = Ay_permuted.row(1);
  A_yz.block<1, 2>(4, 0) = -Ay_permuted.row(2);
  A_yz.block<1, 2>(5, 0) = Ay_permuted.row(2);
  A_yz.block<1, 2>(6, 0) = -Ay_permuted.row(3);
  A_yz.block<1, 4>(7, 2) = Az.row(0);

  expected_size = (b_yz.size() * (b_yz.size() + 1)) / 2;
  EXPECT_EQ(relaxation->linear_constraints()[4].evaluator()->num_constraints(),
            expected_size);

  value = relaxation->EvalBindingAtInitialGuess(
      relaxation->linear_constraints()[4]);
  VectorXd yz_initial(6);
  yz_initial << y_test, z_test;
  expected_mat =
      (A_yz * yz_initial - b_yz) * (A_yz * yz_initial - b_yz).transpose() -
      b_yz * b_yz.transpose();
  expected = math::ToLowerTriangularColumnsFromMatrix(expected_mat);
  EXPECT_TRUE(CompareMatrices(value, expected, 1e-12));
}

TEST_F(LinearConstraintTest, XY_and_YZ_subsets_Multiplied) {
  // Adds a linear constraint on a subset of the x variable.
  Eigen::MatrixXd Ax_small{2, 1};
  Ax_small << -3, 7;
  Eigen::Vector2d bx_small_lower{-1, -kInf};
  Eigen::Vector2d bx_small_upper{kInf, -1};
  Eigen::Vector<Variable, 1> x1{x(1)};
  prog.AddLinearConstraint(Ax_small, bx_small_lower, bx_small_upper, x1);

  // This variable group is a superset of the variables in the bx_small_lower ≤
  // Ax_small * x(1) ≤ bx_small_upper constraint and the lby ≤ Ay*y ≤ uby
  // constraint. Therefore, these two constraints will get multiplied by
  // themselves and each other, but the lbx ≤ Ax*x ≤ ubx constraint will not get
  // multiplied.
  Variables xy_subset{x(1), x(2), y(0), y(1)};
  // This group of variables is not a superset of any variables in any
  // constraints and so does nothing.
  Variables yz_subset{y(0), z(0), z(3)};
  auto relaxation = MakeSemidefiniteRelaxation(
      prog, std::vector<Variables>({xy_subset, yz_subset}));

  //  1 additional linear constraint due to the product of the constraints
  //  bx_small_lower ≤ Ax_small * x(1) ≤ bx_small_upper constraint and lby ≤
  //  Ay*y ≤ uby.
  TestCommonPartsOfRelaxation(*relaxation, 1);

  const Vector3d x_test{1.3, 0.24, -0.11};
  const Vector2d y_test{0.2, 1.2};
  VectorXd z_test(4);
  z_test << 0.5, 0.7, -0.2, 0.4;
  VectorXd initial_guess(3 + 2 + 4);
  initial_guess << x_test, y_test, z_test;

  SetRelaxationInitialGuess(initial_guess, relaxation.get());

  // The fourth linear constraint is bx_small_lower ≤ Ax_small*x(1) ≤
  // bx_small_upper.
  EXPECT_TRUE(CompareMatrices(
      Ax_small, relaxation->linear_constraints()[3].evaluator()->GetDenseA()));
  EXPECT_TRUE(CompareMatrices(
      bx_small_lower,
      relaxation->linear_constraints()[3].evaluator()->lower_bound()));
  EXPECT_TRUE(CompareMatrices(
      bx_small_upper,
      relaxation->linear_constraints()[3].evaluator()->upper_bound()));
  EXPECT_TRUE(
      relaxation->linear_constraints()[3].variables()(0).equal_to(x1(0)));

  // The fifth linear (in the new decision variables) constraint is 0 ≤
  // (Av-b)(Av-b)ᵀ, where A and b represent all the constraints representing
  // finite bounds on the variable group {x(1), x(2), y(0), y(1)} and
  // v = [x(1), x(2), y(0), y(1)].
  VectorXd b(9);  // all the finite lower/upper bounds.
  b << -lby[0], uby[0], -lby[1], uby[1], -lby[2], uby[2], -lby[3],
      -bx_small_lower[0], bx_small_upper[1];
  MatrixXd A = Eigen::MatrixXd::Zero(11, 4);
  A.block<1, 2>(0, 2) = -Ay_permuted.row(0);
  A.block<1, 2>(1, 2) = Ay_permuted.row(0);
  A.block<1, 2>(2, 2) = -Ay_permuted.row(1);
  A.block<1, 2>(3, 2) = Ay_permuted.row(1);
  A.block<1, 2>(4, 2) = -Ay_permuted.row(2);
  A.block<1, 2>(5, 2) = Ay_permuted.row(2);
  A.block<1, 2>(6, 2) = -Ay_permuted.row(3);
  A.block<1, 1>(7, 0) = -Ax_small.row(0);
  A.block<1, 1>(8, 0) = Ax_small.row(1);
  int expected_size = (b.size() * (b.size() + 1)) / 2;
  EXPECT_EQ(relaxation->linear_constraints()[4].evaluator()->num_constraints(),
            expected_size);

  VectorXd value = relaxation->EvalBindingAtInitialGuess(
      relaxation->linear_constraints()[4]);
  VectorXd subset_initial(4);
  subset_initial << x_test(1), x_test(2), y_test(0), y_test(1);
  MatrixXd expected_mat =
      (A * subset_initial - b) * (A * subset_initial - b).transpose() -
      b * b.transpose();
  VectorXd expected = math::ToLowerTriangularColumnsFromMatrix(expected_mat);
  EXPECT_TRUE(CompareMatrices(value, expected, 1e-12));
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, LinearEqualityConstraint) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  MatrixXd A(3, 2);
  A << 0.5, 0.7, -0.2, 0.4, -2.3, -4.5;
  const Vector3d b(1.3, -0.24, 0.25);
  prog.AddLinearEqualityConstraint(A, b, y);
  auto relaxation = MakeSemidefiniteRelaxation(prog, std::nullopt);

  EXPECT_EQ(relaxation->num_vars(), 6);
  EXPECT_EQ(relaxation->positive_semidefinite_constraints().size(), 1);
  EXPECT_EQ(relaxation->bounding_box_constraints().size(), 1);
  EXPECT_EQ(relaxation->linear_equality_constraints().size(), 2);

  const Vector2d y_test(1.3, 0.24);
  SetRelaxationInitialGuess(y_test, relaxation.get());

  for (int i = 0; i < ssize(prog.linear_equality_constraints()); ++i) {
    const auto cur_constraints =
        relaxation->linear_equality_constraints().at(i);
    EXPECT_TRUE(CompareMatrices(A, cur_constraints.evaluator()->GetDenseA()));
    EXPECT_TRUE(CompareMatrices(b, cur_constraints.evaluator()->lower_bound()));
    for (int col = 0; col < A.cols(); ++col) {
      EXPECT_TRUE(cur_constraints.variables()(col).equal_to(
          prog.linear_equality_constraints().at(i).variables()(col)));
    }
    MatrixXd expected = A * y_test;
    VectorXd value = relaxation->EvalBindingAtInitialGuess(
        relaxation->linear_equality_constraints()[i]);
    EXPECT_TRUE(CompareMatrices(value, expected, 1e-12));
  }
  // The linear constraint is (Ay-b)yᵀ = 0
  MatrixXd expected = (A * y_test - b) * y_test.transpose();
  VectorXd value = relaxation->EvalBindingAtInitialGuess(
      relaxation->linear_equality_constraints()[1]);
  EXPECT_TRUE(CompareMatrices(
      value, Eigen::Map<VectorXd>(expected.data(), expected.size()), 1e-12));
}

GTEST_TEST(MakeSemidefiniteRelaxationTest, QuadraticConstraint) {
  MathematicalProgram prog;
  const auto y = prog.NewContinuousVariables<2>("y");
  Matrix2d Q;
  Q << 1, 2, 3, 4;
  const Vector2d b(0.2, 0.4);
  const double lb = -.4, ub = 0.5;
  prog.AddQuadraticConstraint(Q, b, lb, ub, y);
  const auto relaxation = MakeSemidefiniteRelaxation(prog, std::nullopt);
  const auto relaxation_empty_vector =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>());
  const auto relaxation_with_groups =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>{Variables(y)});
  // All relaxations should be exactly equal.
  ProgramsEqual(*relaxation, *relaxation_empty_vector);
  ProgramsEqual(*relaxation, *relaxation_with_groups);

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
  const auto relaxation = MakeSemidefiniteRelaxation(prog, std::nullopt);
  const auto relaxation_empty_vector =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>());
  const auto relaxation_with_groups =
      MakeSemidefiniteRelaxation(prog, std::vector<Variables>{Variables(y)});
  // All relaxations should be exactly equal.
  ProgramsEqual(*relaxation, *relaxation_empty_vector);
  ProgramsEqual(*relaxation, *relaxation_with_groups);

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
