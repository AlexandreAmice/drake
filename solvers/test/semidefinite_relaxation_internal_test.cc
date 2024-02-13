#include "drake/solvers/semidefinite_relaxation_internal.h"

#include <gtest/gtest.h>

#include "drake/common/ssize.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace solvers {
namespace internal {
using symbolic::Expression;
using symbolic::Variable;
using symbolic::Variables;

GTEST_TEST(MakeSemidefiniteRelaxationInternalTest, TestSparseKron) {
  Eigen::MatrixXd A(3, 3);
  // clang-format off
  A <<  1.77, -4.38, -3.63,
       -2.03, -0.57,  4.56,
       -1.66,  2.15, 10.02;
  // clang-format on
  Eigen::MatrixXd B(2, 2);
  // clang-format off
  B << 7.16, 1.8,
       2.39, 5.77;
  // clang-format on

  Eigen::SparseMatrix<double> C(7, 13);
  std::vector<Eigen::Triplet<double>> C_triplets;
  C_triplets.emplace_back(0, 9, 1.77);
  C_triplets.emplace_back(1, 5, -0.57);
  C_triplets.emplace_back(2, 3, 4.56);
  C_triplets.emplace_back(3, 12, -1.66);
  C_triplets.emplace_back(4, 7, 2.15);
  C_triplets.emplace_back(5, 6, 10.02);
  C_triplets.emplace_back(0, 1, -0.45);
  C.setFromTriplets(C_triplets.begin(), C_triplets.end());

  auto TestKron = [](const Eigen::SparseMatrix<double>& A_test,
                     const Eigen::SparseMatrix<double>& B_test) {
    // AXB = (B.T ⊗ A)vec(X) so we compute this as a numerical sanity check.
    const Eigen::SparseMatrix<double> M1 = B_test.transpose();
    const Eigen::SparseMatrix<double> M2 = A_test;

    Eigen::SparseMatrix<double> kron = SparseKroneckerProduct(M1, M2);
    EXPECT_EQ(kron.rows(), M1.rows() * M2.rows());
    EXPECT_EQ(kron.cols(), M1.cols() * M2.cols());
    EXPECT_EQ(kron.nonZeros(), M1.nonZeros() * M2.nonZeros());

    Eigen::MatrixXd testMatrix(A_test.cols(), B_test.rows());
    for (int i = 0; i < testMatrix.rows(); ++i) {
      for (int j = 0; j < testMatrix.cols(); ++j) {
        // put arbitrary values in testMatrix
        testMatrix(i, j) = 2 * (i + j) / (i + j + 1);
      }
    }
    Eigen::MatrixXd AXB = A_test * testMatrix * B_test;
    Eigen::VectorXd kron_vec = kron * Eigen::Map<const Eigen::VectorXd>(
                                          testMatrix.data(), testMatrix.size());
    EXPECT_TRUE(CompareMatrices(
        Eigen::Map<const Eigen::VectorXd>(AXB.data(), AXB.size()), kron_vec,
        1e-10));
  };

  TestKron(A.sparseView(), B.sparseView());
  TestKron(B.sparseView(), A.sparseView());
  TestKron(C, A.sparseView());
  TestKron(B.sparseView(), C);
  TestKron(C, B.sparseView());
}

GTEST_TEST(MakeSemidefiniteRelaxationInternalTest,
           TestToSymmetricMatrixFromTensorVector) {
  // Check the 2x2 ⊗  3x3 case.
  int num_elts =
      3 * 6;  // There are 3 elements in 2x2 basis and 6 in the 3x3 basis.
  VectorX<symbolic::Variable> y(num_elts);
  for (int i = 0; i < num_elts; ++i) {
    y(i) = symbolic::Variable("y_" + std::to_string(i));
  }
  Eigen::MatrixX<symbolic::Variable> Y =
      ToSymmetricMatrixFromTensorVector(y, 2, 3);
  Eigen::MatrixX<symbolic::Variable> Y_expected(6, 6);
  // clang-format off
  Y_expected << y(0), y(1),  y(2),  y(6),  y(7),  y(8),
                y(1), y(3),  y(4),  y(7),  y(9),  y(10),
                y(2), y(4),  y(5),  y(8),  y(10), y(11),
                y(6), y(7),  y(8),  y(12), y(13), y(14),
                y(7), y(9),  y(10), y(13), y(15), y(16),
                y(8), y(10), y(11), y(14), y(16), y(17);
  // clang-format on
  EXPECT_EQ(Y.rows(), 6);
  EXPECT_EQ(Y.cols(), 6);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_TRUE(Y_expected(i, j).equal_to(Y(i, j)));
    }
  }

  // Check the 4x4 ⊗  3x3 case.
  num_elts = 10 * 6;
  VectorX<symbolic::Variable> x(num_elts);
  for (int i = 0; i < num_elts; ++i) {
    x(i) = symbolic::Variable(fmt::format("x({}),", std::to_string(i)));
  }
  Eigen::MatrixX<symbolic::Variable> X =
      ToSymmetricMatrixFromTensorVector(x, 4, 3);
  Eigen::MatrixX<symbolic::Variable> X_expected(12, 12);
  // The following was checked by hand. It may seem big, but it is necessary to
  // check something this large.
  // clang-format off
  // NOLINT
  X_expected <<  x(0),  x(1),  x(2),  x(6),  x(7),  x(8), x(12), x(13), x(14), x(18), x(19), x(20),// NOLINT
                 x(1),  x(3),  x(4),  x(7),  x(9), x(10), x(13), x(15), x(16), x(19), x(21), x(22),// NOLINT
                 x(2),  x(4),  x(5),  x(8), x(10), x(11), x(14), x(16), x(17), x(20), x(22), x(23),// NOLINT
                 x(6),  x(7),  x(8), x(24), x(25), x(26), x(30), x(31), x(32), x(36), x(37), x(38),// NOLINT
                 x(7),  x(9), x(10), x(25), x(27), x(28), x(31), x(33), x(34), x(37), x(39), x(40),// NOLINT
                 x(8), x(10), x(11), x(26), x(28), x(29), x(32), x(34), x(35), x(38), x(40), x(41),// NOLINT
                x(12), x(13), x(14), x(30), x(31), x(32), x(42), x(43), x(44), x(48), x(49), x(50),// NOLINT
                x(13), x(15), x(16), x(31), x(33), x(34), x(43), x(45), x(46), x(49), x(51), x(52),// NOLINT
                x(14), x(16), x(17), x(32), x(34), x(35), x(44), x(46), x(47), x(50), x(52), x(53),// NOLINT
                x(18), x(19), x(20), x(36), x(37), x(38), x(48), x(49), x(50), x(54), x(55), x(56),// NOLINT
                x(19), x(21), x(22), x(37), x(39), x(40), x(49), x(51), x(52), x(55), x(57), x(58),// NOLINT
                x(20), x(22), x(23), x(38), x(40), x(41), x(50), x(52), x(53), x(56), x(58), x(59);// NOLINT
  // clang-format on
  EXPECT_EQ(X.rows(), 12);
  EXPECT_EQ(X.cols(), 12);
  for (int i = 0; i < 12; ++i) {
    for (int j = 0; j < 12; ++j) {
      EXPECT_TRUE(X_expected(i, j).equal_to(X(i, j)));
    }
  }
}

GTEST_TEST(MakeSemidefiniteRelaxationInternalTest, TestWAdj) {
  Eigen::MatrixXd Y(5, 5);
  // clang-format off
  Y << -3.08, -0.84,  0.32,  0.54,  0.51,
       -0.84,  2.6 ,  1.72,  0.09,  0.79,
        0.32,  1.72,  3.09,  1.19,  0.31,
        0.54,  0.09,  1.19, -0.37, -2.57,
        0.51,  0.79,  0.31, -2.57, -0.08;
  // clang-format on
  // The lower triangular part of Y has 15 entries
  Eigen::VectorXd y_tril(15);
  y_tril = math::ToLowerTriangularColumnsFromMatrix(Y);

  Eigen::SparseMatrix<double> W_adj = GetWAdjForTril(6);
  EXPECT_EQ(W_adj.rows(), 6);
  EXPECT_EQ(W_adj.cols(), y_tril.size());
  EXPECT_EQ(W_adj.nonZeros(), 14 /* 2 * (6-1) + 4 */);

  Eigen::VectorXd result(6);
  result = W_adj * y_tril;
  const double tol{1e-10};
  EXPECT_NEAR(result(0), Y.trace(), tol);
  EXPECT_NEAR(result(1), Y(0, 0) - Y.block(1, 1, 4, 4).trace(), tol);
  for (int i = 2; i < W_adj.rows(); ++i) {
    EXPECT_NEAR(result(i), 2 * Y(0, i - 1), tol);
  }
}

namespace {
// Makes a random unitary matrix of size r.
Eigen::MatrixXd GenerateRandomUnitary(const int r) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);
  Eigen::MatrixXd T = Eigen::MatrixXd::NullaryExpr(r, r, [&]() {
    return distribution(generator);
  });
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(T);
  Eigen::MatrixXd Q = qr.householderQ();
  return Q;
};

// Makes a random vector with all positive entries that is not in the Lorentz
// cone.
Eigen::VectorXd MakeRandomPositiveOrthantVector(const int r,
                                                const double scale) {
  Eigen::VectorXd ret =
      scale * (Eigen::VectorXd::Random(r) + Eigen::VectorXd::Ones(r));
  while (r > 1 && ret(0) > ret.tail(r - 1).norm()) {
    ret(0) = 1 / (0.5 * ret.tail(r - 1).norm()) *
             (Eigen::VectorXd::Random(1)(0) + 1);
  }
  return ret;
}

// Returns true if the vector x has all entries which are positive
bool CheckIsPositiveOrthantVector(const Eigen::Ref<const Eigen::VectorXd>& x) {
  return (x.array() >= 0).all();
}

// Construct a random matrix A that is guaranteed to map a positive vector in m
// dimensions to another positive vector in r dimensions. This method does not
// sample uniformly over all such maps.
Eigen::MatrixXd MakeRandomPositivePositiveOrthantMap(const int r, const int m,
                                                     const double scale) {
  Eigen::MatrixXd A(r, m);
  for (int i = 0; i < m; i++) {
    A.col(i) = MakeRandomPositiveOrthantVector(r, scale);
  }
  return A;
}

// Constructs a random vector that is in the Lorentz cone.
Eigen::VectorXd MakeRandomLorentzVector(const int r, const double scale) {
  Eigen::VectorXd ret(r);
  if (r > 1) {
    ret.tail(r - 1) = scale * Eigen::VectorXd::Random(r - 1);
    // Ensures that ret(0) ≥ norm(ret(r-1)) and that ret(0) ≥ 0.
    do {
      ret(0) = 2 * scale * (Eigen::VectorXd::Random(1)(0) + 1);
    } while (ret(0) < ret.tail(r - 1).norm() || ret(0) < 0);
  } else {
    ret(0) = scale * (Eigen::VectorXd::Random(1)(0) + 1);
  }
  return ret;
}

// Returns true if the vector x is in the lorentz cone
bool CheckIsLorentzVector(const Eigen::Ref<const Eigen::VectorXd>& x) {
  if (x.rows() == 1) {
    return x(0) >= 0;
  }
  return x(0) >= x.tail(x.rows() - 1).norm();
}

// Construct a random matrix A that is guaranteed to map a lorentz vector in m
// dimensions to another lorentz vector in r. This method does not sample
// uniformly over all such maps.
Eigen::MatrixXd MakeRandomPositiveLorentzMap(const int r, const int m,
                                             const double scale) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(r, m);
  int unitary_block_size = std::min(r - 1, m - 1);
  int num_blocks = 0;
  int cur_row = 1;
  while(unitary_block_size > 0) {
    if(cur_row + unitary_block_size - 1 < r) {
      A.block(cur_row, 1, unitary_block_size, unitary_block_size) =
          GenerateRandomUnitary(unitary_block_size);
      cur_row += unitary_block_size;
      ++num_blocks;
    }
    else {--unitary_block_size;}
  }
//  if (unitary_block_size > 0) {
//    int cur_row = 1;
//    while (cur_row + unitary_block_size - 1 < r) {
//      A.block(cur_row, 1, unitary_block_size, unitary_block_size) =
//          GenerateRandomUnitary(unitary_block_size);
//      cur_row += unitary_block_size;
//    }
//  }
  // Scale the first entry by at least the number of unitary blocks added
  A(0, 0) = num_blocks + (Eigen::VectorXd::Random(1)(0) + 1);
  return A;
}

// Eigen::VectorXd MakeRandomRotatedLorentzVector(int r, double scale) {
//  DRAKE_DEMAND(r >= 3);
//  Eigen::VectorXd ret(r);
//  ret.tail(r - 2) = scale * Eigen::VectorXd::Random(r - 2);
//  // Ensures that ret(0) * ret(1) ≥ norm(ret(r-2)) and that ret(0), ret(1) ≥
//  // 0.
//  const double shift = std::max(ret.tail(r - 2).norm() / scale, 1.0);
//  ret(0) = scale * (std::rand() + shift);
//  ret(1) = scale * (std::rand() + shift);
//  return ret;
//}

}  // namespace

class PositiveOrthantLorentzSeparabilityTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {
 public:
  PositiveOrthantLorentzSeparabilityTest() : prog_() {
    // Seed the random generator used by Eigen.
    std::srand(91);
    const double scale = 5;

    int m, n;
    std::tie(m, n) = GetParam();
    X_ = prog_.NewContinuousVariables(m, n, "X");

    xr_positive_ = MakeRandomPositiveOrthantVector(m, scale);
    xc_positive_ = MakeRandomPositiveOrthantVector(n, scale);
    yr_positive_ = MakeRandomPositiveOrthantVector(m, scale);
    yc_positive_ = MakeRandomPositiveOrthantVector(n, scale);
    wr_positive_ = MakeRandomPositiveOrthantVector(m, scale);
    wc_positive_ = MakeRandomPositiveOrthantVector(n, scale);

    xr_lorentz_ = MakeRandomLorentzVector(m, scale);
    xc_lorentz_ = MakeRandomLorentzVector(n, scale);
    yr_lorentz_ = MakeRandomLorentzVector(m, scale);
    yc_lorentz_ = MakeRandomLorentzVector(n, scale);
    wr_lorentz_ = MakeRandomLorentzVector(m, scale);
    wc_lorentz_ = MakeRandomLorentzVector(n, scale);

    zr_not_conic_ = scale * Eigen::VectorXd::Random(m);
    zc_not_conic_ = scale * Eigen::VectorXd::Random(n);
  }

  // Test the methods AddMatrixIsPositiveOrthantByLorentzSeparableConstraint by
  // checking that the matrix A * X_ * B is Positive Orthant, Lorentz separable.
  // The matrix A should be a positive orthant, positive map (i.e. all positive
  // entries) and B should be Lorentz positive maps for these tests to be
  // correct. If A and B are not passed, then they are set to the identity.
  void DoPositiveOrthantByLorentzSeparableTest(
      const std::optional<Eigen::MatrixXd>& A_opt = std::nullopt,
      const std::optional<Eigen::MatrixXd>& B_opt = std::nullopt) {
    MatrixX<Expression> Y;
    Eigen::MatrixXd A =
        A_opt.value_or(Eigen::MatrixXd::Identity(X_.rows(), X_.rows()));
    Eigen::MatrixXd B =
        B_opt.value_or(Eigen::MatrixXd::Identity(X_.cols(), X_.cols()));
    if (!A_opt.has_value() && !B_opt.has_value()) {
      Y = X_.template cast<Expression>();
      AddMatrixIsPositiveOrthantByLorentzSeparableConstraint(X_, &prog_);

    } else {
      Y = A * X_ * B;
      AddMatrixIsPositiveOrthantByLorentzSeparableConstraint(Y, &prog_);
    }
    EXPECT_EQ(ssize(prog_.lorentz_cone_constraints()), A.rows());
    EXPECT_EQ(ssize(prog_.GetAllConstraints()), A.rows());

    MakeTestVectors(A, B);
    if (!CheckIsPositiveOrthantVector(xr_positive_) ||
        !CheckIsPositiveOrthantVector(yr_positive_)) {
      throw std::logic_error("A is not a positive orthant positive map.");
    }
    if (!CheckIsLorentzVector(xc_lorentz_) ||
        !CheckIsLorentzVector(yc_lorentz_)) {
      throw std::logic_error("B is not a lorentz positive map.");
    }


    // Y is required to be equal to a simple Positive Orthant, Lorentz separable
    // tensor therefore this program should be feasible.
    Eigen::MatrixXd test_matrix = xr_positive_ * xc_lorentz_.transpose();
    DoTestCase(Y, test_matrix, true);

    // Y is required to be equal to a Positive Orthant, Lorentz separable tensor
    // therefore this program should be feasible.
    test_matrix = (2 * xr_positive_ * yc_lorentz_.transpose() +
                   3 * yr_positive_ * xc_lorentz_.transpose());
    DoTestCase(Y, test_matrix, true);

    // Y is required to be equal to something not that is not Positive Orthant,
    // Lorentz separable, therefore this should be infeasible.
    test_matrix = zr_not_conic_ * zc_not_conic_.transpose();
    DoTestCase(Y, test_matrix, false);

    // A tensor of positve orthant and non-lorentz vector.
    // Y is required to be equal to something not that is not lorentz
    // separable therefore this should be infeasible.
    test_matrix = yr_positive_ * zc_not_conic_.transpose();
    DoTestCase(Y, test_matrix, false);

    // A tensor of non-positve orthant vector and non-lorentz vector.
    // Y is required to be equal to something not that is not lorentz
    // separable therefore this should be infeasible.
    test_matrix = zr_not_conic_ * xc_lorentz_.transpose();
    DoTestCase(Y, test_matrix, false);

    // A tensor that is Positive Orthant separable, but not Positive Orthant,
    // Lorentz separable.
    test_matrix = yr_positive_ * wc_positive_.transpose();
    DoTestCase(Y, test_matrix, false);

    // A tensor that is Lorentz separable, but not Positive Orthant,
    // Lorentz separable.
    test_matrix = wr_lorentz_ * xc_lorentz_.transpose();
    DoTestCase(Y, test_matrix, false);

    // A tensor that is non-conic combination of simple Positive Orthant,
    // Lorentz separable tensors.
    test_matrix = 2 * yr_positive_ * xc_lorentz_.transpose() -
                  10 * xr_positive_ * yc_lorentz_.transpose();
    DoTestCase(Y, test_matrix, false);
  }

  // Test the methods AddMatrixIsLorentzByPositiveOrthantSeparableConstraint by
  // checking that the matrix A * X_ * B is Lorentz, Positive Orthant separable.
  // The matrix A should be a Lorentz positive maps, and B should be
  // positive-orthant positive map (i.e. all positive entries) for these tests
  // to be correct. If A and B are not passed, then they are set to the
  // identity.
  void DoLorentzByPositiveOrthantSeparableTest(
      const std::optional<Eigen::MatrixXd>& A_opt = std::nullopt,
      const std::optional<Eigen::MatrixXd>& B_opt = std::nullopt) {
    MatrixX<Expression> Y;
    Eigen::MatrixXd A =
        A_opt.value_or(Eigen::MatrixXd::Identity(X_.rows(), X_.rows()));
    Eigen::MatrixXd B =
        B_opt.value_or(Eigen::MatrixXd::Identity(X_.cols(), X_.cols()));
    if (!A_opt.has_value() && !B_opt.has_value()) {
      Y = X_.template cast<Expression>();
      AddMatrixIsLorentzByPositiveOrthantSeparableConstraint(X_, &prog_);

    } else {
      Y = A * X_ * B;
      AddMatrixIsLorentzByPositiveOrthantSeparableConstraint(Y, &prog_);
    }
    EXPECT_EQ(ssize(prog_.lorentz_cone_constraints()), B.cols());
    EXPECT_EQ(ssize(prog_.GetAllConstraints()), B.cols());

    MakeTestVectors(A, B);
    if (!CheckIsLorentzVector(xr_lorentz_) ||
        !CheckIsLorentzVector(yr_lorentz_)) {
      throw std::logic_error("A is not a lorentz positive map.");
    }
    if (!CheckIsPositiveOrthantVector(xc_positive_) ||
        !CheckIsPositiveOrthantVector(yc_positive_)) {
      throw std::logic_error("B is not a positive orthant positive map.");
    }

    // Y is required to be equal to a simple Lorentz, Positive Orthant separable
    // tensor therefore this program should be feasible.
    Eigen::MatrixXd test_matrix = xr_lorentz_ * xc_positive_.transpose();
    DoTestCase(Y, test_matrix, true);

    // Y is required to be equal to a Lorentz, Positive Orthant,  separable
    // tensor therefore this program should be feasible.
    test_matrix = (2 * xr_lorentz_ * yc_positive_.transpose() +
                   3 * yr_lorentz_ * xc_positive_.transpose());
    DoTestCase(Y, test_matrix, true);

    // Y is required to be equal to something not that is not Lorentz, Positive
    // Orthant separable, therefore this should be infeasible.
    test_matrix = zr_not_conic_ * zc_not_conic_.transpose();
    DoTestCase(Y, test_matrix, false);

    // A tensor of lorentz and non-lorentz vector.
    // Y is required to be equal to something not that is not lorentz
    // separable therefore this should be infeasible.
    test_matrix = yr_lorentz_ * zc_not_conic_.transpose();
    DoTestCase(Y, test_matrix, false);

    // A tensor of non-positive orthant vector and positive orthant vector.
    // Y is required to be equal to something not that is not lorentz
    // separable therefore this should be infeasible.
    test_matrix = zr_not_conic_ * yc_positive_.transpose();
    DoTestCase(Y, test_matrix, false);

    // A tensor that is Positive Orthant separable, but not Positive Orthant,
    // Lorentz separable.
    test_matrix = wr_positive_ * xc_positive_.transpose();
    DoTestCase(Y, test_matrix, false);

    // A tensor that is Lorentz separable, but not Positive Orthant,
    // Lorentz separable.
    test_matrix = xr_lorentz_ * wc_lorentz_.transpose();
    DoTestCase(Y, test_matrix, false);

    // A tensor that is non-conic combination of simple Lorentz, Positive
    // Orthant separable tensors.
    test_matrix = 2 * yr_lorentz_ * xc_positive_.transpose() -
                  10 * xr_lorentz_ * yc_positive_.transpose();
    DoTestCase(Y, test_matrix, false);
  }

 protected:
  MathematicalProgram prog_;
  MatrixX<Variable> X_;

  Eigen::VectorXd xr_positive_;
  Eigen::VectorXd xc_positive_;
  Eigen::VectorXd yr_positive_;
  Eigen::VectorXd yc_positive_;
  Eigen::VectorXd wr_positive_;
  Eigen::VectorXd wc_positive_;

  Eigen::VectorXd xr_lorentz_;
  Eigen::VectorXd xc_lorentz_;
  Eigen::VectorXd yr_lorentz_;
  Eigen::VectorXd yc_lorentz_;
  Eigen::VectorXd wr_lorentz_;
  Eigen::VectorXd wc_lorentz_;

  Eigen::VectorXd zr_not_conic_;
  Eigen::VectorXd zc_not_conic_;

 private:
  void MakeTestVectors(const Eigen::Ref<const Eigen::MatrixXd>& A,
                       const Eigen::Ref<const Eigen::MatrixXd>& B) {
    const double scale = 5;
    // Notice that if A is not a lorentz positive map, then xr_lorentz_ and
    // yr_lorentz_ will not necessarily be in the lorentz cone. Ditto for all
    // the other cases.
    xr_lorentz_ = A * MakeRandomLorentzVector(A.cols(), scale);
    yr_lorentz_ = A * MakeRandomLorentzVector(A.cols(), scale);
    xc_lorentz_ = B.transpose() * MakeRandomLorentzVector(B.rows(), scale);
    yc_lorentz_ = B.transpose() * MakeRandomLorentzVector(B.rows(), scale);

    // Vectors guaranteed to be in the Lorentz cone, but not necessarily in the
    // image of A (resp. B).
    wr_lorentz_ = MakeRandomLorentzVector(A.rows(), scale);
    wc_lorentz_ = MakeRandomLorentzVector(B.cols(), scale);

    // Notice that if A is not a positive orthant positive map, then
    // xr_positive_ and yr_positive_ will not necessarily be in the lorentz
    // cone. Ditto for all the other cases.
    xr_positive_ = A * MakeRandomPositiveOrthantVector(A.cols(), scale);
    yr_positive_ = A * MakeRandomPositiveOrthantVector(A.cols(), scale);
    xc_positive_ =
        B.transpose() * MakeRandomPositiveOrthantVector(B.rows(), scale);
    yc_positive_ =
        B.transpose() * MakeRandomPositiveOrthantVector(B.rows(), scale);

    // Vectors guaranteed to be in the positive orthant, but not necessarily in
    // the image of A (resp. B).
    wr_positive_ = MakeRandomPositiveOrthantVector(A.rows(), scale);
    wc_positive_ = MakeRandomPositiveOrthantVector(B.cols(), scale);

    zr_not_conic_ = Eigen::VectorXd::Random(A.rows());
    if (zr_not_conic_(0) > 0) {
      zr_not_conic_(0) *= -1;
    }
    zc_not_conic_ = Eigen::VectorXd::Random(B.cols());
    if (zc_not_conic_(0) > 0) {
      zc_not_conic_(0) *= -1;
    }
  }

  void DoTestCase(const Eigen::Ref<const MatrixX<Expression>>& Y,
                  const Eigen::Ref<const Eigen::MatrixXd>& test_mat,
                  bool expected_outcome) {
    auto constraint = prog_.AddLinearEqualityConstraint(Y == test_mat);
    auto result = Solve(prog_);
    if (result.is_success() != expected_outcome && result.is_success()) {
      std::cout << fmt::format("X=\n{}", fmt_eigen(result.GetSolution(X_)))
                << std::endl;
    }
    EXPECT_EQ(result.is_success(), expected_outcome);
    prog_.RemoveConstraint(constraint);
  }
};

TEST_P(PositiveOrthantLorentzSeparabilityTest,
       AddMatrixIsPositiveOrthantByLorentzSeparableConstraintVariable) {
  this->DoPositiveOrthantByLorentzSeparableTest();
}

TEST_P(
    PositiveOrthantLorentzSeparabilityTest,
    AddMatrixIsPositiveOrthantByLorentzSeparableConstraintExpressionKeepSize) {
  int m, n;
  std::tie(m, n) = GetParam();

  const double scale = 3;
  // This maps need to be a positive, positive orthant maps.
  Eigen::MatrixXd A = MakeRandomPositivePositiveOrthantMap(m, m, scale);
  // This maps need to be a positive loretnz maps.
  Eigen::MatrixXd B = MakeRandomPositiveLorentzMap(n, n, scale);
  this->DoPositiveOrthantByLorentzSeparableTest(A, B);
}

TEST_P(
    PositiveOrthantLorentzSeparabilityTest,
    AddMatrixIsPositiveOrthantByLorentzSeparableConstraintChangeSize) {
  int m, n;
  std::tie(m, n) = GetParam();
  const int r1 = m+2;
  const int r2 = m-1;
  const int c1 = n+3;
  const int c2 = n-2;

  const double scale = 3;
  // This maps need to be a positive, positive orthant maps.
  Eigen::MatrixXd A1 = MakeRandomPositivePositiveOrthantMap(r1, m, scale);
  Eigen::MatrixXd A2 = MakeRandomPositivePositiveOrthantMap(r2, m, scale);
  // This maps need to be a positive lorentz maps.
  Eigen::MatrixXd B1 = MakeRandomPositiveLorentzMap(n, c1, scale);
  Eigen::MatrixXd B2 = MakeRandomPositiveLorentzMap(n, c2, scale);

  std::cout << fmt::format("A1=\n{}", fmt_eigen(A1)) << std::endl;
  std::cout << fmt::format("A2=\n{}", fmt_eigen(A2)) << std::endl;
  std::cout << fmt::format("B1=\n{}", fmt_eigen(B1)) << std::endl;
  std::cout << fmt::format("B2=\n{}", fmt_eigen(B2)) << std::endl;

  this->DoPositiveOrthantByLorentzSeparableTest(A1, B1);
  this->DoPositiveOrthantByLorentzSeparableTest(A1, B2);
  this->DoPositiveOrthantByLorentzSeparableTest(A2, B1);
  this->DoPositiveOrthantByLorentzSeparableTest(A2, B2);
}

TEST_P(PositiveOrthantLorentzSeparabilityTest,
       AddMatrixIsLorentzByPositiveOrthantSeparableConstraintVariable) {
  this->DoLorentzByPositiveOrthantSeparableTest();
}

TEST_P(
    PositiveOrthantLorentzSeparabilityTest,
    AddMatrixIsLorentzByPositiveOrthantSeparableConstraintExpressionKeepSize) {
  int m, n;
  std::tie(m, n) = GetParam();

  const double scale = 3;
  // This maps need to be a positive, positive orthant maps.
  Eigen::MatrixXd A = MakeRandomPositiveLorentzMap(m, m, scale);
  // This maps need to be a positive loretnz maps.
  Eigen::MatrixXd B = MakeRandomPositivePositiveOrthantMap(n, n, scale);
  this->DoLorentzByPositiveOrthantSeparableTest(A, B);
}

INSTANTIATE_TEST_SUITE_P(test, PositiveOrthantLorentzSeparabilityTest,
                         ::testing::Values(std::pair<int, int>{4, 5},  // m ≤ n
                                           std::pair<int, int>{5, 4},  // m ≥ n
                                           std::pair<int, int>{6, 6}   // m == n
                                           // There are no special cases
                                           ));

class LorentzLorentzSeparabilityTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {
 public:
  LorentzLorentzSeparabilityTest() : prog_() {
    // Seed the random generator used by Eigen.
    std::srand(99);
    int m, n;
    std::tie(m, n) = GetParam();
    X_ = prog_.NewContinuousVariables(m, n, "X");
  }

  // Tests the function AddMatrixIsLorentzByLorentzSeparableConstraint by
  // checking whether A * X_ * B is Lorentz separable. The matrices A and B
  // should be Lorentz positive maps for these tests to be correct. If A and B
  // are not passed, then they are set to the identity.
  void DoTest(const std::optional<Eigen::MatrixXd>& A = std::nullopt,
              const std::optional<Eigen::MatrixXd>& B = std::nullopt) {
    MatrixX<Expression> Y;
    if (!A.has_value() && !B.has_value()) {
      Y = X_.template cast<Expression>();
      AddMatrixIsLorentzByLorentzSeparableConstraint(X_, &prog_);
    } else {
      Y = A.value() * X_ * B.value();
      AddMatrixIsLorentzByLorentzSeparableConstraint(Y, &prog_);
    }
    int m, n;
    std::tie(m, n) = GetParam();
    int r = Y.rows();
    int c = Y.cols();

    MakeTestVectors(A, B);

    if (r >= 3 && c >= 3) {
      EXPECT_EQ(ssize(prog_.positive_semidefinite_constraints()), 1);
      EXPECT_EQ(ssize(prog_.linear_equality_constraints()), 1);
      EXPECT_EQ(ssize(prog_.GetAllConstraints()), 2);

      const Binding<PositiveSemidefiniteConstraint> psd_constraint =
          prog_.positive_semidefinite_constraints()[0];
      EXPECT_EQ(psd_constraint.evaluator()->matrix_rows(), (r - 1) * (c - 1));
    } else if (r == 1 && c == 1) {
      EXPECT_EQ(ssize(prog_.bounding_box_constraints()), 1);
    } else if (r <= 2 && c <= 2) {
      EXPECT_EQ(ssize(prog_.linear_constraints()), 1);
      EXPECT_EQ(ssize(prog_.GetAllConstraints()), 1);
    } else if (r > 2) {
      EXPECT_EQ(ssize(prog_.lorentz_cone_constraints()), c);
      EXPECT_EQ(ssize(prog_.GetAllConstraints()), c);
    } else {
      EXPECT_EQ(ssize(prog_.lorentz_cone_constraints()), r);
      EXPECT_EQ(ssize(prog_.GetAllConstraints()), r);
      std::cout << prog_ << std::endl;
    }

    // Y is required to be equal to a simple Lorentz separable tensor
    // therefore this program should be feasible.
    DoTestCase(Y, xr_lorentz_ * xc_lorentz_.transpose(), true);
    // Y is required to be equal to a lorentz separable tensor therefore
    //  this program should be feasible.
    DoTestCase(Y,
               (2 * xr_lorentz_ * yc_lorentz_.transpose() +
                3 * yr_lorentz_ * xc_lorentz_.transpose()),
               true);

    // If r = 1 or c = 1, then zr_not_lorentz_ *
    // zc_not_lorentz_.transpose() will be generate positive number in the
    // first entry and therefore zr_not_lorentz_ * zc_not_lorentz_.transpose()
    // will be a Lorentz separable tensor. We exclude this example by changing
    // the sign. A tensor of two non-lorentz vectors.
    double coeff = ((c == 1) ? -1 : 1) * ((r == 1) ? -1 : 1) *
                   (((r == 1 && c == 1)) ? -1 : 1);
    // Y is required to be equal to something not that is not lorentz
    // separable therefore this should be infeasible.
    DoTestCase(Y, coeff * zr_not_lorentz_ * zc_not_lorentz_.transpose(), false);

    // A tensor of lorentz and non-lorentz vectors.
    // Y is required to be equal to something not that is not lorentz
    // separable therefore this should be infeasible.
    DoTestCase(Y, yr_lorentz_ * zc_not_lorentz_.transpose(), false);

    // A non-conic combination lorentz separable tensors.
    // Y is required to be equal to something not that is not lorentz
    // separable therefore this should be infeasible.
    DoTestCase(Y,
               (2 * xr_lorentz_ * yc_lorentz_.transpose() -
                10 * yr_lorentz_ * xc_lorentz_.transpose()),
               false);
  }

 protected:
  MathematicalProgram prog_;
  MatrixX<Variable> X_;

  Eigen::VectorXd xr_lorentz_;
  Eigen::VectorXd xc_lorentz_;
  Eigen::VectorXd yr_lorentz_;
  Eigen::VectorXd yc_lorentz_;

  Eigen::VectorXd zr_not_lorentz_;
  Eigen::VectorXd zc_not_lorentz_;

 private:
  void MakeTestVectors(const std::optional<Eigen::MatrixXd>& A_optional,
                       const std::optional<Eigen::MatrixXd>& B_optional) {
    Eigen::MatrixXd A =
        A_optional.value_or(Eigen::MatrixXd::Identity(X_.rows(), X_.rows()));
    Eigen::MatrixXd B =
        B_optional.value_or(Eigen::MatrixXd::Identity(X_.cols(), X_.cols()));
    const double scale = 5;
    const int r = A.rows();
    const int c = B.cols();
    xr_lorentz_ = A * MakeRandomLorentzVector(A.cols(), scale);
    xc_lorentz_ = B.transpose() * MakeRandomLorentzVector(B.rows(), scale);
    yr_lorentz_ = A * MakeRandomLorentzVector(A.cols(), scale);
    yc_lorentz_ = B.transpose() * MakeRandomLorentzVector(B.rows(), scale);

    if (r == 1) {
      // A small negative number, since the Lorentz cone in 1D is the
      // non-negative numbers.
      zr_not_lorentz_.resize(1);
      zr_not_lorentz_(0) = -0.07;
    } else {
      // A positive vector that is not in the Lorentz Cone.
      zr_not_lorentz_ = MakeRandomPositiveOrthantVector(r, scale);
      zr_not_lorentz_(0) = zr_not_lorentz_.norm() / 10;
    }
    if (c == 1) {
      // A negative number, since the Lorentz cone in 1D is the non-negative
      // numbers.
      zc_not_lorentz_.resize(1);
      zc_not_lorentz_(0) = -11;
    }
    if (c >= 2) {
      // This is not in the Lorentz cone because the first entry is negative.
      zc_not_lorentz_ = MakeRandomPositiveOrthantVector(c, scale);
      zc_not_lorentz_(0) = -10 * zc_not_lorentz_(0);
    }
  }

  void DoTestCase(const Eigen::Ref<const MatrixX<Expression>>& Y,
                  const Eigen::Ref<const Eigen::MatrixXd>& test_mat,
                  bool expected_outcome) {
    auto constraint = prog_.AddLinearEqualityConstraint(Y == test_mat);
    auto result = Solve(prog_);
    if (result.is_success() != expected_outcome) {
      std::cout << prog_ << std::endl;
    }
    EXPECT_EQ(result.is_success(), expected_outcome);
    prog_.RemoveConstraint(constraint);
  }
};

TEST_P(LorentzLorentzSeparabilityTest,
       AddMatrixIsLorentzByLorentzSeparableConstraintVariable) {
  int m, n;
  std::tie(m, n) = GetParam();
  this->DoTest();
}

TEST_P(LorentzLorentzSeparabilityTest,
       AddMatrixIsLorentzByLorentzSeparableConstraintExpressionKeepSize) {
  // We test whether A * X_ * B is Lorentz separable. For this to be true in
  // general A needs to be an m-Lorentz positive map and B needs to be an
  // n-Lorentz positive map. An easy way to generate such a map is to put a
  // unitary matrix in the (r-1) bottom right corner of the matrix as this
  // preserves the norm. This method employs this technique.
  int m, n;
  std::tie(m, n) = GetParam();
  double scale = 3;
  // These maps need to be Lorentz-positive maps.
  Eigen::MatrixXd A = MakeRandomPositiveLorentzMap(m, m, scale);
  Eigen::MatrixXd B = MakeRandomPositiveLorentzMap(n, n, scale);
  this->DoTest(A, B);
}

// TEST_P(LorentzLorentzSeparabilityTest,
//       AddMatrixIsLorentzByLorentzSeparableConstraintExpressionChangeSize) {
//  int m, n;
//  std::tie(m, n) = GetParam();
//  int i = 1 + 2 * (m - 1);  // m + 3;
//  int j = std::max(m - 1, 1);
//  int k = n + 2;
//  int l = std::max(n - 2, 1);
//  double scale = 3;
//  Eigen::MatrixXd A1 = MakeRandomPositiveLorentzMap(i, m, scale);
//  Eigen::MatrixXd B1 =
//      Eigen::MatrixXd::Identity(n, n);  // Eigen::MatrixXd::Zero(n, k);
//                                        //  B1(0, 0) = 3e4;
//  //  B1.block(1, 1, n - 1, n - 1) = GenerateRandomUnitary(n - 1);
//
//  Eigen::MatrixXd A2 = Eigen::MatrixXd::Zero(j, m);
//  A2(0, 0) = 3;
//  A2.block(1, 1, j - 1, j - 1) = GenerateRandomUnitary(j - 1);
//  //
//  Eigen::MatrixXd B2 = Eigen::MatrixXd::Zero(n, l);
//  B2(0, 0) = 3;
//  B2.block(1, 1, l - 1, l - 1) = GenerateRandomUnitary(l - 1);
//
////  std::cout << fmt::format("A1=\n{}", fmt_eigen(A1)) << std::endl;
//  //  std::cout << fmt::format("A2=\n{}", fmt_eigen(A2)) << std::endl;
////  std::cout << fmt::format("B2=\n{}", fmt_eigen(B2)) << std::endl;
//  //  std::cout << fmt::format("B2=\n{}", fmt_eigen(B2)) << std::endl;
//  //  std::cout << fmt::format("A1 * X * B1=\n{}", fmt_eigen(A1 * X_ * B1))
//  //            << std::endl;
//  if(i > 2 && j > 2 && k > 2 && l > 2) {
//    this->DoTest<Expression>(A1, B1);
//    this->DoTest<Expression>(A2, B1);
////  this->DoTest<Expression>(A1, B2);
////  this->DoTest<Expression>(A2, B2);
//  }
//}

INSTANTIATE_TEST_SUITE_P(
    test, LorentzLorentzSeparabilityTest,
    ::testing::Values(std::pair<int, int>{3, 4},  // m ≤ n
                      std::pair<int, int>{4, 3},  // m ≥ n
                      std::pair<int, int>{5, 5},  // m == n
                      std::pair<int, int>{1, 1},  // special case m = n = 1
                      std::pair<int, int>{2, 1},  // special case m = 2, n = 1
                      std::pair<int, int>{1, 2},  // special case m = 1, m = 1
                      std::pair<int, int>{2, 2},  // special case m = 2, m = 2
                      std::pair<int, int>{1, 4},  // special case m = 1, n ≥ 3
                      std::pair<int, int>{2, 5},  // special case m = 2, n ≥ 3
                      std::pair<int, int>{3, 1},  // special case m ≥ 3, n = 1
                      std::pair<int, int>{7, 2}   // special case m ≥ 3, n = 2
                      ));

}  // namespace internal
}  // namespace solvers
}  // namespace drake
