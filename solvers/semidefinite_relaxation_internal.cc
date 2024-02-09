#include "drake/solvers/semidefinite_relaxation_internal.h"

#include <algorithm>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/common/symbolic/decompose.h"
#include "drake/math/matrix_util.h"

namespace drake {
namespace solvers {
namespace internal {
using Eigen::SparseMatrix;
using Eigen::Triplet;
using symbolic::Expression;
using symbolic::Variable;
using symbolic::Variables;

namespace {
Variables GetVariablesInMatrixExpression(
    const Eigen::Ref<const MatrixX<Expression>>& X) {
  Variables vars;
  for (int i = 0; i < X.rows(); ++i) {
    for (int j = 0; j < X.cols(); ++j) {
      vars.insert(X(i, j).GetVariables());
    }
  }
  return vars;
}
}  // namespace

Eigen::SparseMatrix<double> SparseKroneckerProduct(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::SparseMatrix<double>& B) {
  Eigen::SparseMatrix<double> C(A.rows() * B.rows(), A.cols() * B.cols());
  std::vector<Eigen::Triplet<double>> C_triplets;
  C_triplets.reserve(A.nonZeros() * B.nonZeros());
  C.reserve(A.nonZeros() * B.nonZeros());
  for (int iA = 0; iA < A.outerSize(); ++iA) {
    for (SparseMatrix<double>::InnerIterator itA(A, iA); itA; ++itA) {
      for (int iB = 0; iB < B.outerSize(); ++iB) {
        for (SparseMatrix<double>::InnerIterator itB(B, iB); itB; ++itB) {
          C_triplets.emplace_back(itA.row() * B.rows() + itB.row(),
                                  itA.col() * B.cols() + itB.col(),
                                  itA.value() * itB.value());
        }
      }
    }
  }
  C.setFromTriplets(C_triplets.begin(), C_triplets.end());
  return C;
}

SparseMatrix<double> GetWAdjForTril(const int r) {
  DRAKE_DEMAND(r > 0);
  // Y is a symmetric matrix of size (r-1) hence we have (r choose 2) lower
  // triangular entries.
  const int Y_tril_size = (r * (r - 1)) / 2;

  std::vector<Triplet<double>> W_adj_triplets;
  // The map operates on the diagonal twice, and then on one of the columns
  // without the first element once.
  W_adj_triplets.reserve(2 * (r - 1) + (r - 2));

  int idx = 0;
  for (int i = 0; idx < Y_tril_size; ++i) {
    W_adj_triplets.emplace_back(0, idx, 1);
    W_adj_triplets.emplace_back(1, idx, idx > 0 ? -1 : 1);
    idx += (r - 1) - i;
  }

  for (int i = 2; i < r; ++i) {
    W_adj_triplets.emplace_back(i, i - 1, 2);
  }
  SparseMatrix<double> W_adj(r, Y_tril_size);
  W_adj.setFromTriplets(W_adj_triplets.begin(), W_adj_triplets.end());
  return W_adj;
}

namespace {
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, Expression> ||
                                      std::is_same_v<T, Variable>>>
void DoAddMatrixIsLorentzByPositiveOrthantSeparableConstraint(
    const Eigen::Ref<const MatrixX<T>>& X, MathematicalProgram* prog) {
  for (int i = 0; i < X.cols(); ++i) {
    VectorX<T> x = X.col(i);
    // TODO(Alexandre.Amice) figure out why we need to make this temporary copy
    // rather than being able to directly call
    // prog->AddLorentzConeConstraint(X.col(i)).
    prog->AddLorentzConeConstraint(x);
  }
}

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, Expression> ||
                                      std::is_same_v<T, Variable>>>
void DoAddMatrixIsPositiveOrthantByLorentzSeparableConstraint(
    const Eigen::Ref<const MatrixX<T>>& X, MathematicalProgram* prog) {
  for (int i = 0; i < X.rows(); ++i) {
    VectorX<T> x = X.row(i);
    // TODO(Alexandre.Amice) figure out why we need to make this temporary copy
    // rather than being able to directly call
    // prog->AddLorentzConeConstraint(X.row(i)).
    prog->AddLorentzConeConstraint(x);
  }
}
}  //  namespace

void AddMatrixIsLorentzByPositiveOrthantSeparableConstraint(
    const Eigen::Ref<const MatrixX<Variable>>& X, MathematicalProgram* prog) {
  DoAddMatrixIsLorentzByPositiveOrthantSeparableConstraint(X, prog);
}

void AddMatrixIsLorentzByPositiveOrthantSeparableConstraint(
    const Eigen::Ref<const MatrixX<Expression>>& X, MathematicalProgram* prog) {
  DoAddMatrixIsLorentzByPositiveOrthantSeparableConstraint(X, prog);
}

void AddMatrixIsPositiveOrthantByLorentzSeparableConstraint(
    const Eigen::Ref<const MatrixX<Variable>>& X, MathematicalProgram* prog) {
  DoAddMatrixIsPositiveOrthantByLorentzSeparableConstraint(X, prog);
}

void AddMatrixIsPositiveOrthantByLorentzSeparableConstraint(
    const Eigen::Ref<const MatrixX<Expression>>& X, MathematicalProgram* prog) {
  DoAddMatrixIsPositiveOrthantByLorentzSeparableConstraint(X, prog);
}

namespace {

// Special case when min(X.rows(), X.cols()) ≤ 2. In this case, the Lorentz cone
// is just the positive orthant in two dimensions, and therefore this is just
// the product of the positive orthant with the lorentz cone.
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, Expression> ||
                                      std::is_same_v<T, Variable>>>
void DoAddMatrixIsLorentzByLorentzSeparableConstraintSimplicialCase(
    const Eigen::Ref<const MatrixX<T>>& X, MathematicalProgram* prog) {
  // Note that if x is in the 2d Lorentz cone is equivalent to x[0] ≥ 0, and
  // x[0] - x[1] ≥ 0 and so we enforce this as a linear constraint.
  DRAKE_DEMAND(X.rows() <= 2 || X.cols() <= 2);
  Eigen::MatrixX<Expression> X_expr = X.template cast<Expression>();
  Eigen::Matrix2d A{{1, 0}, {1, -1}};
  if (X.rows() == 2) {
    X_expr = A * X_expr;
  }
  if (X.cols() == 2) {
    X_expr = X_expr * A;
  }

  if (X.rows() <= 2 && X.cols() <= 2) {
    prog->AddLinearConstraint(X_expr, Eigen::Matrix2d::Zero(),
                              Eigen::Matrix2d::Zero());
  } else if (X.rows() > 2) {
    AddMatrixIsLorentzByPositiveOrthantSeparableConstraint(X, prog);
  } else if (X.cols() > 2) {
    AddMatrixIsPositiveOrthantByLorentzSeparableConstraint(X, prog);
  } else {
    DRAKE_UNREACHABLE();
  }
}

void DoAddMatrixIsLorentzByLorentzSeparableConstraint(
    const Eigen::Ref<const VectorX<Variable>>& x, const SparseMatrix<double>& A,
    const int m, const int n, MathematicalProgram* prog) {
  DRAKE_DEMAND(A.rows() == n * m);
  DRAKE_DEMAND(A.cols() == x.rows());
  if (std::min(m, n) <= 2) {
    DoAddMatrixIsLorentzByLorentzSeparableConstraintSimplicialCase<Variable>(
        Eigen::Map<const MatrixX<Variable>>(x.data(), n, m), prog);
  } else {
    // The lower triagular part of Y ∈ S⁽ⁿ⁻¹⁾ ⊗ S⁽ᵐ⁻¹⁾
    auto y = prog->NewContinuousVariables((n * (n - 1) * m * (m - 1)) / 4, "y");
    MatrixX<Variable> Y = ToSymmetricMatrixFromTensorVector(y, n - 1, m - 1);
    prog->AddPositiveSemidefiniteConstraint(Y);

    const SparseMatrix<double> W_adj_n = GetWAdjForTril(n);
    const SparseMatrix<double> W_adj_m = GetWAdjForTril(m);
    // [W_adj_n ⊗ W_adj_m; -A]
    SparseMatrix<double> CoefficientMat(
        W_adj_m.rows() * W_adj_n.rows(),
        W_adj_m.cols() * W_adj_n.cols() + A.cols());
    std::vector<Triplet<double>> CoefficientMat_triplets;
    CoefficientMat_triplets.reserve(W_adj_n.nonZeros() * W_adj_m.nonZeros() +
                                    A.nonZeros());
    // Set the left columns of CoefficientMat to W_adj_n ⊗ W_adj_m.
    for (int idx_n = 0; idx_n < W_adj_n.outerSize(); ++idx_n) {
      for (SparseMatrix<double>::InnerIterator it_n(W_adj_n, idx_n); it_n;
           ++it_n) {
        for (int idx_m = 0; idx_m < W_adj_m.outerSize(); ++idx_m) {
          for (SparseMatrix<double>::InnerIterator it_m(W_adj_m, idx_m); it_m;
               ++it_m) {
            CoefficientMat_triplets.emplace_back(
                it_n.row() * W_adj_m.rows() + it_m.row(),
                it_n.col() * W_adj_m.cols() + it_m.col(),
                it_n.value() * it_m.value());
          }
        }
      }
    }
    int col = W_adj_m.cols() * W_adj_n.cols();
    // Set the right columns of CoefficientMat to -A.
    for (int i = 0; i < A.outerSize(); ++i) {
      for (SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
        CoefficientMat_triplets.emplace_back(it.row(), it.col() + col,
                                             -it.value());
      }
    }
    CoefficientMat.setFromTriplets(CoefficientMat_triplets.begin(),
                                   CoefficientMat_triplets.end());
    VectorX<Variable> yx(y.size() + x.size());
    yx << y, x;
    prog->AddLinearEqualityConstraint(CoefficientMat,
                                      Eigen::VectorXd::Zero(x.size()), yx);
  }
}
}  // namespace

void AddMatrixIsLorentzByLorentzSeparableConstraint(
    const Eigen::Ref<const MatrixX<Variable>>& X, MathematicalProgram* prog) {
  if (std::min(X.rows(), X.cols()) <= 2) {
    DoAddMatrixIsLorentzByLorentzSeparableConstraintSimplicialCase(X, prog);
  } else {
    const int m = X.rows();
    const int n = X.cols();
    SparseMatrix<double> I(n * m, n * m);
    I.setIdentity();
    const Eigen::VectorX<Variable> x =
        Eigen::Map<const Eigen::VectorX<Variable>>(X.data(), X.size());
    DoAddMatrixIsLorentzByLorentzSeparableConstraint(x, I, m, n, prog);
  }
}

void AddMatrixIsLorentzByLorentzSeparableConstraint(
    const Eigen::Ref<const MatrixX<Expression>>& X, MathematicalProgram* prog) {
  if (std::min(X.rows(), X.cols()) <= 2) {
    DoAddMatrixIsLorentzByLorentzSeparableConstraintSimplicialCase(X, prog);
  } else {
    const int m = X.rows();
    const int n = X.cols();
    const Variables vars{GetVariablesInMatrixExpression(X)};
    VectorX<Variable> vector_vars{vars.size()};
    int i = 0;
    for (const auto& v : vars) {
      vector_vars(i++) = v;
    }

    Eigen::MatrixXd A;
    symbolic::DecomposeLinearExpressions(
        Eigen::Map<const VectorX<Expression>>(X.data(), X.size()), vector_vars,
        &A);
    DoAddMatrixIsLorentzByLorentzSeparableConstraint(
        vector_vars, A.sparseView(), m, n, prog);
  }
}

namespace {

SparseMatrix<double> GetRotatedLorentzToLorentzMap(const int n) {
  // TODO(Alexandre.Amice) do this
  SparseMatrix<double> I(n * n, n * n);
  I.setIdentity();
  return I;
}
}  // namespace

void AddMatrixIsRotatedLorentzByLorentzSeparableConstraint(
    const Eigen::Ref<const MatrixX<Variable>>& X, MathematicalProgram* prog) {
  DRAKE_THROW_UNLESS(X.rows() >= 3);
  SparseMatrix<double> R = GetRotatedLorentzToLorentzMap(X.rows());
  SparseMatrix<double> I(R.cols(), R.cols());
  I.setIdentity();
  const Eigen::Ref<const VectorX<Variable>> x{
      Eigen::Map<const VectorX<Variable>>(X.data(), X.size())};
  SparseMatrix<double> R_for_flat_x = SparseKroneckerProduct(I, R);
  DoAddMatrixIsLorentzByLorentzSeparableConstraint(x, R_for_flat_x, X.rows(),
                                                   X.cols(), prog);
}

void AddMatrixIsRotatedLorentzByLorentzSeparableConstraint(
    const Eigen::Ref<const MatrixX<Expression>>& X, MathematicalProgram* prog) {
  DRAKE_THROW_UNLESS(X.rows() >= 3);
  SparseMatrix<double> R = GetRotatedLorentzToLorentzMap(X.rows());
  AddMatrixIsLorentzByLorentzSeparableConstraint(R * X, prog);
}

void AddMatrixIsLorentzByRotatedLorentzSeparableConstraint(
    const Eigen::Ref<const MatrixX<Variable>>& X, MathematicalProgram* prog) {
  AddMatrixIsRotatedLorentzByLorentzSeparableConstraint(X.transpose(), prog);
}

void AddMatrixIsLorentzByRotatedLorentzSeparableConstraint(
    const Eigen::Ref<const MatrixX<Expression>>& X, MathematicalProgram* prog) {
  AddMatrixIsRotatedLorentzByLorentzSeparableConstraint(X.transpose(), prog);
}

void AddMatrixIsRotatedLorentzByRotatedLorentzSeparableConstraint(
    const Eigen::Ref<const MatrixX<Variable>>& X, MathematicalProgram* prog) {
  DRAKE_THROW_UNLESS(X.rows() >= 3 && X.cols() >= 3);
  SparseMatrix<double> R_rows = GetRotatedLorentzToLorentzMap(X.rows());
  SparseMatrix<double> R_cols = GetRotatedLorentzToLorentzMap(X.cols());
  const Eigen::Ref<const VectorX<Variable>> x{
      Eigen::Map<const VectorX<Variable>>(X.data(), X.size())};
  SparseMatrix<double> R_for_flat_x =
      SparseKroneckerProduct(R_cols.transpose(), R_rows);
  DoAddMatrixIsLorentzByLorentzSeparableConstraint(x, R_for_flat_x, X.rows(),
                                                   X.cols(), prog);
}

void AddMatrixIsRotatedLorentzByRotatedLorentzSeparableConstraint(
    const Eigen::Ref<const MatrixX<Expression>>& X, MathematicalProgram* prog) {
  DRAKE_THROW_UNLESS(X.rows() >= 3 && X.cols() >= 3);
  DRAKE_THROW_UNLESS(X.rows() >= 3);
  const SparseMatrix<double> R_rows = GetRotatedLorentzToLorentzMap(X.rows());
  const SparseMatrix<double> R_cols = GetRotatedLorentzToLorentzMap(X.cols());
  AddMatrixIsLorentzByLorentzSeparableConstraint(R_rows * X * R_cols, prog);
}

}  // namespace internal
}  // namespace solvers
}  // namespace drake
