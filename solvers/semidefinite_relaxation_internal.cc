#include "drake/solvers/semidefinite_relaxation_internal.h"

#include <iostream>

#include "drake/common/fmt_eigen.h"
#include "drake/math/matrix_util.h"

namespace drake {
namespace solvers {
namespace internal {
using Eigen::SparseMatrix;
using Eigen::Triplet;

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

// Eigen::SparseMatrix<double>
// ComputeTensorProductOfSymmetricMatrixToRealVecOperators(
//    const Eigen::SparseMatrix<double>& A,
//    const Eigen::SparseMatrix<double>& B) {
//  auto r_choose_2 = [](const int r) {
//    return r * (r - 1) / 2;
//  };
//  auto compute_inv_triangular_number = [](const int Tr) {
//    return static_cast<int>((-1 + sqrt(1 + 8 * Tr)) / 2);
//  };
//  const int result_symmetric_space_cols =
//      compute_inv_triangular_number(A.cols()) *
//      compute_inv_triangular_number(B.cols());
//  const int result_cols = r_choose_2(result_symmetric_space_cols + 1);
//  const int result_rows = A.rows() * B.rows();
//
//  Eigen::SparseMatrix<double> C(result_rows, result_cols);
//  std::vector<Eigen::Triplet<double>> C_triplets;
//
//    std::cout << "result_rows: " << result_rows << std::endl;
//    std::cout << "result_cols: " << result_cols << std::endl;
//
//  // TODO(Alexandre.Amice) make this way more efficient.
//  for (int i = 0; i < A.cols(); ++i) {
//    Eigen::SparseVector<double> ei(A.cols());
//    ei.coeffRef(i) = 1;
//    for (int j = 0; j < B.cols(); ++j) {
//      Eigen::SparseVector<double> ej(B.cols());
//      ej.coeffRef(j) = 1;
//      SparseMatrix<double> temp = SparseKroneckerProduct(A * ei, B * ej);
//      for (int k = 0; k < temp.outerSize(); ++k) {
//        for (SparseMatrix<double>::InnerIterator it(temp, k); it; ++it) {
//          int row = it.row();
//          int col = i * B.cols() + j;
////              math::SymmetricMatrixIndexToLowerTriangularLinearIndex(
////              i, j, A.cols() * B.cols());
//          double val = it.value();
//          std::cout << fmt::format("it.row(), it.col(), it.value(): {}, {},
//          {}",
//                                   row, col, val)
//                    << std::endl;
//          C_triplets.emplace_back(row, col, val);
//        }
//      }
//    }
//  }
//  C.setFromTriplets(C_triplets.begin(), C_triplets.end());
//  return C;
//}

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

Eigen::SparseMatrix<double> GetSkewAdjointForLowerTri(const int r) {
  DRAKE_DEMAND(r > 0);
  const int map_num_rows = (r * (r - 1)) / 2;
  const int map_num_cols = ((r + 1) * r) / 2;
  std::vector<Triplet<double>> map_triplets;
  map_triplets.reserve(map_num_rows);
  int i = 0;
  int j = 0;
  for (int row = 0; row < r; ++row) {
    for (int col = row; col < r; ++col) {
      if (row != col) {
        map_triplets.emplace_back(i, j, -2);
        ++i;
      }
      ++j;
    }
  }
  Eigen::SparseMatrix<double> map(map_num_rows, map_num_cols);
  map.setFromTriplets(map_triplets.begin(), map_triplets.end());
  return map;
}

namespace {

// Special case when min(X.rows(), X.cols()) ≤ 2.
void AddMatrixIsLorentzSeparableConstraintSimplicialCase(
    const Eigen::Ref<const Eigen::MatrixX<symbolic::Variable>>& X,
    MathematicalProgram* prog) {
  unused(X);
  unused(prog);
  throw std::logic_error(
      "The case when min(X.rows(), X.cols()) ≤ 2 is not implemented yet.");
}

}  // namespace

void AddMatrixIsLorentzSeparableConstraint(
    const Eigen::Ref<const Eigen::MatrixX<symbolic::Variable>>& X,
    MathematicalProgram* prog) {
  if (std::min(X.rows(), X.cols()) <= 2) {
    AddMatrixIsLorentzSeparableConstraintSimplicialCase(X, prog);
    return;
  } else {
    const int m = X.rows();
    const int n = X.cols();
    // The lower triagular part of Y ∈ S⁽ⁿ⁻¹⁾ ⊗ S⁽ᵐ⁻¹⁾
    auto y = prog->NewContinuousVariables((n * (n - 1) * m * (m - 1)) / 4, "y");
    Eigen::MatrixX<symbolic::Variable> Y =
        ToSymmetricMatrixFromTensorVector(y, n - 1, m - 1);
    prog->AddPositiveSemidefiniteConstraint(Y);

    const SparseMatrix<double> W_adj_n_Kron_W_adj_m =
        SparseKroneckerProduct(GetWAdjForTril(n), GetWAdjForTril(m));

    const Eigen::VectorX<symbolic::Variable> x =
        Eigen::Map<const Eigen::VectorX<symbolic::Variable>>(X.data(),
                                                             X.size());

    // TODO(Alexandre.Amice) make sure these stay as sparse
    prog->AddLinearEqualityConstraint(W_adj_n_Kron_W_adj_m * y - x,
                                      Eigen::VectorXd::Zero(x.size()));
  }
}

}  // namespace internal
}  // namespace solvers
}  // namespace drake