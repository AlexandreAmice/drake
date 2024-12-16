#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/program_attribute.h"

namespace drake {
namespace solvers {

/** Stores the information required to represent a convex program  (specficially
 * an LP, QP, SOCP, or SDP), in the following conic standard form (sometimes
 * called the dual standard form) form: min 〈 c, x 〉+ d subject to Ax - b ∈ K
 * Where K is the product of the cones:
 * 1) Zero cone {x | x = 0 }
 * 2) Positive orthant {x | x ≥ 0 }
 * 3) Second-order cone {(t, x) | |x|₂ ≤ t }
 * 4) Positive semidefinite cone { X |  min(eig(X)) ≥ 0, X = Xᵀ }
 */
class ConicStandardForm {
 public:
  explicit ConicStandardForm(const MathematicalProgram& prog);

  /** The linear cost of the program */
  const Eigen::SparseVector<double>& c() const { return c_; }
  /** The constant cost of the program */
  double d() const { return d_; }
  /** The A matrix in the expression  Ax - b ∈ K */
  const Eigen::SparseMatrix<double>& A() const { return A_; }
  /** The b matrix in the expression  Ax - b ∈ K */
  const Eigen::SparseVector<double>& b() const { return b_; }
  /** The x variables in the expression  Ax - b ∈ K. These are guaranteed to be
   * the same decision variables in the original program, but not necessarily in
   * the same order. */
  const VectorX<symbolic::Variable>& x() const { return x_; }

  /** A map from the supported ProgramAttributes to a list of [start, end)
   * tuples representing that the expression (Ax - b)[start:end, :] is in the
   * cone corresponding to the attribute type.
   *
   * @note The condition that s := (Ax - b)[start:end, :] ∈ PSD means that s is
   * the inner product preserving vectorization of a PSD matrix S. That is to
   * say s is a vector the size of the lower triangular part of S with the
   * entries corresponding to off-diagonal entries scaled by 1/sqrt(2). For
   * example, s = [s₀, s₁, s₂, s₃, s₄, s₅] corresponds to the matrix
   * S = [s₀,   s₁/√2,  s₂/√2]
   *     [s₁/√2, s₃,    s₄/√2]
   *     [s₂/√2  s₄/√2, s₅]
   * see https://clarabel.org/stable/examples/example_sdp/ for details.
   */
  const std::unordered_map<ProgramAttribute, std::vector<std::pair<int, int>>>&
  attributes_to_start_end_pairs() const {
    return attributes_to_start_end_pairs_;
  }

  /**
   * Return a MathematicalProgram represented by this standard conic form data.
   */
  std::unique_ptr<MathematicalProgram> MakeProgram() const;

 private:
  /** a sparse vector encoding the cost. */
  Eigen::SparseVector<double> c_;
  /** d is the constant term in the cost. */
  double d_{0};
  /** A is the sparse matrix encoding the conic constraint. */
  Eigen::SparseMatrix<double> A_;
  /** b is a sparse matrix encoding the conic constraint. */
  Eigen::SparseVector<double> b_;

  /** The decision variables x of the program.*/
  const VectorX<symbolic::Variable> x_;

  std::unordered_map<ProgramAttribute, std::vector<std::pair<int, int>>>
      attributes_to_start_end_pairs_{};
};

}  // namespace solvers
}  // namespace drake
