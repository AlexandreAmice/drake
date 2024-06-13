#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/program_attribute.h"

namespace drake {
namespace solvers {

/** Stores the information required to represent a convex program  (specficially
 * an LP, QP, SOCP, or SDP), in standard primal conic
 * form:
 * min 〈 c, x 〉+ d subject to
 * Ax + b ∈ K
 * Where K is the product of the cones:
 * 1) Zero cone {x | x = 0 }
 * 2) Positive orthant {x | x ≥ 0 }
 * 3) Second-order cone {(t, x) | |x|₂ ≤ t }
 * 4) Positive semidefinite cone { X |  min(eig(X)) ≥ 0, X = Xᵀ }g
 */
class ConicStandardForm {
 public:
  ConicStandardForm(const MathematicalProgram& prog);

  const Eigen::SparseVector<double>& c() const { return c_; }
  double d() const { return d_; }
  const Eigen::SparseMatrix<double>& A() const { return A_; }
  const Eigen::SparseVector<double>& b() const { return b_; }

  const std::unordered_map<ProgramAttribute, std::vector<std::pair<int, int>>>&
  attributes_to_start_end_pairs() const {
    return attributes_to_start_end_pairs_;
  }

  /** Return a MathematicalProgram represented by this standard conic form data.
   */
  std::unique_ptr<MathematicalProgram> MakeProgram() const;

 private:
  /** c a sparse vector encoding the cost. */
  Eigen::SparseVector<double> c_;
  /** d is the constant term in the cost. */
  double d_{0};
  /** A is the sparse matrix encoding the conic constraint. */
  Eigen::SparseMatrix<double> A_;
  /** b is a sparse matrix encoding the conic constraint. */
  Eigen::SparseVector<double> b_;

  /**
   * attributes_to_start_end_pairs[type][i] is a pair of indices
   * [start, end) encoding that (Ax+b)[start:end, :] is in the cone
   * corresponding to the attribute type.
   * Note that if (Ax+b)[start:end, :] is in the psd cone, then it is a vector
   * corresponding to the lower-triangular part of a PSD matrices with the
   * off-diagonal entires scaled by √2 to preserve inner products. For example
   * if (Ax+b)[start:end, :] = [y0, y1, y2, y3, y4, y5], then
   * Y = [y0,    y1/√2,  y2/√2]
   *     [y1/√2,   y3,  y4/√2]
   *     [y2/√2, y4/√2, y5]]
   * */
  std::unordered_map<ProgramAttribute, std::vector<std::pair<int, int>>>
      attributes_to_start_end_pairs_{};
};

/**
 * Given a convex program (specficially an LP, QP, SOCP, or SDP), parse it into
 * an equivalent program in standard primal conic form:
 * min        〈 c, x 〉
 * subject to  Ax + b ∈ K
 * Where K is the product of the cones:
 * 1) Zero cone {x | x = 0 }
 * 2) Positive orthant {x | x ≥ 0 }
 * 3) Second-order cone {(t, x) | |x|₂ ≤ t }
 * 4) Positive semidefinite cone { X | min(eig(X)) ≥ 0, X = Xᵀ }
 *
 * Currently, we only support programs with linear costs.
 */
ConicStandardForm ParseToConicStandardForm(const MathematicalProgram& prog);

}  // namespace solvers
}  // namespace drake
