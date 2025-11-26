#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/program_attribute.h"

namespace drake {
namespace solvers {

struct ConicStandardFormOptions {
  /// If true, keeps quadratic costs as-is instead of converting them to
  /// epigraph form.
  bool keep_quadratic_costs{false};

  /// If true, keeps rotated Lorentz cone constraints as-is instead of
  /// converting them to standard Lorentz cone constraints.
  // bool keep_rotated_lorentz_cones{false};

  /// If true, then off diagonal elements of the psd cone are scaled by sqrt(2)
  /// so that the vectorized inner product is preserved.
  // bool preserve_psd_inner_product_vectorization{true};

  /**
   * When converting linear constraints of the form lb ≤ Ax ≤ ub, we guarantee
   * that the bounds are stored back to back, i.e. [A]x+[lb] ∈ R₊ [A]x+[ub] ∈ R₊
   * This setting merely controls the attributes_to_start_end_pairs. If this
   * setting is true, then ProgramAttribute kBoundingBox constraint will be
   * present, and the start index will be the beginning of the lower bound, and
   * the end index will be the end of the lower bound, with the break between
   * the two bounds happening at (start + end)/2. If this setting is false,
   * ProgramAttribute kLinearConstarint will be used instead, and both
   * constraints will be as separate positive orthant constraints.
   */
  bool use_bounding_box_for_indexing{false};
};

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
  explicit ConicStandardForm(const MathematicalProgram& prog);

  const Eigen::SparseMatrix<double>& P() const { return P_; }
  const Eigen::SparseVector<double>& c() const { return c_; }
  double d() const { return d_; }
  const Eigen::SparseMatrix<double>& A() const { return A_; }
  const Eigen::SparseVector<double>& b() const { return b_; }
  const VectorX<symbolic::Variable>& x() const { return x_; }

  const Eigen::VectorXd& bb_lower_bounds() const { return bb_lower_bounds_; }
  const Eigen::VectorXd& bb_upper_bounds() const { return bb_upper_bounds_; }

  const std::unordered_map<ProgramAttribute, std::vector<std::pair<int, int>>>&
  attributes_to_start_end_pairs() const {
    return attributes_to_start_end_pairs_;
  }

  /** Return a MathematicalProgram represented by this standard conic form data.
   */
  std::unique_ptr<MathematicalProgram> MakeProgram() const;

 private:
  /** P a sparse matrix encoding the dense part of the cost*/
  Eigen::SparseMatrix<double> P_;
  /** c a sparse vector encoding the cost. */
  Eigen::SparseVector<double> c_;
  /** d is the constant term in the cost. */
  double d_{0};
  /** A is the sparse matrix encoding the conic constraint. */
  Eigen::SparseMatrix<double> A_;
  /** b is a sparse matrix encoding the conic constraint. */
  Eigen::SparseVector<double> b_;

  /** The decision variables x of the program.*/
  VectorX<symbolic::Variable> x_;

  Eigen::VectorXd bb_lower_bounds_;
  Eigen::VectorXd bb_upper_bounds_;

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
