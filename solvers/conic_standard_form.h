#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/program_attribute.h"

namespace drake {
namespace solvers {

struct ConicStandardFormOptions {
  // Whether to parse the program with a quadratic cost.
  bool use_quadratic_cost = false;
  // Whether to vectorize matrices using the upper triangular or lower
  // triangular part.
  bool parse_psd_using_upper_triangular = false;

  /** Ensure that the cones are in the sorted order
   * 1) Zero cone {x | x = 0 }
   * 2) Positive orthant {x | x ≥ 0 }
   * 3) Second-order cone {(t, x) | |x|₂ ≤ t }
   * 4) Positive semidefinite cone { X |  min(eig(X)) ≥ 0, X = Xᵀ }
   * 5) Exponential cone { x |  x₀ ≥ x₁ exp(x₂/ x₁), x₁ > 0}
   */

  bool sort_cones = false;
};

namespace internal {

struct EqualityConstraintInfo {
  // A vector of triplets to construct the matrix A.
  std::vector<Eigen::Triplet<double>> A_triplets;
  // A vector of values to construct the vector b.
  std::vector<double> b_std;
  // The number of rows in the matrix A.
  int A_row_count{0};

  // The start indices of the dual variables for each linear equality
  // constraints.
  std::vector<int> linear_eq_dual_variable_start_indices;
};

struct BoundingBoxConstraintInfo {
  std::vector<Eigen::Triplet<double>> A_eq_triplets;
  std::vector<double> b_eq_std;
  int A_eq_row_count;

  std::vector<Eigen::Triplet<double>> A_ineq_triplets;
  std::vector<double> b_ineq_std;
  int A_ineq_row_count;

  // bounding_box_constraint_dual_indices[i][j] are the indices of
  // the dual variable for the j'th row of prog.bounding_box_constraints()[i].
  // We use -1 to indicate that it is impossible for this constraint to be
  // active (for example, another BoundingBoxConstraint imposes a tighter bound
  // on the same variable).
  std::vector<std::vector<std::pair<int, int>>>
      bounding_box_constraint_dual_indices;
};

struct LinearConstraintInfo {
  // A vector of triplets to construct the matrix A.
  std::vector<Eigen::Triplet<double>> A_triplets;
  // A vector of values to construct the vector b.
  std::vector<double> b_std;
  // The number of rows in the matrix A.
  int A_row_count{0};

  // The start indices of the dual variables for each linear constraints.
  std::vector<std::vector<std::pair<int, int>>> linear_constraint_dual_indices;

  // See ParseScalarPositiveSemidefiniteConstraints for the meaning of these
  // variables.
  std::vector<std::optional<int>> scalar_psd_dual_indices;
  std::vector<std::optional<int>> scalar_lmi_dual_indices;
};

struct SocConstraintInfo {
  // A vector of triplets to construct the matrix A.
  std::vector<Eigen::Triplet<double>> A_triplets;
  // A vector of values to construct the vector b.
  std::vector<double> b_std;
  // The number of rows in the matrix A.
  int A_row_count{0};

  // The lengths of each second order cone.
  std::vector<int> second_order_cone_lengths;

  // lorentz_cone_y_start_indices y[lorentz_cone_y_start_indices[i]:
  // lorentz_cone_y_start_indices[i] + second_order_cone_length[i]] are the dual
  // variables for prog.lorentz_cone_constraints()[i]. See
  // ParseSecondOrderConeConstraints for more details.
  std::vector<int> lorentz_cone_dual_variable_start_indices;
  // y[rotated_lorentz_cone_y_start_indices[i]:
  // rotated_lorentz_cone_y_start_indices[i] +
  // prog.rotate_lorentz_cone()[i].evaluator().A().rows] are the y variables for
  // prog.rotated_lorentz_cone_constraints()[i]. See
  // ParseSecondOrderConeConstraints for more details.
  std::vector<int> rotated_lorentz_cone_dual_variable_start_indices;

  std::vector<int> l2norm_costs_lorentz_cone_y_start_indices;
  std::vector<int> l2norm_costs_t_slack_indices;

  // See Parse2x2PositiveSemidefiniteConstraints for the meaning of these
  // variables.
  std::vector<std::optional<int>> twobytwo_psd_dual_start_indices;
  std::vector<std::optional<int>> twobytwo_lmi_dual_start_indices;
};

struct PsdConstraintInfo {
  // A vector of triplets to construct the matrix A.
  std::vector<Eigen::Triplet<double>> A_triplets;
  // A vector of values to construct the vector b.
  std::vector<double> b_std;
  // The number of rows in the matrix A.
  int A_row_count{0};

  // Whether vectorized PSD constraints correspond to the upper or lower
  // triangular part of the matrix.
  bool parse_psd_using_upper_triangular{true};

  // The size of each psd cone.
  std::vector<std::optional<int>> psd_cone_length;
  std::vector<std::optional<int>> lmi_cone_length;

  // See ParsePositiveSemidefiniteConstraints for the meaning of these
  // variables.
  std::vector<std::optional<int>> psd_y_start_indices;
  std::vector<std::optional<int>> lmi_y_start_indices;
};

struct ExponentialConeInfo {
  // A vector of triplets to construct the matrix A.
  std::vector<Eigen::Triplet<double>> A_triplets;
  // A vector of values to construct the vector b.
  std::vector<double> b_std;
  // The number of rows in the matrix A.
  int A_row_count{0};

  // The number of exponential cone constraints parsed.
  int num_exponential_cone_constraints{0};
};

struct CostAggregationInfo {
  // A vector of triplets to construct the matrix P.
  std::vector<Eigen::Triplet<double>> P_upper_triplets;
  // A vector of values to construct the vector c.
  std::vector<double> c_std;

  // The constant term in the cost.
  double d{0};
};

struct ConvexAggregationInfo {
  EqualityConstraintInfo equality_constraint_info;
  BoundingBoxConstraintInfo bounding_box_constraint_info;
  LinearConstraintInfo linear_constraint_info;
  SocConstraintInfo soc_constraint_info;
  PsdConstraintInfo psd_constraint_info;
  ExponentialConeInfo exponential_cone_info;
  CostAggregationInfo cost_info;
  // The number of variables required to parse the cost into standard form. This
  // is guaranteed to be at least as large as the number of variables in the
  // program.
  int num_x{0};
};

// struct ConstraintAggregationInfo2 {
//   // A vector of triplets to construct the matrix A.
//   std::vector<Eigen::Triplet<double>> A_triplets;
//   // A vector of values to construct the vector b.
//   std::vector<double> b_std;
//   // The number of rows in the matrix A.
//   int A_row_count{0};
//   // The total number of variables with bounding box constraints such that
//   // lb < ub.
//   int num_bounding_box_inequality_constraint_rows{0};
//   // The total number of linear equality constraints.
//   int num_linear_equality_constraint_rows{0};
//   // The total number of linear constraints.
//   int num_linear_constraint_rows{0};
//   // The lengths of each second order cone s in the Cartesian product K. See
//   //  // ParseSecondOrderConeConstraints for more details.
//   std::vector<int> second_order_cone_lengths;
//
//   // Whether vectorized PSD constraints correspond to the upper or lower
//   // triangular part of the matrix.
//   bool parse_psd_using_upper_triangular{true};
//
//   // The size of each psd cone.
//   std::vector<std::optional<int>> psd_cone_length;
//   // The size of each lmi cone.
//   std::vector<std::optional<int>> lmi_cone_length;
//
//   // The number of psd matrices parsed as linear constraints.
//   int scalar_psd_positive_cone_length{0};
//
//   // The number of vectorized PSD and LMI constraints parsed as lorentz cone
//   // constraints.
//   int num_twobytwo_psd_and_lmi_constraints{0};
//
//   // The number of exponential cone constraints parsed.
//   int num_exponential_cone_constraints{0};
//
//   // An unordered map from cone attributes to [row_start, row_end) pairs.
//   std::unordered_map<ProgramAttribute, std::vector<std::pair<int, int>>>
//       attributes_to_start_end_pairs{};
// };
//
// struct CostAggregationInfo2 {
//   // A vector of triplets to construct the matrix P.
//   std::vector<Eigen::Triplet<double>> P_upper_triplets;
//   // A vector of values to construct the vector c.
//   std::vector<double> c_std;
//
//   // The constant term in the cost.
//   double d{0};
//
//   // The number of variables required to parse the cost into standard form.
//   This
//   // is guaranteed to be at least as large as the number of variables in the
//   // program.
//   int num_x{0};
//
//   ConstraintAggregationInfo constraint_info;
// };

struct DualInfo {
  // bounding_box_constraint_dual_indices[i][j] are the indices of
  // the dual variable for the j'th row of prog.bounding_box_constraints()[i].
  // We use -1 to indicate that it is impossible for this constraint to be
  // active (for example, another BoundingBoxConstraint imposes a tighter bound
  // on the same variable).
  std::vector<std::vector<std::pair<int, int>>>
      bounding_box_constraint_dual_indices;

  // The start indices of the dual variables for each linear equality
  // constraints.
  std::vector<int> linear_eq_dual_variable_start_indices;

  // The start indices of the dual variables for each linear constraints.
  std::vector<std::vector<std::pair<int, int>>> linear_constraint_dual_indices;

  // lorentz_cone_y_start_indices y[lorentz_cone_y_start_indices[i]:
  // lorentz_cone_y_start_indices[i] + second_order_cone_length[i]] are the dual
  // variables for prog.lorentz_cone_constraints()[i]. See
  // ParseSecondOrderConeConstraints for more details.
  std::vector<int> lorentz_cone_dual_variable_start_indices;
  // y[rotated_lorentz_cone_y_start_indices[i]:
  // rotated_lorentz_cone_y_start_indices[i] +
  // prog.rotate_lorentz_cone()[i].evaluator().A().rows] are the y variables for
  // prog.rotated_lorentz_cone_constraints()[i]. See
  // ParseSecondOrderConeConstraints for more details.
  std::vector<int> rotated_lorentz_cone_dual_variable_start_indices;

  std::vector<int> l2norm_costs_lorentz_cone_y_start_indices;
  std::vector<int> l2norm_costs_t_slack_indices;

  // See ParsePositiveSemidefiniteConstraints for the meaning of these
  // variables.
  std::vector<std::optional<int>> psd_y_start_indices;
  std::vector<std::optional<int>> lmi_y_start_indices;

  // See ParseScalarPositiveSemidefiniteConstraints for the meaning of these
  // variables.
  std::vector<std::optional<int>> scalar_psd_dual_indices;
  std::vector<std::optional<int>> scalar_lmi_dual_indices;

  // See Parse2x2PositiveSemidefiniteConstraints for the meaning of these
  // variables.
  std::vector<std::optional<int>> twobytwo_psd_dual_start_indices;
  std::vector<std::optional<int>> twobytwo_lmi_dual_start_indices;
};

struct ConicStandardFormInfo {
  // A vector of triplets to construct the matrix A.
  std::vector<Eigen::Triplet<double>> A_triplets;
  // A vector of values to construct the vector b.
  std::vector<double> b_std;
  // The number of rows in the matrix A.
  int A_row_count{0};

  // An unordered map from cone attributes to [row_start, row_end) pairs.
  std::unordered_map<ProgramAttribute, std::vector<std::pair<int, int>>>
      attributes_to_start_end_pairs{};

  DualInfo dual_info;
};
struct ConicStandardFormParsingInfo {
  CostAggregationInfo cost_info;
  ConstraintAggregationInfo constraint_info;
  DualInfo dual_info;
};

std::unique_ptr<ConvexAggregationInfo> ParseConicStandardForm(
    const MathematicalProgram& prog,
    const ConicStandardFormOptions& options = {});

}  // namespace internal

/** Stores the information required to represent a convex program  (specficially
 * an LP, QP, SOCP, or SDP), in the following conic standard form (sometimes
 * called the dual standard form) form:
 * min xᵀPx +cᵀx + d
 * subject to Ax - b ∈ K
 * Where K is the product of the cones:
 * 1) Zero cone {x | x = 0 }
 * 2) Positive orthant {x | x ≥ 0 }
 * 3) Second-order cone {(t, x) | |x|₂ ≤ t }
 * 4) Positive semidefinite cone { X |  min(eig(X)) ≥ 0, X = Xᵀ }
 * 5) Exponential cone { x |  x₀ ≥ x₁ exp(x₂/ x₁), x₁ > 0}
 *
 * If quadratic_cost is false, then the program is parsed so that the cost has
 * no quadratic term.
 */
class ConicStandardForm {
 public:
  explicit ConicStandardForm(const MathematicalProgram& prog,
                             const ConicStandardFormOptions& options = {});

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
  /** A sparse matrix encoding the quadratic part of the cost.*/
  Eigen::SparseMatrix<double> P_;

  /** A sparse vector encoding the cost. */
  Eigen::SparseVector<double> c_;
  /** The constant term in the cost. */
  double d_{0};

  /** A is the sparse matrix encoding the conic constraint. */
  Eigen::SparseMatrix<double> A_;

  /** b is a sparse matrix encoding the conic constraint. */
  Eigen::SparseVector<double> b_;

  /** The decision variables x of the program.*/
  const VectorX<symbolic::Variable> x_;

  std::unordered_map<ProgramAttribute, std::vector<std::pair<int, int>>>
      attributes_to_start_end_pairs_{};

  internal::DualInfo dual_variable_information_;
};

}  // namespace solvers
}  // namespace drake
