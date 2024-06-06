#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/solvers/binding.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/cost.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {
/** Given many linear costs, aggregate them into
 *
 *     aᵀ*x + b,
 * @param linear_costs the linear costs to be aggregated.
 * @param linear_coeff[out] a in the documentation above.
 * @param vars[out] x in the documentation above.
 * @param constant_cost[out] b in the documentation above.
 */
void AggregateLinearCosts(const std::vector<Binding<LinearCost>>& linear_costs,
                          Eigen::SparseVector<double>* linear_coeff,
                          VectorX<symbolic::Variable>* vars,
                          double* constant_cost);

/** Given many linear and quadratic costs, aggregate them into
 *
 *     0.5*x₁ᵀQx₁ + bᵀx₂ + c
 * where x₁ and x₂ don't need to be the same.
 * @param quadratic_costs The quadratic costs to be aggregated.
 * @param linear_costs The linear costs to be aggregated.
 * @param Q_lower[out] The lower triangular part of the matrix Q.
 * @param quadratic_vars[out] x₁ in the documentation above.
 * @param linear_coeff[out] b in the documentation above.
 * @param linear_vars[out] x₂ in the documentation above.
 * @param constant_cost[out] c in the documentation above.
 */
void AggregateQuadraticAndLinearCosts(
    const std::vector<Binding<QuadraticCost>>& quadratic_costs,
    const std::vector<Binding<LinearCost>>& linear_costs,
    Eigen::SparseMatrix<double>* Q_lower,
    VectorX<symbolic::Variable>* quadratic_vars,
    Eigen::SparseVector<double>* linear_coeff,
    VectorX<symbolic::Variable>* linear_vars, double* constant_cost);

/**
 * Stores the lower and upper bound of a variable.
 */
struct Bound {
  /** Lower bound. */
  double lower{};
  /** Upper bound. */
  double upper{};
};

/**
 * Aggregates many bounding box constraints, returns the intersection (the
 * tightest bounds) of these constraints.
 * @param bounding_box_constraints The constraints to be aggregated.
 * @retval aggregated_bounds aggregated_bounds[var.get_id()] returns the
 * (lower, upper) bounds of that variable as the tightest bounds of @p
 * bounding_box_constraints.
 */
[[nodiscard]] std::unordered_map<symbolic::Variable, Bound>
AggregateBoundingBoxConstraints(
    const std::vector<Binding<BoundingBoxConstraint>>&
        bounding_box_constraints);

/**
 * Aggregates all the BoundingBoxConstraints inside @p prog, returns the
 * intersection of the bounding box constraints as the lower and upper bound for
 * each variable in @p prog.
 * @param prog The optimization program containing decision variables and
 * BoundingBoxConstraints.
 * @param[out] lower (*lower)[i] is the lower bound of
 * prog.decision_variable(i).
 * @param[out] upper (*upper)[i] is the upper bound of
 * prog.decision_variable(i).
 */
void AggregateBoundingBoxConstraints(const MathematicalProgram& prog,
                                     Eigen::VectorXd* lower,
                                     Eigen::VectorXd* upper);

/**
 * Overloads AggregateBoundingBoxConstraints, but the type of lower and upper
 * are std::vector<double>.
 */
void AggregateBoundingBoxConstraints(const MathematicalProgram& prog,
                                     std::vector<double>* lower,
                                     std::vector<double>* upper);

/**
 For linear expression A * vars where `vars` might contain duplicated entries,
 rewrite this linear expression as A_new * vars_new where vars_new doesn't
 contain duplicated entries.
 */
void AggregateDuplicateVariables(const Eigen::SparseMatrix<double>& A,
                                 const VectorX<symbolic::Variable>& vars,
                                 Eigen::SparseMatrix<double>* A_new,
                                 VectorX<symbolic::Variable>* vars_new);

/**
 * Aggregate all convex (conic) constraints into conic standard form i.e. a
 * single constraint Ax + b ∈ K where K is the product of cones.
 */
void AggregateConvexConstraints(const MathematicalProgram& prog,
                                Eigen::SparseMatrix<double>* A,
                                Eigen::VectorXd* b,
                                Eigen::SparseMatrix<double>* Aeq,
                                Eigen::SparseMatrix<double>* beq);

namespace internal {

// Information required to aggregate the convex constraints of a program into
// the form.
//   A x + s = b
//     s in K
// where K is a Cartesian product of some primitive cones. This information is
// needed by SCS and Clarabel to transcribe the problem to the solvers and
// extract dual solutions.
struct ConvexConstraintAggregationInfo {
  // A vector of triplets to construct the matrix A.
  std::vector<Eigen::Triplet<double>> A_triplets;
  // A vector of values to construct the vector b.
  std::vector<double> b_std;
  // The number of rows in the matrix A.
  int A_row_count{0};
  // The total number of variables with bounding box constraints such that
  // lb < ub.
  int num_bounding_box_inequality_constraint_rows{0};
  //  bounding_box_constraint_dual_indices[i][j] are the indices of
  // the dual variable for the j'th row of prog.bounding_box_constraints()[i].
  // We use -1 to indicate that it is impossible for this constraint to be
  // active (for example, another BoundingBoxConstraint imposes a tighter bound
  // on the same variable).
  std::vector<std::vector<std::pair<int, int>>>
      bounding_box_constraint_dual_indices;
  // The total number of linear equality constraints.
  int num_linear_equality_constraint_rows{0};
  // The start indices of the dual variables for each linear equality
  // constraints.
  std::vector<int> linear_eq_dual_variable_start_indices;
  // The total number of linear constraints.
  int num_linear_constraint_rows{0};
  // The start indices of the dual variables for each linear constraints.
  std::vector<std::vector<std::pair<int, int>>> linear_constraint_dual_indices;
  // The lengths of each second order cone s in the Cartesian product K. See
  //  // ParseSecondOrderConeConstraints for more details.
  std::vector<int> second_order_cone_lengths;
  // lorentz_cone_y_start_indices y[lorentz_cone_y_start_indices[i]:
  // lorentz_cone_y_start_indices[i] + second_order_cone_length[i]] are the dual
  // variables for prog.lorentz_cone_constraints()[i]. See
  //  // ParseSecondOrderConeConstraints for more details.
  std::vector<int> lorentz_cone_dual_variable_start_indices;
  // y[rotated_lorentz_cone_y_start_indices[i]:
  // rotated_lorentz_cone_y_start_indices[i] +
  // prog.rotate_lorentz_cone()[i].evaluator().A().rows] are the y variables for
  // prog.rotated_lorentz_cone_constraints()[i]. See
  // ParseSecondOrderConeConstraints for more details.
  std::vector<int> rotated_lorentz_cone_dual_variable_start_indices;
  // The length of all the psd cones from
  // prog.positive_semidefinite_constraints() and
  // prog.linear_matrix_inequality_constraints(). See
  // ParsePositiveSemidefiniteConstraints for more details.
  std::vector<int> psd_cone_lengths;
};

// Iterate over the convex (conic) constraints of prog and aggregate the
// information required to write them in the form.
//   A x + s = b
//     s in K
// where K is a Cartesian product of some primitive cones.
// The supported cones are
// Zero cone {x | x = 0 }
// Positive orthant {x | x ≥ 0 }
// Box cones orthant {x | ub ≥ x ≥ lb }
// Second-order cone {(t, x) | |x|₂ ≤ t }
// Positive semidefinite cone { X | min(eig(X)) ≥ 0, X = Xᵀ }
// Exponential cone {x,y,z): y>0, y*exp(x/y) <= z}
// Power cone {(x, y, z): pow(x, α)*pow(y, 1-α) >= |z|, x>=0, y>=0} with α in
// (0, 1)
// This convention is compatible with both the SCS and Clarabel solvers.
void DoAggregateConvexConstraints(const MathematicalProgram& prog,
                                  ConvexConstraintAggregationInfo* info);

// Parse all the bounding box constraints in `prog` to the form
// A_eq*x+s=b_eq, s in zero cone and A_ineq*x+s=b_ineq, s in positive cone.
// @param[in/out] A_eq_triplets Append non-zero (row, col, val) triplets of the
// equality bounding box constraints to A_eq_triplets.
// @param[in/out] b_eq append entries to b_eq.
// @param[in/out] A_eq_row_count The number of rows in A_eq before/after calling
// this function.
// @param[in/out] A_ineq_triplets Append non-zero (row, col, val) triplets of
// the equality bounding box constraints to A_ineq_triplets.
// @param[in/out] b_ineq append entries to b_ineq.
// @param[in/out] A_ineq_row_count The number of rows in A_ineq before/after
// calling this function.
// @param[out] bbcon_dual_indices bbcon_dual_indices[i][j] are the indices of
// the dual variable for the j'th row of prog.bounding_box_constraints()[i]. We
// use -1 to indicate that it is impossible for this constraint to be active
// (for example, another BoundingBoxConstraint imposes a tighter bound on the
// same variable).
void ParseBoundingBoxConstraints(
    const MathematicalProgram& prog,
    std::vector<Eigen::Triplet<double>>* A_eq_triplets,
    std::vector<double>* b_eq, int* A_eq_row_count,
    std::vector<Eigen::Triplet<double>>* A_ineq_triplets,
    std::vector<double>* b_ineq, int* A_ineq_row_count,
    std::vector<std::vector<std::pair<int, int>>>* bbcon_dual_indices);

// Returns the first non-convex quadratic cost among @p quadratic_costs. If
// all quadratic costs are convex, then return a nullptr.
[[nodiscard]] const Binding<QuadraticCost>* FindNonconvexQuadraticCost(
    const std::vector<Binding<QuadraticCost>>& quadratic_costs);
// Returns the first non-convex quadratic constraint among @p
// quadratic_constraints. If all quadratic constraints are convex, then returns
// a nullptr.
[[nodiscard]] const Binding<QuadraticConstraint>*
FindNonconvexQuadraticConstraint(
    const std::vector<Binding<QuadraticConstraint>>& quadratic_constraints);

// If the program is compatible with this solver (the solver meets the required
// capabilities of the program, and the program is convex), returns true and
// clears the explanation.  Otherwise, returns false and sets the explanation.
// In either case, the explanation can be nullptr in which case it is ignored.
bool CheckConvexSolverAttributes(const MathematicalProgram& prog,
                                 const ProgramAttributes& solver_capabilities,
                                 std::string_view solver_name,
                                 std::string* explanation);

// Aggregates all prog.linear_costs(), such that the aggregated linear cost is
// ∑ᵢ (*c)[i] * prog.decision_variable(i) + *constant
// @note c and constant might not be zero. This function adds
// prog.linear_costs() to the coefficient c and constant.
// @pre c->size() >= prog.num_vars();
void ParseLinearCosts(const MathematicalProgram& prog, std::vector<double>* c,
                      double* constant);

// Parses all prog.linear_equality_constraints() to
// A*x = b
// Some convex solvers (like SCS and Clarabel) aggregates all constraints in the
// form of
// A*x + s = b
// s in K
// This function appends all prog.linear_equality_constraints() to the existing
// A*x+s=b.
// @param[in/out] A_triplets The triplets on the non-zero entries in A.
// prog.linear_equality_constraints() will be appended to A_triplets.
// @param[in/out] b The righthand side of A*x=b.
// prog.linear_equality_constraints() will be appended to b.
// @param[in/out] A_row_count The number of rows in A before and after calling
// this function.
// @param[out] linear_eq_y_start_indices linear_eq_y_start_indices[i] is the
// starting index of the dual variable for the constraint
// prog.linear_equality_constraints()[i]. Namely y[linear_eq_y_start_indices[i]:
// linear_eq_y_start_indices[i] +
// prog.linear_equality_constraints()[i].evaluator()->num_constraints] are the
// dual variables for  the linear equality constraint
// prog.linear_equality_constraint()(i), where y is the vector containing all
// dual variables.
// @param[out] num_linear_equality_constraints_rows The number of new rows
// appended to A*x+s=b in all prog.linear_equality_constraints(). Note
// num_linear_equality_constraints_rows is A_row_count AFTER calling this
// function minus A_row_count BEFORE calling this function.
void ParseLinearEqualityConstraints(
    const solvers::MathematicalProgram& prog,
    std::vector<Eigen::Triplet<double>>* A_triplets, std::vector<double>* b,
    int* A_row_count, std::vector<int>* linear_eq_y_start_indices,
    int* num_linear_equality_constraints_rows);

// Parses all prog.linear_constraints() to
// A*x + s = b
// s in the nonnegative orthant cone.
// Some convex solvers (like SCS and Clarabel) aggregates all constraints in the
// form of
// A*x + s = b
// s in K
// This function appends all prog.linear_constraints() to the existing
// A*x+s=b.
// @param[in/out] A_triplets The triplets on the non-zero entries in A.
// prog.linear_constraints() will be appended to A_triplets.
// @param[in/out] b The righthand side of A*x+s=b.
// prog.linear_equality_constraints() will be appended to b.
// @param[in/out] A_row_count The number of rows in A before and after calling
// this function.
// @param[out] linear_constraint_dual_indices
// linear_constraint_dual_indices[i][j].first/linear_constraint_dual_indices[i][j].second
// is the dual variable for the lower/upper bound of the j'th row in the
// linear constraint prog.linear_constraint()[i], we use -1 to indicate that
// the lower or upper bound is infinity.
// @param[out] num_linear_constraint_rows The number of new rows appended to
// A*x+s = b in all
// prog.linear_constraints()
void ParseLinearConstraints(const solvers::MathematicalProgram& prog,
                            std::vector<Eigen::Triplet<double>>* A_triplets,
                            std::vector<double>* b, int* A_row_count,
                            std::vector<std::vector<std::pair<int, int>>>*
                                linear_constraint_dual_indices,
                            int* num_linear_constraint_rows);

// Aggregates all quadratic prog.quadratic_costs() and add the aggregated cost
// to 0.5*x'P*x + c'*x + constant. where x is prog.decision_variables().
void ParseQuadraticCosts(const MathematicalProgram& prog,
                         std::vector<Eigen::Triplet<double>>* P_upper_triplets,
                         std::vector<double>* c, double* constant);

// Parse a L2NormCost |Cx+d|₂ to Clarabel/SCS format.
// We need to
// 1. Append a slack variable t.
// 2. Impose the constraint t >= |Cx+d|₂ as a Lorentz cone constraint.
// 3. Add the cost t.
// For the second step, we formulate it as
// [ 0  -1] * [x] + s = [0]       (1)
// [-C   0]   [t]       [d]
// s is in Lorentz cone.          (2)
// @param[in] prog We parse all prog.l2norm_costs().
// @param[in/out] num_solver_variables The number of variables in the solver
// before/after parsing this L2NormCost. The new slack variable t will be
// appended after the existing variables in the solver.
// @param[in/out] A_triplets We append the left hand side of the linear
// constraints (1) to A_triplets.
// @param[in/out] b We append the right hand side of the linear constraints (1)
// to b.
// @param[in/out] A_row_count The number of rows in A before/after appending the
// second order cone constraints.
// @param[in/out] second_order_cone_length The dimension of each second order
// cone in the overall program.
// @param[in/out] lorentz_cone_y_start_indices The starting indices of s in each
// Lorentz cone constraints, before/after calling this function. We append the
// indices of the dual variables for the Lorentz cone constraint to
// lorentz_cone_y_start_indices.
// @param[out] cost_coeffs The coefficient of each variable in the cost.
// @param[out] t_slack_indices The index of the new variable t in the solver
// (such as Clarabel or SCS, not in the MathematicalProgram object).
void ParseL2NormCosts(const MathematicalProgram& prog,
                      int* num_solver_variables,
                      std::vector<Eigen::Triplet<double>>* A_triplets,
                      std::vector<double>* b, int* A_row_count,
                      std::vector<int>* second_order_cone_length,
                      std::vector<int>* lorentz_cone_y_start_indices,
                      std::vector<double>* cost_coeffs,
                      std::vector<int>* t_slack_indices);

// Parse all second order cone constraints (including both Lorentz cone and
// rotated Lorentz cone constraint) to the form A*x+s=b, s in K where K is the
// Cartesian product of Lorentz cone {s | sqrt(s₁²+...+sₙ₋₁²) ≤ s₀}
// @param[in/out] A_triplets We append the second order cone constraints to
// A_triplets.
// @param[in/out] b We append the second order cone constraints to b.
// @param[in/out] The number of rows in A before/after appending the second
// order cone constraints.
// @param[out] second_order_cone_length The lengths of each second order cone s
// in K in the Cartesian product K.
// @param[out] lorentz_cone_y_start_indices y[lorentz_cone_y_start_indices[i]:
// lorentz_cone_y_start_indices[i] + second_order_cone_length[i]]
// are the dual variables for prog.lorentz_cone_constraints()[i].
// @param[out] rotated_lorentz_cone_y_start_indices
// y[rotated_lorentz_cone_y_start_indices[i]:
// rotated_lorentz_cone_y_start_indices[i] +
// prog.rotate_lorentz_cone()[i].evaluator().A().rows] are the y variables for
// prog.rotated_lorentz_cone_constraints()[i]. Note that we assume the Cartesian
// product K doesn't contain a
// rotated Lorentz cone constraint, instead we convert the rotated Lorentz
// cone constraint to the Lorentz cone constraint through a linear
// transformation. Hence we need to apply the transpose of that linear
// transformation on the y variable to get the dual variable in the dual cone
// of rotated Lorentz cone.
void ParseSecondOrderConeConstraints(
    const MathematicalProgram& prog,
    std::vector<Eigen::Triplet<double>>* A_triplets, std::vector<double>* b,
    int* A_row_count, std::vector<int>* second_order_cone_length,
    std::vector<int>* lorentz_cone_y_start_indices,
    std::vector<int>* rotated_lorentz_cone_y_start_indices);

// Add a rotated Lorentz cone constraint that A_cone * x + b_cone is in the
// rotated Lorentz cone.
//
// We add these rotated Lorentz cones in the form A*x+s=b and s in K, where K is
// the Cartesian product of Lorentz cones.
// @param A_cone_triplets The triplets of non-zero entries in A_cone.
// @param b_cone A_cone * x + b_cone is in the rotated Lorentz cone.
// @param x_indices The index of the variables.
// @param A_triplets[in/out] The non-zero entry triplet in A before and after
// adding the Lorentz cone.
// @param b[in/out] The right-hand side vector b before and after adding the
// Lorentz cone.
// @param A_row_count[in/out] The number of rows in A before and after adding
// the Lorentz cone.
// @param second_order_cone_length[in/out] The length of each Lorentz cone
// before and after adding the Lorentz cone constraint.
// @param rotated_lorentz_cone_y_start_indices[in/out] The starting index of y
// corresponds to this rotated Lorentz cone. If set to nullopt, then we don't
// modify rotated_lorentz_cone_y_start_indices.
void ParseRotatedLorentzConeConstraint(
    const std::vector<Eigen::Triplet<double>>& A_cone_triplets,
    const Eigen::Ref<const Eigen::VectorXd>& b_cone,
    const std::vector<int>& x_indices,
    std::vector<Eigen::Triplet<double>>* A_triplets, std::vector<double>* b,
    int* A_row_count, std::vector<int>* second_order_cone_length,
    std::optional<std::vector<int>*> rotated_lorentz_cone_y_start_indices);

// Parses all prog.exponential_cone_constraints() to
// A*x + s = b
// s in the exponential cone.
// Some convex solvers (like SCS and Clarabel) aggregates all constraints in the
// form of
// A*x + s = b
// s in K
// Note that the definition of SCS/Clarabel's exponential cone is different from
// Drake's definition. In SCS/Clarabel, a vector s is in the exponential cone if
// s₂≥ s₁*exp(s₀ / s₁). In Drake, a vector z is in the exponential cone if z₀ ≥
// z₁ * exp(z₂ / z₁). Note that the index 0 and 2 is swapped. This function
// appends all prog.exponential_cone_constraints() to the existing A*x+s=b.
// @param[in/out] A_triplets The triplets on the non-zero entries in A.
// prog.exponential_cone_constraints() will be appended to A_triplets.
// @param[in/out] b The righthand side of A*x+s=b.
// prog.exponential_cone_constraints() will be appended to b.
// @param[in/out] A_row_count The number of rows in A before and after calling
// this function.
void ParseExponentialConeConstraints(
    const MathematicalProgram& prog,
    std::vector<Eigen::Triplet<double>>* A_triplets, std::vector<double>* b,
    int* A_row_count);

// This function parses prog.positive_semidefinite_constraints() and
// prog.linear_matrix_inequality_constraints() into SCS/Clarabel format.
// A * x + s = b
// s in K
// Note that the SCS/Clarabel solver defines its psd cone with a √2 scaling on
// the off-diagonal terms in the positive semidefinite matrix. Refer to
// https://www.cvxgrp.org/scs/api/cones.html#semidefinite-cones and
// https://oxfordcontrol.github.io/ClarabelDocs/stable/examples/example_sdp/ for
// an explanation.
// @param[in] upper_triangular Whether we use the upper triangular or lower
// triangular part of the symmetric matrix. SCS uses the lower triangular part,
// and Clarabel uses the upper triangular part.
// @param[in/out] A_triplets The triplets on the non-zero entries in A.
// prog.positive_semidefinite_constraints() and
// prog.linear_matrix_inequality_constraints() will be appended to A_triplets.
// @param[in/out] b The righthand side of A*x+s=b.
// prog.positive_semidefinite_constraints() and
// prog.linear_matrix_inequality_constraints() will be appended to b.
// @param[in/out] A_row_count The number of rows in A before and after calling
// this function.
// @param[out] psd_cone_length The length of all the psd cones from
// prog.positive_semidefinite_constraints() and
// prog.linear_matrix_inequality_constraints().
void ParsePositiveSemidefiniteConstraints(
    const MathematicalProgram& prog, bool upper_triangular,
    std::vector<Eigen::Triplet<double>>* A_triplets, std::vector<double>* b,
    int* A_row_count, std::vector<int>* psd_cone_length);
}  // namespace internal
}  // namespace solvers
}  // namespace drake
