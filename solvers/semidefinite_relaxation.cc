#include "drake/solvers/semidefinite_relaxation.h"

#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "drake/common/fmt_eigen.h"
#include "drake/common/ssize.h"
#include "drake/common/text_logging.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/program_attribute.h"

namespace drake {
namespace solvers {

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using symbolic::Expression;
using symbolic::Variable;
using symbolic::Variables;

namespace {

const double kInf = std::numeric_limits<double>::infinity();

}  // namespace

std::unique_ptr<MathematicalProgram> MakeSemidefiniteRelaxation(
    const MathematicalProgram& prog,
    const std::optional<std::vector<Variables>> variable_groups) {
  std::string unsupported_message{};
  const ProgramAttributes supported_attributes(
      std::initializer_list<ProgramAttribute>{
          ProgramAttribute::kLinearCost, ProgramAttribute::kQuadraticCost,
          ProgramAttribute::kLinearConstraint,
          ProgramAttribute::kLinearEqualityConstraint,
          ProgramAttribute::kQuadraticConstraint});
  if (!AreRequiredAttributesSupported(prog.required_capabilities(),
                                      supported_attributes,
                                      &unsupported_message)) {
    throw std::runtime_error(fmt::format(
        "MakeSemidefiniteRelaxation() does not (yet) support this program: {}.",
        unsupported_message));
  }

  auto relaxation = std::make_unique<MathematicalProgram>();

  // Build a symmetric matrix X of decision variables using the original
  // program variables (so that GetSolution, etc, works using the original
  // variables).
  relaxation->AddDecisionVariables(prog.decision_variables());
  MatrixX<Variable> X(prog.num_vars() + 1, prog.num_vars() + 1);
  // X = xxᵀ; x = [prog.decision_vars(); 1].
  X.topLeftCorner(prog.num_vars(), prog.num_vars()) =
      relaxation->NewSymmetricContinuousVariables(prog.num_vars(), "Y");
  X.topRightCorner(prog.num_vars(), 1) = prog.decision_variables();
  X.bottomLeftCorner(1, prog.num_vars()) =
      prog.decision_variables().transpose();
  // X(-1,-1) = 1.
  Variable one("one");
  X(prog.num_vars(), prog.num_vars()) = one;
  relaxation->AddDecisionVariables(Vector1<Variable>(one));
  relaxation->AddBoundingBoxConstraint(1, 1,
                                       X(prog.num_vars(), prog.num_vars()));
  // X ≽ 0.
  relaxation->AddPositiveSemidefiniteConstraint(X);

  auto x = X.col(prog.num_vars());

  // Returns the {a, vars} in relaxation, such that a' vars = 0.5*tr(QY). This
  // assumes Q=Q', which is ensured by QuadraticCost and QuadraticConstraint.
  auto half_trace_QY = [&X, &prog](const Eigen::MatrixXd& Q,
                                   const VectorXDecisionVariable& prog_vars)
      -> std::pair<VectorXd, VectorX<Variable>> {
    const int N = prog_vars.size();
    const int num_vars = N * (N + 1) / 2;
    const std::vector<int> indices =
        prog.FindDecisionVariableIndices(prog_vars);
    VectorXd a = VectorXd::Zero(num_vars);
    VectorX<Variable> y(num_vars);
    int count = 0;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j <= i; ++j) {
        // tr(QY) = ∑ᵢ ∑ⱼ Qᵢⱼ Yⱼᵢ.
        a[count] = ((i == j) ? 0.5 : 1.0) * Q(i, j);
        y[count] = X(indices[i], indices[j]);
        ++count;
      }
    }
    return {a, y};
  };

  // Linear costs => Linear costs.
  for (const auto& binding : prog.linear_costs()) {
    relaxation->AddCost(binding);
  }
  // Quadratic costs.
  // 0.5 y'Qy + b'y + c => 0.5 tr(QY) + b'y + c
  for (const auto& binding : prog.quadratic_costs()) {
    const int N = binding.variables().size();
    const int num_vars = N + (N * (N + 1) / 2);
    std::pair<VectorXd, VectorX<Variable>> quadratic_terms =
        half_trace_QY(binding.evaluator()->Q(), binding.variables());
    VectorXd a(num_vars);
    VectorX<Variable> vars(num_vars);
    a << quadratic_terms.first, binding.evaluator()->b();
    vars << quadratic_terms.second, binding.variables();
    relaxation->AddLinearCost(a, binding.evaluator()->c(), vars);
  }

  // Linear constraints and bounding box constraints
  // lb ≤ Ay ≤ ub => lb ≤ Ay ≤ ub.

  // A map from the variable groups to all the linear constraints for which the
  // variable group intersects the variables of the linear constraint.
  std::map<Variables, std::vector<Triplet<double>>>
      variable_groups_to_A_triplets;
  std::map<Variables, std::vector<double>> variable_groups_to_b_vector;
  std::map<Variables, std::set<int>> group_to_submatrix_indices;
  if (!variable_groups.has_value()) {
    const Variables vars{prog.decision_variables()};
    variable_groups_to_A_triplets.emplace(vars, std::vector<Triplet<double>>{});
    variable_groups_to_b_vector.emplace(vars, std::vector<double>{});

    variable_groups_to_A_triplets.at(vars).reserve(
        ssize(prog.linear_constraints()));
    variable_groups_to_b_vector.at(vars).reserve(
        ssize(prog.linear_constraints()));
    group_to_submatrix_indices.emplace(vars, std::set<int>{});
    for (const auto& var : vars) {
      group_to_submatrix_indices.at(vars).emplace(
          prog.FindDecisionVariableIndex(var));
    }
  } else {
    for (const auto& group : variable_groups.value()) {
      variable_groups_to_A_triplets.emplace(group,
                                            std::vector<Triplet<double>>{});
      variable_groups_to_b_vector.emplace(group, std::vector<double>{});
      group_to_submatrix_indices.emplace(group, std::set<int>{});
      for (const auto& var : group) {
        group_to_submatrix_indices.at(group).emplace(
            prog.FindDecisionVariableIndex(var));
      }
    }
  }

  // Adds all the linear constraints and build up the sparse matrices for
  // representing Az ≤ b where z is each variable group. Later we will need
  // these sparse matrices in order to represent the products 0 ≤ (Az-b)(Az-b)ᵀ.

  // First we do the bounding box constraints.
  // TODO(bernhardpg): Consider special-casing linear equality constraints
  // that are added as bounding box or linear constraints with lb == ub
  for (const auto& binding : prog.bounding_box_constraints()) {
    relaxation->AddConstraint(binding);
    for (auto& [group, triplets] : variable_groups_to_A_triplets) {
      const Variables binding_vars{binding.variables()};
      if (binding_vars.IsSubsetOf(group)) {
        int cur_row_count = ssize(variable_groups_to_b_vector.at(group));
        std::map<Variable, int> local_variable_to_group_index;
        for (int i = 0; i < binding.evaluator()->num_constraints(); ++i) {
          const Variable cur_var{binding.variables()(i)};
          int group_index = static_cast<int>(
              std::distance(group.begin(), group.find(cur_var)));
          if (std::isfinite(binding.evaluator()->lower_bound()[i])) {
            triplets.push_back(
                Triplet<double>(cur_row_count, group_index, -1.0));
            variable_groups_to_b_vector.at(group).push_back(
                -binding.evaluator()->lower_bound()[i]);
            ++cur_row_count;
          }
          if (std::isfinite(binding.evaluator()->upper_bound()[i])) {
            triplets.push_back(
                Triplet<double>(cur_row_count, group_index, 1.0));
            variable_groups_to_b_vector.at(group).push_back(
                binding.evaluator()->upper_bound()[i]);
            ++cur_row_count;
          }
        }
      }
    }
  }

  // Now we do the linear constraints.
  for (const auto& binding : prog.linear_constraints()) {
    relaxation->AddConstraint(binding);
    for (auto& [group, triplets] : variable_groups_to_A_triplets) {
      Variables binding_vars{binding.variables()};
      if (binding_vars.IsSubsetOf(group)) {
        int cur_row_count = ssize(variable_groups_to_b_vector.at(group));
        std::map<Variable, int> local_variable_to_group_index;
        std::map<int, int> A_row_to_triplet_row_lower_bound;
        std::map<int, int> A_row_to_triplet_row_upper_bound;

        for (int k = 0; k < binding.evaluator()->get_sparse_A().outerSize();
             ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(
                   binding.evaluator()->get_sparse_A(), k);
               it; ++it) {
            const Variable cur_var{binding.variables()(it.col())};
            if (!local_variable_to_group_index.contains(cur_var)) {
              local_variable_to_group_index.emplace(
                  cur_var, static_cast<int>(std::distance(
                               group.begin(), group.find(cur_var))));
            }
            if (std::isfinite(binding.evaluator()->lower_bound()[it.row()])) {
              if (!A_row_to_triplet_row_lower_bound.contains(it.row())) {
                A_row_to_triplet_row_lower_bound.emplace(it.row(),
                                                         cur_row_count);
                variable_groups_to_b_vector.at(group).emplace_back(
                    -binding.evaluator()->lower_bound()[it.row()]);
                cur_row_count = ssize(variable_groups_to_b_vector.at(group));
              }
              triplets.emplace_back(
                  A_row_to_triplet_row_lower_bound.at(it.row()),
                  local_variable_to_group_index.at(cur_var), -it.value());
            }
            if (std::isfinite(binding.evaluator()->upper_bound()[it.row()])) {
              if (!A_row_to_triplet_row_upper_bound.contains(it.row())) {
                A_row_to_triplet_row_upper_bound.emplace(it.row(),
                                                         cur_row_count);
                variable_groups_to_b_vector.at(group).emplace_back(
                    binding.evaluator()->upper_bound()[it.row()]);
                cur_row_count = ssize(variable_groups_to_b_vector.at(group));
              }
              triplets.emplace_back(
                  A_row_to_triplet_row_upper_bound.at(it.row()),
                  local_variable_to_group_index.at(cur_var), it.value());
            }
          }
        }
      }
    }
  }

  for (const auto& [group, triplets] : variable_groups_to_A_triplets) {
    if (ssize(triplets) == 0) {
      continue;
    }
    // 0 ≤ (Az-b)(Az-b)ᵀ, implemented with
    // -bbᵀ ≤ AZAᵀ - b(Az)ᵀ - (Az)bᵀ.
    // where z is the local variable group and Z is [z,1]ᵀ[z,1].

    const std::vector<double> b_vec(variable_groups_to_b_vector.at(group));
    SparseMatrix<double> A(ssize(b_vec), ssize(group));
    A.setFromTriplets(triplets.begin(), triplets.end());
    const Eigen::Map<const VectorXd> b(b_vec.data(), ssize(b_vec));

    VectorX<Variable> z(ssize(group_to_submatrix_indices.at(group)));
    int count = 0;
    for (const int& i : group_to_submatrix_indices.at(group)) {
      z(count) = x(i);
      ++count;
    }
    MatrixX<Variable> Z = math::ExtractPrincipalSubmatrix(
        X, group_to_submatrix_indices.at(group));

    // TODO(russt): Avoid the symbolic computation here.
    const MatrixX<Expression> AZAT = A * Z * A.transpose();

    const VectorX<Expression> rhs_flat_tril =
        math::ToLowerTriangularColumnsFromMatrix(
            AZAT - b * (A * z).transpose() - A * z * b.transpose());
    const VectorXd bbT_flat_tril =
        math::ToLowerTriangularColumnsFromMatrix(-b * b.transpose());
    relaxation->AddLinearConstraint(
        rhs_flat_tril, bbT_flat_tril,
        VectorXd::Constant(bbT_flat_tril.size(), kInf));
  }

  // Linear equality constraints.
  // Ay = b => (Ay-b)xᵀ = Ayxᵀ - bxᵀ = 0.
  // Note that this contains Ay=b since x contains 1.
  std::map<Variables, std::vector<Triplet<double>>>
      variable_groups_to_Ab_eq_triplets;
  std::map<Variables, int> variable_groups_to_Ab_eq_triplets_row_count;
  if (!variable_groups.has_value()) {
    const Variables vars{prog.decision_variables()};
    variable_groups_to_Ab_eq_triplets.emplace(vars,
                                              std::vector<Triplet<double>>{});
    variable_groups_to_Ab_eq_triplets.at(vars).reserve(
        ssize(prog.linear_equality_constraints()));
    variable_groups_to_Ab_eq_triplets_row_count.emplace(vars, 0);
  } else {
    for (const auto& group : variable_groups.value()) {
      variable_groups_to_Ab_eq_triplets.emplace(group,
                                                std::vector<Triplet<double>>{});
      variable_groups_to_Ab_eq_triplets_row_count.emplace(group, 0);
    }
  }

  for (const auto& binding : prog.linear_equality_constraints()) {
    relaxation->AddConstraint(binding);
    for (auto& [group, triplets] : variable_groups_to_Ab_eq_triplets) {
      const Variables binding_vars{binding.variables()};
      if (binding_vars.IsSubsetOf(group)) {
        int cur_row_count =
            variable_groups_to_Ab_eq_triplets_row_count.at(group);
        std::map<Variable, int> local_variable_to_group_index;
        std::map<int, int> Ab_eq_row_to_triplet_row;

        for (int k = 0; k < binding.evaluator()->get_sparse_A().outerSize();
             ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(
                   binding.evaluator()->get_sparse_A(), k);
               it; ++it) {
            const Variable cur_var{binding.variables()(it.col())};
            if (!local_variable_to_group_index.contains(cur_var)) {
              local_variable_to_group_index.emplace(
                  cur_var, static_cast<int>(std::distance(
                               group.begin(), group.find(cur_var))));
            }
            if (!Ab_eq_row_to_triplet_row.contains(it.row())) {
              Ab_eq_row_to_triplet_row.emplace(it.row(), cur_row_count);
              triplets.emplace_back(
                  cur_row_count, ssize(group),
                  -binding.evaluator()->lower_bound()[it.row()]);
              ++cur_row_count;
            }
            triplets.emplace_back(Ab_eq_row_to_triplet_row.at(it.row()),
                                  local_variable_to_group_index.at(cur_var),
                                  it.value());
          }
        }
        variable_groups_to_Ab_eq_triplets_row_count.at(group) = cur_row_count;
      }
    }
  }
  // Now add the constraint AZ-bz = 0.
  for (const auto& [group, triplets] : variable_groups_to_Ab_eq_triplets) {
    SparseMatrix<double> Ab(
        variable_groups_to_Ab_eq_triplets_row_count.at(group),
        ssize(group) + 1);
    Ab.setFromTriplets(triplets.begin(), triplets.end());
    // Add the constraints one column at a time.
    for (const int& j : group_to_submatrix_indices.at(group)) {
      VectorX<Variable> vars(Ab.cols());
      int count = 0;
      for (const int& i : group_to_submatrix_indices.at(group)) {
        vars[count] = X(i, j);
        ++count;
      }
      vars[ssize(group)] = x[j];
      relaxation->AddLinearEqualityConstraint(Ab, VectorXd::Zero(Ab.rows()),
                                              vars);
    }
  }

  // Quadratic constraints.
  // lb ≤ 0.5 y'Qy + b'y ≤ ub => lb ≤ 0.5 tr(QY) + b'y ≤ ub
  for (const auto& binding : prog.quadratic_constraints()) {
    const int N = binding.variables().size();
    const int num_vars = N + (N * (N + 1) / 2);
    std::pair<VectorXd, VectorX<Variable>> quadratic_terms =
        half_trace_QY(binding.evaluator()->Q(), binding.variables());
    VectorXd a(num_vars);
    VectorX<Variable> vars(num_vars);
    a << quadratic_terms.first, binding.evaluator()->b();
    vars << quadratic_terms.second, binding.variables();
    relaxation->AddLinearConstraint(a.transpose(),
                                    binding.evaluator()->lower_bound(),
                                    binding.evaluator()->upper_bound(), vars);
  }

  return relaxation;
}

}  // namespace solvers
}  // namespace drake
