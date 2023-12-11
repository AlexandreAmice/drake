#include "drake/solvers/semidefinite_relaxation.h"

#include <initializer_list>
#include <limits>
#include <string>
#include <utility>
#include <vector>

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

  // Bounding Box constraints
  // lb ≤ y ≤ ub => lb ≤ y ≤ ub
  for (const auto& binding : prog.bounding_box_constraints()) {
    relaxation->AddConstraint(binding);
  }

  // Linear constraints
  // lb ≤ Ay ≤ ub => lb ≤ Ay ≤ ub.
  // TODO(Alexandre.Amice) annotate what you're doing here.
  std::vector<Triplet<double>> A_triplets;
  // The performance of constructing a std::vector is much better than the
  // performance of constructing an Eigen::VectorXd so we first construct the
  // std::vector which we later will convert.
  std::vector<double> b_vector;
  // Worst case we store a quadratic number of coefficients for b.
  b_vector.reserve(ssize(prog.linear_constraints()) *
                   ssize(prog.linear_constraints()));
  // A map from the variable groups to all the linear constraints for which the
  // variable group intersects the variables of the linear constraint.

  std::map<Variables, std::vector<solvers::Binding<solvers::LinearConstraint>>>
      variable_groups_to_linear_constraints;
  std::map<Variables, std::vector<Triplet<double>>>
      variable_groups_to_A_triplets;
  std::map<Variables, std::vector<double>> variable_groups_to_b_vector;
  if (!variable_groups.has_value()) {
    variable_groups_to_linear_constraints.emplace(
        Variables(prog.decision_variables()),
        std::vector<solvers::Binding<solvers::LinearConstraint>>{});
  } else {
    for (const auto& group : variable_groups.value()) {
      variable_groups_to_linear_constraints.emplace(
          group, std::vector<solvers::Binding<solvers::LinearConstraint>>{});
    }
  }
  for (const auto& [group, _] : variable_groups_to_linear_constraints) {
    variable_groups_to_A_triplets.emplace(group,
                                          std::vector<Triplet<double>>{});
    variable_groups_to_b_vector.emplace(group, std::vector<double>{});
  }

  for (const auto& binding : prog.linear_constraints()) {
    relaxation->AddConstraint(binding);
    for (const auto& [group, constraints] :
         variable_groups_to_linear_constraints) {
      Variables binding_vars{binding.variables()};
      if (Variables(binding_vars).IsSubsetOf(group)) {
        variable_groups_to_linear_constraints.at(group).emplace_back(binding);

        int cur_row_count = ssize(variable_groups_to_b_vector.at(group));
        std::map<Variable, int> local_variable_to_group_index;
        for (int k = 0; k < binding.evaluator()->get_sparse_A().outerSize();
             ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(
                   binding.evaluator()->get_sparse_A(), k);
               it; ++it) {
            const Variable cur_var{binding.variables()(it.col())};
            if (!local_variable_to_group_index.contains(cur_var)) {
              local_variable_to_group_index.at(cur_var) = static_cast<int>(
                  std::distance(group.begin(), group.find(cur_var)));
            }
            if (std::isfinite(binding.evaluator()->lower_bound()[it.row()])) {
              variable_groups_to_A_triplets.at(group).emplace_back(
                  cur_row_count, local_variable_to_group_index.at(cur_var),
                  -it.value());
              variable_groups_to_b_vector.at(group).emplace_back(
                  binding.evaluator()->lower_bound()[it.row()]);
              ++cur_row_count;
            }
            if (std::isfinite(binding.evaluator()->upper_bound()[it.row()])) {
              variable_groups_to_A_triplets.at(group).emplace_back(
                  cur_row_count, local_variable_to_group_index.at(cur_var),
                  it.value());
              variable_groups_to_b_vector.at(group).emplace_back(
                  binding.evaluator()->upper_bound()[it.row()]);
              ++cur_row_count;
            }
          }
        }
      }
    }
  }
  // NOW WE SHOULD HAVE Ay <= b for each variable group.

  {
    // Now assemble one big Ay <= b matrix from all bounding box constraints
    // and linear constraints
    // TODO(bernhardpg): Consider special-casing linear equality constraints
    // that are added as bounding box or linear constraints with lb == ub
    int num_constraints = 0;
    int nnz = 0;
    for (const auto& binding : prog.bounding_box_constraints()) {
      for (int i = 0; i < binding.evaluator()->num_constraints(); ++i) {
        if (std::isfinite(binding.evaluator()->lower_bound()[i])) {
          ++num_constraints;
        }
        if (std::isfinite(binding.evaluator()->upper_bound()[i])) {
          ++num_constraints;
        }
      }
      nnz += binding.evaluator()->get_sparse_A().nonZeros();
    }
    for (const auto& binding : prog.linear_constraints()) {
      for (int i = 0; i < binding.evaluator()->num_constraints(); ++i) {
        if (std::isfinite(binding.evaluator()->lower_bound()[i])) {
          ++num_constraints;
        }
        if (std::isfinite(binding.evaluator()->upper_bound()[i])) {
          ++num_constraints;
        }
      }
      nnz += binding.evaluator()->get_sparse_A().nonZeros();
    }

    //    std::vector<Triplet<double>> A_triplets;
    A_triplets.reserve(nnz);
    SparseMatrix<double> A(num_constraints, prog.num_vars());
    VectorXd b(num_constraints);

    int constraint_idx = 0;
    for (const auto& binding : prog.bounding_box_constraints()) {
      const std::vector<int> indices =
          prog.FindDecisionVariableIndices(binding.variables());
      for (int i = 0; i < binding.evaluator()->num_constraints(); ++i) {
        if (std::isfinite(binding.evaluator()->lower_bound()[i])) {
          A_triplets.push_back(
              Triplet<double>(constraint_idx, indices[i], -1.0));
          b(constraint_idx++) = -binding.evaluator()->lower_bound()[i];
        }
        if (std::isfinite(binding.evaluator()->upper_bound()[i])) {
          A_triplets.push_back(
              Triplet<double>(constraint_idx, indices[i], 1.0));
          b(constraint_idx++) = binding.evaluator()->upper_bound()[i];
        }
      }
    }

    for (const auto& binding : prog.linear_constraints()) {
      const std::vector<int> indices =
          prog.FindDecisionVariableIndices(binding.variables());
      // TODO(hongkai-dai): Consider using the SparseMatrix iterators.
      for (int i = 0; i < binding.evaluator()->num_constraints(); ++i) {
        if (std::isfinite(binding.evaluator()->lower_bound()[i])) {
          for (int j = 0; j < binding.evaluator()->num_vars(); ++j) {
            if (binding.evaluator()->get_sparse_A().coeff(i, j) != 0) {
              A_triplets.push_back(Triplet<double>(
                  constraint_idx, indices[j],
                  -binding.evaluator()->get_sparse_A().coeff(i, j)));
            }
          }
          b(constraint_idx++) = -binding.evaluator()->lower_bound()[i];
        }
        if (std::isfinite(binding.evaluator()->upper_bound()[i])) {
          for (int j = 0; j < binding.evaluator()->num_vars(); ++j) {
            if (binding.evaluator()->get_sparse_A().coeff(i, j) != 0) {
              A_triplets.push_back(Triplet<double>(
                  constraint_idx, indices[j],
                  binding.evaluator()->get_sparse_A().coeff(i, j)));
            }
          }
          b(constraint_idx++) = binding.evaluator()->upper_bound()[i];
        }
      }
    }
    A.setFromTriplets(A_triplets.begin(), A_triplets.end());

    // 0 ≤ (Ay-b)(Ay-b)ᵀ, implemented with
    // -bbᵀ ≤ AYAᵀ - b(Ay)ᵀ - (Ay)bᵀ.
    // TODO(russt): Avoid the symbolic computation here.
    // TODO(russt): Avoid the dense matrix.

    const MatrixX<Expression> AYAT =
        A * X.topLeftCorner(prog.num_vars(), prog.num_vars()) * A.transpose();
    const VectorX<Variable> y = x.head(prog.num_vars());

    const VectorX<Expression> rhs_flat_tril =
        math::ToLowerTriangularColumnsFromMatrix(
            AYAT - b * (A * y).transpose() - A * y * b.transpose());
    const VectorXd bbT_flat_tril =
        math::ToLowerTriangularColumnsFromMatrix(-b * b.transpose());

    relaxation->AddLinearConstraint(
        rhs_flat_tril, bbT_flat_tril,
        VectorXd::Constant(bbT_flat_tril.size(), kInf));
  }

  // Linear equality constraints.
  // Ay = b => (Ay-b)xᵀ = Ayxᵀ - bxᵀ = 0.
  // Note that this contains Ay=b since x contains 1.
  for (const auto& binding : prog.linear_equality_constraints()) {
    const int N = binding.variables().size();
    const std::vector<int> indices =
        prog.FindDecisionVariableIndices(binding.variables());
    VectorX<Variable> vars(N + 1);
    // Add the constraints one column at a time:
    // Ayx_j - bx_j = 0.
    MatrixX<double> Ab(binding.evaluator()->num_constraints(), N + 1);
    // TODO(Alexandre.Amice) make this only access the sparse matrix.
    Ab.leftCols(N) = binding.evaluator()->GetDenseA();
    Ab.col(N) = -binding.evaluator()->lower_bound();
    for (int j = 0; j < static_cast<int>(x.size()); ++j) {
      for (int i = 0; i < N; ++i) {
        vars[i] = X(indices[i], j);
      }
      vars[N] = x[j];
      relaxation->AddLinearEqualityConstraint(
          Ab, VectorXd::Zero(binding.evaluator()->num_constraints()), vars);
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
