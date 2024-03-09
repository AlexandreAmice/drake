#include "drake/solvers/semidefinite_relaxation.h"

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

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

void ValidateProgramIsSupported(const MathematicalProgram& prog) {
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
}

bool CheckProgramRequireSemidefiniteRelaxation(
    const MathematicalProgram& prog) {
  for (const auto& cost : prog.quadratic_costs()) {
    if (!cost.evaluator()->is_convex()) {
      return true;
    }
  }
  for (const auto& constraint : prog.quadratic_constraints()) {
    if (!constraint.evaluator()->is_convex()) {
      return true;
    }
  }
  return false;
}

// Constructs the semidefinite relaxation of the program prog and adds it to
// relaxation. We assume that the program attributes of prog are already
// validated and that relaxation already contains all the variables and
// constraints of prog. The variable one is already constrained to be equal to
// one. This is passed so it can be re-used across semidefinite variables in the
// sparse version of MakeSemidefiniteRelaxation. Returns the X matrix of the
// semidefinite relaxation.
MatrixXDecisionVariable DoAddSemidefiniteVariableAndImpliedCostsAndConstraints(
    const MathematicalProgram& prog, const Variable& one,
    MathematicalProgram* relaxation,
    std::optional<int> group_number = std::nullopt) {
  // Build a symmetric matrix X of decision variables using the original
  // program variables (so that GetSolution, etc, works using the original
  // variables).
  relaxation->AddDecisionVariables(prog.decision_variables());
  MatrixX<Variable> X(prog.num_vars() + 1, prog.num_vars() + 1);
  // X = xxᵀ; x = [prog.decision_vars(); 1].
  std::string name =
      group_number.has_value() ? fmt::format("Y{}", group_number.value()) : "Y";
  X.topLeftCorner(prog.num_vars(), prog.num_vars()) =
      relaxation->NewSymmetricContinuousVariables(prog.num_vars(), name);
  // We sort the variables so that the matrix X is ordered in a predictable way.
  // This makes it easier when using the sparsity groups to make the
  // semidefinite matrices agree.
  VectorX<Variable> sorted_variables = prog.decision_variables();
  std::sort(sorted_variables.data(),
            sorted_variables.data() + sorted_variables.size(),
            std::less<Variable>{});
  X.topRightCorner(prog.num_vars(), 1) = sorted_variables;
  X.bottomLeftCorner(1, prog.num_vars()) = sorted_variables.transpose();

  std::map<Variable, int> variables_to_sorted_indices;
  int ctr = 0;
  for (const auto& v : sorted_variables) {
    variables_to_sorted_indices[v] = ctr++;
  }
  // We create a new map mapping decision variables to the sorted variables,
  // since prog.FindDecisionVariableIndices would now find the wrong index for
  // our needs.
  auto FindDecisionVariableIndices =
      [&variables_to_sorted_indices](const VectorXDecisionVariable& vars) {
        std::vector<int> indices;
        indices.reserve(vars.rows());
        for (const auto& v : vars) {
          indices.emplace_back(variables_to_sorted_indices.at(v));
        }
        return indices;
      };

  // X(-1,-1) = 1.
  X(prog.num_vars(), prog.num_vars()) = one;

  // X ≽ 0.
  relaxation->AddPositiveSemidefiniteConstraint(X);

  auto x = X.col(prog.num_vars());

  // Returns the {a, vars} in relaxation, such that a' vars = 0.5*tr(QY). This
  // assumes Q=Q', which is ensured by QuadraticCost and QuadraticConstraint.
  auto half_trace_QY = [&X, &FindDecisionVariableIndices](
                           const Eigen::MatrixXd& Q,
                           const VectorXDecisionVariable& prog_vars)
      -> std::pair<VectorXd, VectorX<Variable>> {
    const int N = prog_vars.size();
    const int num_vars = N * (N + 1) / 2;
    const std::vector<int> indices = FindDecisionVariableIndices(prog_vars);

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

  // Remove the quadratic cost in relaxation and replace it with a linear cost
  // on the semidefinite variables i.e.
  // 0.5 y'Qy + b'y + c => 0.5 tr(QY) + b'y + c
  for (const auto& binding : prog.quadratic_costs()) {
    relaxation->RemoveCost(binding);
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

  // Remove the quadratic constraints and replace them with a linear  constraint
  // on the semidefinite varaibles i.e.
  // lb ≤ 0.5 y'Qy + b'y ≤ ub => lb ≤ 0.5 tr(QY) + b'y ≤ ub
  for (const auto& binding : prog.quadratic_constraints()) {
    relaxation->RemoveConstraint(binding);
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

  // Now add the implied linear constraints.
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

    std::vector<Triplet<double>> A_triplets;
    A_triplets.reserve(nnz);
    SparseMatrix<double> A(num_constraints, prog.num_vars());
    VectorXd b(num_constraints);

    int constraint_idx = 0;
    for (const auto& binding : prog.bounding_box_constraints()) {
      const std::vector<int> indices =
          FindDecisionVariableIndices(binding.variables());
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
          FindDecisionVariableIndices(binding.variables());
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
  // Ay = b => (Ay-b)yᵀ = Ayyᵀ - byᵀ = 0.
  for (const auto& binding : prog.linear_equality_constraints()) {
    const int N = binding.variables().size();
    const std::vector<int> indices =
        FindDecisionVariableIndices(binding.variables());
    VectorX<Variable> vars(N + 1);
    // Add the constraints one column at a time:
    // Ayx_j - bx_j = 0.
    MatrixX<double> Ab(binding.evaluator()->num_constraints(), N + 1);
    // TODO(Alexandre.Amice) make this only access the sparse matrix.
    Ab.leftCols(N) = binding.evaluator()->GetDenseA();
    Ab.col(N) = -binding.evaluator()->lower_bound();
    // We don't need to do the last column of X.
    for (int j = 0; j < static_cast<int>(x.size()) - 1; ++j) {
      for (int i = 0; i < N; ++i) {
        vars[i] = X(indices[i], j);
      }
      vars[N] = x[j];
      relaxation->AddLinearEqualityConstraint(
          Ab, VectorXd::Zero(binding.evaluator()->num_constraints()), vars);
    }
  }
  return X;
}

}  // namespace

std::unique_ptr<MathematicalProgram> MakeSemidefiniteRelaxation(
    const MathematicalProgram& prog) {
  ValidateProgramIsSupported(prog);
  auto relaxation = prog.Clone();
  const Variable one("one");
  relaxation->AddDecisionVariables(Vector1<Variable>(one));
  relaxation->AddLinearEqualityConstraint(one, 1);
  DoAddSemidefiniteVariableAndImpliedCostsAndConstraints(prog, one,
                                                         relaxation.get());
  return relaxation;
}

std::unique_ptr<MathematicalProgram> MakeSemidefiniteRelaxation(
    const MathematicalProgram& prog,
    const std::vector<symbolic::Variables>& variable_groups) {
  auto relaxation = prog.Clone();
  const Variable one("one");
  relaxation->AddDecisionVariables(Vector1<Variable>(one));
  relaxation->AddLinearEqualityConstraint(one, 1);

  std::map<symbolic::Variables, solvers::MathematicalProgram>
      groups_to_container_programs;
  std::map<symbolic::Variables, MatrixXDecisionVariable>
      groups_to_psd_variables;

  for (const auto& group : variable_groups) {
    groups_to_container_programs.try_emplace(group);
    VectorXDecisionVariable group_vec(group.size());
    int i = 0;
    for (const auto& v : group) {
      group_vec(i) = v;
      ++i;
    }
    groups_to_container_programs.at(group).AddDecisionVariables(group_vec);
  }

  for (const auto& constraint : prog.GetAllConstraints()) {
    const Variables constraint_variables{constraint.variables()};
    for (const auto& group : variable_groups) {
      if (constraint_variables.IsSubsetOf(group)) {
        // There is no need to add constraint_variables to the
        // container_program, since the variables are a subset of the group and
        // therefore already in the program.
        groups_to_container_programs.at(group).AddConstraint(constraint);
      }
    }
  }
  for (const auto& cost : prog.GetAllCosts()) {
    const Variables cost_variables{cost.variables()};
    for (const auto& group : variable_groups) {
      if (cost_variables.IsSubsetOf(group)) {
        groups_to_container_programs.at(group).AddCost(cost);
        // Only add the costs once.
        break;
      }
    }
  }

  int group_number = 0;
  for (const auto& [group, container_program] : groups_to_container_programs) {
    groups_to_psd_variables.emplace(
        group, DoAddSemidefiniteVariableAndImpliedCostsAndConstraints(
                   container_program, one, relaxation.get(), group_number));
    ++group_number;
  }

  // Now constrain the semidefinite variables to agree where they overlap.
  for (auto it = groups_to_psd_variables.begin();
       it != groups_to_psd_variables.end(); it++) {
    for (auto it2 = std::next(it); it2 != groups_to_psd_variables.end();
         it2++) {
      const Variables common_variables = intersect(it->first, it2->first);
      if (!common_variables.empty()) {
        auto GetSubmatrixOfVariables =
            [&common_variables](const MatrixXDecisionVariable& X) {
              std::set<int> submatrix_indices;
              for (const auto& v : common_variables) {
                for (int i = 0; i < X.rows() - 1; ++i) {
                  if (X(i, X.cols() - 1).equal_to(v)) {
                    submatrix_indices.insert(i);
                    break;
                  }
                }
              }
              return math::ExtractPrincipalSubmatrix(X, submatrix_indices);
            };
        relaxation->AddLinearEqualityConstraint(
            GetSubmatrixOfVariables(it->second) ==
            GetSubmatrixOfVariables(it2->second));
      }
    }
  }

  if (CheckProgramRequireSemidefiniteRelaxation(*relaxation)) {
    throw std::runtime_error(
        "There is a non-convex cost or constraint in the program whose "
        "variables do not overlap with any variable groups. Therefore, these "
        "costs or constraints would not be converted to convex, semidefinite "
        "constraints and so the returned program would not be convex. Consider "
        "further specifying the variable groups.");
  }

  return relaxation;
}

}  // namespace solvers
}  // namespace drake
