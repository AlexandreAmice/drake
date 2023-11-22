#include "drake/solvers/semidefinite_relaxation_internal.h"


namespace drake {
namespace solvers {
namespace internal {

// Creates the linear constraints in semidefinite relaxation and computes which
// variables appear in constraints together. This does NOT add the semidefinite
// constraint on the aggregated variables. That must occur afterwards. Returns
// the semidefinite relaxation mathematical program without the semidefinite
// constraint added, as well as the variable X. Throughout this method use y =
// prog.decision_vars(), x = [y, 1], Y = yyᵀ, and X = xxᵀ.
std::pair<std::unique_ptr<MathematicalProgram>, MatrixX<Variable>>
MakeSemidefiniteRelaxationLinearConstraints(
    const MathematicalProgram& prog,
    std::optional<std::set<symbolic::Variables>*> variable_dependence_cliques) {
  auto add_to_computed_sparsity_group =
      [&variable_dependence_cliques](const Variables& variables) {
        if (variable_dependence_cliques.has_value()) {
          InsertIfNotSubsetOrReplaceIfSuperset(
              variables, variable_dependence_cliques.value());
        }
      };
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
  //
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
    if (variable_dependence_cliques.has_value()) {
      Variables cur_vars;
      for (int k = 0; k < binding.evaluator()->a().size(); ++k) {
        if (binding.evaluator()->a()(k) != 0) {
          cur_vars.insert(binding.variables()(k));
        }
      }
      if (binding.evaluator()->b() != 0) {
        cur_vars.insert(one);
      }
      add_to_computed_sparsity_group(cur_vars);
    }
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

    Variables relaxation_vars;
    for (int i = 0; i < a.size(); ++i) {
      if (a(i) != 0) {
        relaxation_vars.insert(vars(i));
      }
    }
    if (binding.evaluator()->c() != 0) {
      relaxation_vars.insert(one);
    }
    add_to_computed_sparsity_group(relaxation_vars);
  }

  // Bounding Box constraints
  // lb ≤ y ≤ ub => lb ≤ y ≤ ub
  for (const auto& binding : prog.bounding_box_constraints()) {
    relaxation->AddConstraint(binding);
  }

  // Linear constraints
  // lb ≤ Ay ≤ ub => lb ≤ Ay ≤ ub
  for (const auto& binding : prog.linear_constraints()) {
    relaxation->AddConstraint(binding);
  }

  {  // Now assemble one big Ay <= b matrix from all bounding box constraints
    // and linear constraints
    // TODO(bernhardpg): Consider special-casing linear equality constraints
    // TODO(Alexandre.Amice): Use internal::Parse functions in
    // aggregate_costs_and_constraints.h that are added as bounding box or
    // linear constraints with lb == ub
    int num_constraints = 0;
    int nnz = 0;

    // Count the size of A.
    for (const auto& binding : prog.bounding_box_constraints()) {
      for (int i = 0; i < binding.evaluator()->num_constraints(); ++i) {
        bool has_finite_bound{false};
        if (std::isfinite(binding.evaluator()->lower_bound()[i])) {
          ++num_constraints;
          has_finite_bound = true;
        }
        if (std::isfinite(binding.evaluator()->upper_bound()[i])) {
          ++num_constraints;
          has_finite_bound = true;
        }
        Variables cur_vars{binding.variables()};
        if (has_finite_bound) {
          cur_vars.insert(one);
        }
        add_to_computed_sparsity_group(cur_vars);
      }
      nnz += binding.evaluator()->get_sparse_A().nonZeros();
    }

    for (const auto& binding : prog.linear_constraints()) {
      // If we are computing the sparsity groups, then we iterate over the
      // non-zero entries of this linear constraint to find which variables
      // actually interact. Otherwise, avoid this costly iteration in favor of a
      // simpler, more efficient loop.
      if (variable_dependence_cliques.has_value()) {
        std::vector<bool> has_finite_lower_bound(
            binding.evaluator()->num_constraints(), false);
        std::vector<bool> has_finite_upper_bound(
            binding.evaluator()->num_constraints(), false);
        for (int k = 0; k < binding.evaluator()->get_sparse_A().outerSize();
             ++k) {
          Variables local_vars;
          int cur_row{-1};
          for (SparseMatrix<double>::InnerIterator it(
                   binding.evaluator()->get_sparse_A(), k);
               it; ++it) {
            cur_row = static_cast<int>(it.row());
            local_vars.insert(binding.variables()(it.col()));
            has_finite_lower_bound.at(it.row()) =
                has_finite_lower_bound.at(it.row()) ||
                std::isfinite(binding.evaluator()->lower_bound()[it.row()]);

            has_finite_upper_bound.at(it.row()) =
                has_finite_upper_bound.at(it.row()) ||
                std::isfinite(binding.evaluator()->upper_bound()[it.row()]);
          }
          if (has_finite_lower_bound.at(cur_row) ||
              has_finite_upper_bound.at(cur_row)) {
            local_vars.insert(one);
          }
          add_to_computed_sparsity_group(local_vars);
        }
        num_constraints +=
            static_cast<int>(std::count(has_finite_lower_bound.begin(),
                                        has_finite_lower_bound.end(), true));
        num_constraints +=
            static_cast<int>(std::count(has_finite_upper_bound.begin(),
                                        has_finite_upper_bound.end(), true));
      } else {
        for (int i = 0; i < binding.evaluator()->num_constraints(); ++i) {
          if (std::isfinite(binding.evaluator()->lower_bound()[i])) {
            ++num_constraints;
          }
          if (std::isfinite(binding.evaluator()->upper_bound()[i])) {
            ++num_constraints;
          }
        }
      }
      nnz += binding.evaluator()->get_sparse_A().nonZeros();
    }

    std::vector<Triplet<double>> triplet_list;
    triplet_list.reserve(nnz);
    SparseMatrix<double> A(num_constraints, prog.num_vars());
    VectorXd b(num_constraints);

    int constraint_idx = 0;
    for (const auto& binding : prog.bounding_box_constraints()) {
      const std::vector<int> indices =
          prog.FindDecisionVariableIndices(binding.variables());
      for (int i = 0; i < binding.evaluator()->num_constraints(); ++i) {
        if (std::isfinite(binding.evaluator()->lower_bound()[i])) {
          triplet_list.push_back(
              Triplet<double>(constraint_idx, indices[i], -1.0));
          b(constraint_idx++) = -binding.evaluator()->lower_bound()[i];
        }
        if (std::isfinite(binding.evaluator()->upper_bound()[i])) {
          triplet_list.push_back(
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
              triplet_list.push_back(Triplet<double>(
                  constraint_idx, indices[j],
                  -binding.evaluator()->get_sparse_A().coeff(i, j)));
            }
          }
          b(constraint_idx++) = -binding.evaluator()->lower_bound()[i];
        }
        if (std::isfinite(binding.evaluator()->upper_bound()[i])) {
          for (int j = 0; j < binding.evaluator()->num_vars(); ++j) {
            if (binding.evaluator()->get_sparse_A().coeff(i, j) != 0) {
              triplet_list.push_back(Triplet<double>(
                  constraint_idx, indices[j],
                  binding.evaluator()->get_sparse_A().coeff(i, j)));
            }
          }
          b(constraint_idx++) = binding.evaluator()->upper_bound()[i];
        }
      }
    }
    A.setFromTriplets(triplet_list.begin(), triplet_list.end());

    // 0 ≤ (Ay-b)(Ay-b)ᵀ, implemented with
    // -bbᵀ ≤ AYAᵀ - b(Ay)ᵀ - (Ay)bᵀ.
    // TODO(russt): Avoid the symbolic computation here.
    // TODO(russt): Avoid the dense matrix.
    // TODO(russt): Only add the lower triangular constraints
    // (MathematicalProgram::AddLinearEqualityConstraint has this option, but
    // AddLinearConstraint does not yet).
    const MatrixX<Expression> AYAT =
        A * X.topLeftCorner(prog.num_vars(), prog.num_vars()) * A.transpose();
    const VectorX<Variable> y = x.head(prog.num_vars());
    relaxation->AddLinearConstraint(
        AYAT - b * (A * y).transpose() - A * y * b.transpose(),
        -b * b.transpose(),
        MatrixXd::Constant(num_constraints, num_constraints, kInf));
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
      add_to_computed_sparsity_group(Variables(vars));
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

    Variables relaxation_vars{binding.variables()};
    if (!binding.evaluator()->upper_bound().isZero() ||
        !binding.evaluator()->lower_bound().isZero()) {
      relaxation_vars.insert(one);
    }
    add_to_computed_sparsity_group(relaxation_vars);
  }

  return std::make_pair(std::move(relaxation), X);
}
}  // namespace internal

std::unique_ptr<MathematicalProgram> MakeSemidefiniteRelaxation(
    const MathematicalProgram& prog,
    const SemidefiniteRelaxationSparsityType& sparsity) {
  std::set<symbolic::Variables> computed_sparsity_groups;
  auto prog_and_X = internal::MakeSemidefiniteRelaxationLinearConstraints(
      prog, sparsity != SemidefiniteRelaxationSparsityType::kDense
                ? std::optional{&computed_sparsity_groups}
                : std::nullopt);
  switch (sparsity) {
    case kDense:
      prog_and_X.first->AddPositiveSemidefiniteConstraint(prog_and_X.second);
      break;
    case kTermSparse:
      AddMinorsArePsdConstraints(prog_and_X.second, computed_sparsity_groups,
                                 prog_and_X.first.get());
      break;
      DRAKE_UNREACHABLE();
  }
  return std::move(prog_and_X.first);
}

std::unique_ptr<MathematicalProgram> MakeSemidefiniteRelaxation(
    const MathematicalProgram& prog,
    const std::map<symbolic::Variables, bool>& variables_to_enforce_sparsity) {
  auto prog_and_X =
      internal::MakeSemidefiniteRelaxationLinearConstraints(prog, std::nullopt);
  std::set<symbolic::Variables> sparsity_to_apply;
  const Variables prog_vars{prog.decision_variables()};
  for (const auto& [vars, use_one] : variables_to_enforce_sparsity) {
    Variables local_vars{vars};
    DRAKE_THROW_UNLESS(local_vars.IsSubsetOf(prog_vars));
    if (use_one) {
      local_vars.insert(prog_and_X.second.bottomRightCorner<1, 1>()(0));
    }
    InsertIfNotSubsetOrReplaceIfSuperset(local_vars, &sparsity_to_apply);
  };
  //  for (const auto& vars : sparsity_to_apply) {
  //    std::cout << vars << std::endl;
  //  }
  AddMinorsArePsdConstraints(prog_and_X.second, sparsity_to_apply,
                             prog_and_X.first.get());
  return std::move(prog_and_X.first);
};

}  // namespace internal
}  // namespace solvers
}  // namespace drake