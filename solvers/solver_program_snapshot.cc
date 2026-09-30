#include "drake/solvers/solver_program_snapshot.h"

#include <utility>

namespace drake {
namespace solvers {
namespace internal {
namespace {
template <typename DerivedA, typename DerivedB>
bool Equal(const Eigen::MatrixBase<DerivedA>& a,
           const Eigen::MatrixBase<DerivedB>& b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         (a.array() == b.array()).all();
}

bool Equal(const Eigen::SparseMatrix<double>& a,
           const Eigen::SparseMatrix<double>& b) {
  if (a.rows() != b.rows() || a.cols() != b.cols() ||
      a.nonZeros() != b.nonZeros())
    return false;
  for (int col = 0; col < a.outerSize(); ++col) {
    Eigen::SparseMatrix<double>::InnerIterator i(a, col), j(b, col);
    for (; i && j; ++i, ++j) {
      if (i.row() != j.row() || i.value() != j.value()) return false;
    }
    if (i || j) return false;
  }
  return true;
}
}  // namespace

SolverBindingSnapshot::SolverBindingSnapshot(Binding<EvaluatorBase> binding_in,
                                             const MathematicalProgram& prog)
    : binding(std::move(binding_in)),
      variable_indices(prog.FindDecisionVariableIndices(binding.variables())) {
  const auto* e = binding.evaluator().get();
  recognized = current_A() || current_Q() || current_v() ||
               dynamic_cast<const PositiveSemidefiniteConstraint*>(e) ||
               dynamic_cast<const LinearMatrixInequalityConstraint*>(e) ||
               dynamic_cast<const ExponentialConeConstraint*>(e);
  Refresh();
}

const Eigen::SparseMatrix<double>* SolverBindingSnapshot::current_A() const {
  const auto* e = binding.evaluator().get();
  if (const auto* c = dynamic_cast<const LinearConstraint*>(e))
    return &c->get_sparse_A();
  if (const auto* c = dynamic_cast<const LorentzConeConstraint*>(e))
    return &c->A();
  if (const auto* c = dynamic_cast<const RotatedLorentzConeConstraint*>(e))
    return &c->A();
  if (const auto* c = dynamic_cast<const L2NormCost*>(e))
    return &c->get_sparse_A();
  return nullptr;
}

const Eigen::MatrixXd* SolverBindingSnapshot::current_Q() const {
  if (const auto* c =
          dynamic_cast<const QuadraticCost*>(binding.evaluator().get()))
    return &c->Q();
  return nullptr;
}

const Eigen::VectorXd* SolverBindingSnapshot::current_v() const {
  const auto* e = binding.evaluator().get();
  if (const auto* c = dynamic_cast<const LinearCost*>(e)) return &c->a();
  if (const auto* c = dynamic_cast<const QuadraticCost*>(e)) return &c->b();
  if (const auto* c = dynamic_cast<const LinearConstraint*>(e))
    return &c->lower_bound();
  if (const auto* c = dynamic_cast<const LorentzConeConstraint*>(e))
    return &c->b();
  if (const auto* c = dynamic_cast<const RotatedLorentzConeConstraint*>(e))
    return &c->b();
  if (const auto* c = dynamic_cast<const L2NormCost*>(e)) return &c->b();
  return nullptr;
}

const Eigen::VectorXd* SolverBindingSnapshot::current_w() const {
  if (const auto* c =
          dynamic_cast<const LinearConstraint*>(binding.evaluator().get()))
    return &c->upper_bound();
  return nullptr;
}

double SolverBindingSnapshot::current_constant() const {
  const auto* e = binding.evaluator().get();
  if (const auto* c = dynamic_cast<const LinearCost*>(e)) return c->b();
  if (const auto* c = dynamic_cast<const QuadraticCost*>(e)) return c->c();
  return 0;
}

bool SolverBindingSnapshot::MatrixMatches() const {
  if (!recognized) return false;
  if (const auto* a = current_A(); a && !Equal(A, *a)) {
    return false;
  }
  if (const auto* q = current_Q(); q && !Equal(Q, *q)) {
    return false;
  }
  return true;
}

bool SolverBindingSnapshot::VectorsMatch() const {
  if (!recognized || constant != current_constant()) return false;
  if (const auto* b = current_v(); b && !Equal(v, *b)) {
    return false;
  }
  if (const auto* b = current_w(); b && !Equal(w, *b)) {
    return false;
  }
  return true;
}

bool SolverBindingSnapshot::DimensionsMatch() const {
  if (const auto* a = current_A();
      a && (a->rows() != A.rows() || a->cols() != A.cols())) {
    return false;
  }
  if (const auto* q = current_Q();
      q && (q->rows() != Q.rows() || q->cols() != Q.cols())) {
    return false;
  }
  if (const auto* b = current_v(); b && b->size() != v.size()) {
    return false;
  }
  if (const auto* b = current_w(); b && b->size() != w.size()) {
    return false;
  }
  return true;
}

void SolverBindingSnapshot::Refresh() {
  if (const auto* a = current_A()) A = *a;
  if (const auto* q = current_Q()) Q = *q;
  RefreshVectors();
}

void SolverBindingSnapshot::RefreshVectors() {
  if (const auto* b = current_v()) v = *b;
  if (const auto* b = current_w()) w = *b;
  constant = current_constant();
}

SolverProgramSnapshot::SolverProgramSnapshot(const MathematicalProgram& prog)
    : variables_(prog.decision_variables()),
      scaling_(prog.GetVariableScaling()) {
  for (const auto& binding : prog.GetAllCosts())
    costs.emplace_back(binding, prog);
  for (const auto& binding : prog.GetAllConstraints())
    constraints.emplace_back(binding, prog);
}

std::string SolverProgramSnapshot::CheckStructure(
    const MathematicalProgram& prog) const {
  if (variables_.size() != prog.num_vars()) return "decision variables changed";
  for (int i = 0; i < variables_.size(); ++i) {
    if (!variables_(i).equal_to(prog.decision_variables()(i)))
      return "decision variables changed";
  }
  if (scaling_ != prog.GetVariableScaling()) return "variable scaling changed";
  const auto check = [](const auto& old, const auto&... groups) {
    size_t index = 0;
    bool matches = true;
    const auto visit = [&](const auto& group) {
      for (const auto& current : group) {
        if (index >= old.size()) {
          matches = false;
          break;
        }
        const auto& previous = old[index++];
        if (previous.binding.evaluator().get() != current.evaluator().get() ||
            !previous.DimensionsMatch()) {
          matches = false;
          continue;
        }
        const auto& a = previous.binding.variables();
        const auto& b = current.variables();
        if (a.size() != b.size()) {
          matches = false;
          continue;
        }
        for (int j = 0; j < a.size(); ++j) {
          if (!a(j).equal_to(b(j))) matches = false;
        }
      }
    };
    (visit(groups), ...);
    return matches && index == old.size();
  };
  // Match GetAllCosts/GetAllConstraints order, but avoid allocating temporary
  // Binding vectors and copying every binding's variable vector on each solve.
  if (!check(costs, prog.generic_costs(), prog.linear_costs(),
             prog.quadratic_costs(), prog.l2norm_costs()) ||
      !check(constraints, prog.generic_constraints(),
             prog.quadratic_constraints(), prog.linear_constraints(),
             prog.linear_equality_constraints(),
             prog.bounding_box_constraints(), prog.lorentz_cone_constraints(),
             prog.rotated_lorentz_cone_constraints(),
             prog.linear_matrix_inequality_constraints(),
             prog.positive_semidefinite_constraints(),
             prog.linear_complementarity_constraints(),
             prog.exponential_cone_constraints()))
    return "bindings changed";
  return {};
}

bool SolverProgramSnapshot::IsUnchanged() const {
  for (const auto* entries : {&costs, &constraints}) {
    for (const auto& entry : *entries) {
      if (!entry.MatrixMatches() || !entry.VectorsMatch()) return false;
    }
  }
  return true;
}

}  // namespace internal
}  // namespace solvers
}  // namespace drake
