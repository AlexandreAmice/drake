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
// Overload resolution uses the type of the program's binding collection.
// Generic evaluator collections remain conservative even if their evaluator
// happens to implement one of the specialized families.
SolverBindingCoefficients ReadCoefficients(const EvaluatorBase&) {
  return {.recognized = false};
}
SolverBindingCoefficients ReadCoefficients(const LinearCost& c) {
  return {.v = &c.a(), .constant = c.b()};
}
SolverBindingCoefficients ReadCoefficients(const QuadraticCost& c) {
  return {.Q = &c.Q(), .v = &c.b(), .constant = c.c()};
}
SolverBindingCoefficients ReadCoefficients(const LinearConstraint& c) {
  return {.A = &c.get_sparse_A(), .v = &c.lower_bound(), .w = &c.upper_bound()};
}
SolverBindingCoefficients ReadCoefficients(const LorentzConeConstraint& c) {
  return {.A = &c.A(), .v = &c.b()};
}
SolverBindingCoefficients ReadCoefficients(
    const RotatedLorentzConeConstraint& c) {
  return {.A = &c.A(), .v = &c.b()};
}
SolverBindingCoefficients ReadCoefficients(const L2NormCost& c) {
  return {.A = &c.get_sparse_A(), .v = &c.b()};
}
SolverBindingCoefficients ReadCoefficients(
    const PositiveSemidefiniteConstraint&) {
  return {};
}
SolverBindingCoefficients ReadCoefficients(
    const LinearMatrixInequalityConstraint&) {
  return {};
}
SolverBindingCoefficients ReadCoefficients(const ExponentialConeConstraint&) {
  return {};
}

template <typename Evaluator>
class TypedCoefficientReader final : public SolverCoefficientReader {
 public:
  explicit TypedCoefficientReader(const Evaluator* evaluator)
      : evaluator_(evaluator) {}
  SolverBindingCoefficients Read() const final {
    return ReadCoefficients(*evaluator_);
  }

 private:
  // The snapshot's Binding owns the evaluator for this reader's lifetime.
  const Evaluator* const evaluator_;
};

template <typename Evaluator>
void AppendSnapshots(const std::vector<Binding<Evaluator>>& bindings,
                     const MathematicalProgram& prog,
                     std::vector<SolverBindingSnapshot>* snapshots) {
  for (const auto& binding : bindings) {
    snapshots->emplace_back(binding, prog,
                            std::make_unique<TypedCoefficientReader<Evaluator>>(
                                binding.evaluator().get()));
  }
}
}  // namespace

SolverBindingSnapshot::SolverBindingSnapshot(
    Binding<EvaluatorBase> binding_in, const MathematicalProgram& prog,
    std::unique_ptr<SolverCoefficientReader> reader)
    : binding(std::move(binding_in)),
      variable_indices(prog.FindDecisionVariableIndices(binding.variables())),
      reader_(std::move(reader)) {
  recognized = reader_->Read().recognized;
  Refresh();
}

const Eigen::SparseMatrix<double>* SolverBindingSnapshot::current_A() const {
  return reader_->Read().A;
}
const Eigen::MatrixXd* SolverBindingSnapshot::current_Q() const {
  return reader_->Read().Q;
}
const Eigen::VectorXd* SolverBindingSnapshot::current_v() const {
  return reader_->Read().v;
}
const Eigen::VectorXd* SolverBindingSnapshot::current_w() const {
  return reader_->Read().w;
}
double SolverBindingSnapshot::current_constant() const {
  return reader_->Read().constant;
}

bool SolverBindingSnapshot::MatrixMatches() const {
  const auto current = reader_->Read();
  if (!recognized) return false;
  if (const auto* a = current.A; a && !Equal(A, *a)) {
    return false;
  }
  if (const auto* q = current.Q; q && !Equal(Q, *q)) {
    return false;
  }
  return true;
}

bool SolverBindingSnapshot::VectorsMatch() const {
  const auto current = reader_->Read();
  if (!recognized || constant != current.constant) return false;
  if (const auto* b = current.v; b && !Equal(v, *b)) {
    return false;
  }
  if (const auto* b = current.w; b && !Equal(w, *b)) {
    return false;
  }
  return true;
}

bool SolverBindingSnapshot::DimensionsMatch() const {
  const auto current = reader_->Read();
  if (const auto* a = current.A;
      a && (a->rows() != A.rows() || a->cols() != A.cols())) {
    return false;
  }
  if (const auto* q = current.Q;
      q && (q->rows() != Q.rows() || q->cols() != Q.cols())) {
    return false;
  }
  if (const auto* b = current.v; b && b->size() != v.size()) {
    return false;
  }
  if (const auto* b = current.w; b && b->size() != w.size()) {
    return false;
  }
  return true;
}

void SolverBindingSnapshot::Refresh() {
  const auto current = reader_->Read();
  if (const auto* a = current.A) A = *a;
  if (const auto* q = current.Q) Q = *q;
  RefreshVectors();
}

void SolverBindingSnapshot::RefreshVectors() {
  const auto current = reader_->Read();
  if (const auto* b = current.v) v = *b;
  if (const auto* b = current.w) w = *b;
  constant = current.constant;
}

SolverProgramSnapshot::SolverProgramSnapshot(const MathematicalProgram& prog)
    : variables_(prog.decision_variables()),
      scaling_(prog.GetVariableScaling()) {
  AppendSnapshots(prog.generic_costs(), prog, &costs);
  AppendSnapshots(prog.linear_costs(), prog, &costs);
  AppendSnapshots(prog.quadratic_costs(), prog, &costs);
  AppendSnapshots(prog.l2norm_costs(), prog, &costs);
  AppendSnapshots(prog.generic_constraints(), prog, &constraints);
  AppendSnapshots(prog.quadratic_constraints(), prog, &constraints);
  AppendSnapshots(prog.linear_constraints(), prog, &constraints);
  AppendSnapshots(prog.linear_equality_constraints(), prog, &constraints);
  AppendSnapshots(prog.bounding_box_constraints(), prog, &constraints);
  AppendSnapshots(prog.lorentz_cone_constraints(), prog, &constraints);
  AppendSnapshots(prog.rotated_lorentz_cone_constraints(), prog, &constraints);
  AppendSnapshots(prog.linear_matrix_inequality_constraints(), prog,
                  &constraints);
  AppendSnapshots(prog.positive_semidefinite_constraints(), prog, &constraints);
  AppendSnapshots(prog.linear_complementarity_constraints(), prog,
                  &constraints);
  AppendSnapshots(prog.exponential_cone_constraints(), prog, &constraints);
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
