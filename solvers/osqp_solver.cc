#include "drake/solvers/osqp_solver.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <osqp.h>

#include "drake/common/scope_exit.h"
#include "drake/common/text_logging.h"
#include "drake/math/eigen_sparse_triplet.h"
#include "drake/solvers/aggregate_costs_constraints.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solver_cache_options.h"
#include "drake/solvers/solver_cache_profiler.h"
#include "drake/solvers/solver_program_snapshot.h"
#include "drake/solvers/solver_result_access.h"
#include "drake/solvers/solver_sparse_update.h"
#include "drake/solvers/specific_options.h"

// This function must appear in the global namespace -- the Serialize pattern
// uses ADL (argument-dependent lookup) and the namespace for the OSQPSettings
// struct is the global namespace. (We can't even use an anonymous namespace!)
static void Serialize(
    drake::solvers::internal::SpecificOptions* archive,
    // NOLINTNEXTLINE(runtime/references) to match Serialize concept.
    OSQPSettings& settings) {
  using drake::MakeNameValue;
  archive->Visit(MakeNameValue("device", &settings.device));
  // TODO(jwnimmer-tri) Add me here:
  // enum osqp_linsys_solver_type linsys_solver
  archive->Visit(MakeNameValue("allocate_solution",  // BR
                               &settings.allocate_solution));
  archive->Visit(MakeNameValue("verbose", &settings.verbose));
  archive->Visit(MakeNameValue("profiler_level", &settings.profiler_level));
  archive->Visit(MakeNameValue("warm_starting", &settings.warm_starting));
  archive->Visit(MakeNameValue("scaling", &settings.scaling));
  archive->Visit(MakeNameValue("polishing", &settings.polishing));
  archive->Visit(MakeNameValue("rho", &settings.rho));
  archive->Visit(MakeNameValue("rho_is_vec", &settings.rho_is_vec));
  archive->Visit(MakeNameValue("sigma", &settings.sigma));
  archive->Visit(MakeNameValue("alpha", &settings.alpha));
  archive->Visit(MakeNameValue("cg_max_iter", &settings.cg_max_iter));
  archive->Visit(MakeNameValue("cg_tol_reduction", &settings.cg_tol_reduction));
  archive->Visit(MakeNameValue("cg_tol_fraction", &settings.cg_tol_fraction));
  // TODO(jwnimmer-tri) Add me here:
  // osqp_precond_type cg_precond
  archive->Visit(MakeNameValue("adaptive_rho", &settings.adaptive_rho));
  archive->Visit(MakeNameValue("adaptive_rho_interval",  // BR
                               &settings.adaptive_rho_interval));
  archive->Visit(MakeNameValue("adaptive_rho_fraction",  // BR
                               &settings.adaptive_rho_fraction));
  archive->Visit(MakeNameValue("adaptive_rho_tolerance",
                               &settings.adaptive_rho_tolerance));
  archive->Visit(MakeNameValue("max_iter", &settings.max_iter));
  archive->Visit(MakeNameValue("eps_abs", &settings.eps_abs));
  archive->Visit(MakeNameValue("eps_rel", &settings.eps_rel));
  archive->Visit(MakeNameValue("eps_prim_inf", &settings.eps_prim_inf));
  archive->Visit(MakeNameValue("eps_dual_inf", &settings.eps_dual_inf));
  archive->Visit(MakeNameValue("scaled_termination",  // BR
                               &settings.scaled_termination));
  archive->Visit(MakeNameValue("check_termination",  // BR
                               &settings.check_termination));
  archive->Visit(MakeNameValue("check_dualgap", &settings.check_dualgap));
  archive->Visit(MakeNameValue("time_limit", &settings.time_limit));
  archive->Visit(MakeNameValue("delta", &settings.delta));
  archive->Visit(MakeNameValue("polish_refine_iter",  // BR
                               &settings.polish_refine_iter));
}

namespace drake {
namespace solvers {
using internal::SolverCachePhase;
using internal::SolverCachePhaseScope;

namespace {

void ParseQuadraticCosts(const MathematicalProgram& prog,
                         Eigen::SparseMatrix<OSQPFloat>* P_upper,
                         std::vector<OSQPFloat>* q,
                         double* constant_cost_term) {
  DRAKE_ASSERT(static_cast<int>(q->size()) == prog.num_vars());
  std::vector<Eigen::Triplet<OSQPFloat>> P_upper_triplets;
  internal::ParseQuadraticCosts(prog, &P_upper_triplets, q, constant_cost_term);
  // Scale the matrix P in the cost.
  // Note that the linear term is scaled in ParseLinearCosts().
  const auto& scale_map = prog.GetVariableScaling();
  if (!scale_map.empty()) {
    for (auto& triplet : P_upper_triplets) {
      // Column
      const auto column = scale_map.find(triplet.col());
      if (column != scale_map.end()) {
        triplet = Eigen::Triplet<double>(triplet.row(), triplet.col(),
                                         triplet.value() * (column->second));
      }
      // Row
      const auto row = scale_map.find(triplet.row());
      if (row != scale_map.end()) {
        triplet = Eigen::Triplet<double>(triplet.row(), triplet.col(),
                                         triplet.value() * (row->second));
      }
    }
  }

  P_upper->resize(prog.num_vars(), prog.num_vars());
  P_upper->setFromTriplets(P_upper_triplets.begin(), P_upper_triplets.end());
}

void ParseLinearCosts(const MathematicalProgram& prog,
                      std::vector<OSQPFloat>* q, double* constant_cost_term) {
  // Add the linear costs to the osqp cost.
  DRAKE_ASSERT(static_cast<int>(q->size()) == prog.num_vars());
  internal::ParseLinearCosts(prog, q, constant_cost_term);

  // Scale the vector q in the cost.
  const auto& scale_map = prog.GetVariableScaling();
  if (!scale_map.empty()) {
    for (const auto& [index, scale] : scale_map) {
      q->at(index) *= scale;
    }
  }
}

// OSQP defines its own infinity in osqp/include/glob_opts.h.
OSQPFloat ConvertInfinity(double val) {
  if (std::isinf(val)) {
    if (val > 0) {
      return OSQP_INFTY;
    }
    return -OSQP_INFTY;
  }
  return static_cast<OSQPFloat>(val);
}

// Will call this function to parse both LinearConstraint and
// LinearEqualityConstraint.
template <typename C>
void ParseLinearConstraints(
    const MathematicalProgram& prog,
    const std::vector<Binding<C>>& linear_constraints,
    std::vector<Eigen::Triplet<OSQPFloat>>* A_triplets,
    std::vector<OSQPFloat>* l, std::vector<OSQPFloat>* u, int* num_A_rows,
    std::unordered_map<Binding<Constraint>, int, internal::BindingHash,
                       internal::BindingEqual>* constraint_start_row) {
  // Loop over the linear constraints, stack them to get l, u and A.
  for (const auto& constraint : linear_constraints) {
    const std::vector<int> x_indices =
        prog.FindDecisionVariableIndices(constraint.variables());
    const std::vector<Eigen::Triplet<double>> Ai_triplets =
        math::SparseMatrixToTriplets(constraint.evaluator()->get_sparse_A());
    const Binding<Constraint> constraint_cast =
        internal::BindingDynamicCast<Constraint>(constraint);
    constraint_start_row->emplace(constraint_cast, *num_A_rows);
    // Append constraint.A to osqp A.
    for (const auto& Ai_triplet : Ai_triplets) {
      A_triplets->emplace_back(*num_A_rows + Ai_triplet.row(),
                               x_indices[Ai_triplet.col()],
                               static_cast<OSQPFloat>(Ai_triplet.value()));
    }
    const int num_Ai_rows = constraint.evaluator()->num_constraints();
    l->reserve(l->size() + num_Ai_rows);
    u->reserve(u->size() + num_Ai_rows);
    for (int i = 0; i < num_Ai_rows; ++i) {
      l->push_back(ConvertInfinity(constraint.evaluator()->lower_bound()(i)));
      u->push_back(ConvertInfinity(constraint.evaluator()->upper_bound()(i)));
    }
    *num_A_rows += num_Ai_rows;
  }
}

void ParseBoundingBoxConstraints(
    const MathematicalProgram& prog,
    std::vector<Eigen::Triplet<OSQPFloat>>* A_triplets,
    std::vector<OSQPFloat>* l, std::vector<OSQPFloat>* u, int* num_A_rows,
    std::unordered_map<Binding<Constraint>, int, internal::BindingHash,
                       internal::BindingEqual>* constraint_start_row) {
  // Loop over the linear constraints, stack them to get l, u and A.
  for (const auto& constraint : prog.bounding_box_constraints()) {
    const Binding<Constraint> constraint_cast =
        internal::BindingDynamicCast<Constraint>(constraint);
    constraint_start_row->emplace(constraint_cast, *num_A_rows);
    // Append constraint.A to osqp A.
    for (int i = 0; i < static_cast<int>(constraint.GetNumElements()); ++i) {
      A_triplets->emplace_back(
          *num_A_rows + i,
          prog.FindDecisionVariableIndex(constraint.variables()(i)),
          static_cast<OSQPFloat>(1));
    }
    const int num_Ai_rows = constraint.evaluator()->num_constraints();
    l->reserve(l->size() + num_Ai_rows);
    u->reserve(u->size() + num_Ai_rows);
    for (int i = 0; i < num_Ai_rows; ++i) {
      l->push_back(ConvertInfinity(constraint.evaluator()->lower_bound()(i)));
      u->push_back(ConvertInfinity(constraint.evaluator()->upper_bound()(i)));
    }
    *num_A_rows += num_Ai_rows;
  }
}

void ParseAllLinearConstraints(
    const MathematicalProgram& prog, Eigen::SparseMatrix<OSQPFloat>* A,
    std::vector<OSQPFloat>* l, std::vector<OSQPFloat>* u,
    std::unordered_map<Binding<Constraint>, int, internal::BindingHash,
                       internal::BindingEqual>* constraint_start_row) {
  std::vector<Eigen::Triplet<OSQPFloat>> A_triplets;
  l->clear();
  u->clear();
  int num_A_rows = 0;
  ParseLinearConstraints(prog, prog.linear_constraints(), &A_triplets, l, u,
                         &num_A_rows, constraint_start_row);
  ParseLinearConstraints(prog, prog.linear_equality_constraints(), &A_triplets,
                         l, u, &num_A_rows, constraint_start_row);
  ParseBoundingBoxConstraints(prog, &A_triplets, l, u, &num_A_rows,
                              constraint_start_row);

  // Scale the matrix A.
  // Note that we only scale the columns of A, because the constraint has the
  // form l <= Ax <= u where the scaling of x enters the columns of A instead of
  // rows of A.
  const auto& scale_map = prog.GetVariableScaling();
  if (!scale_map.empty()) {
    for (auto& triplet : A_triplets) {
      auto column = scale_map.find(triplet.col());
      if (column != scale_map.end()) {
        triplet = Eigen::Triplet<double>(triplet.row(), triplet.col(),
                                         triplet.value() * (column->second));
      }
    }
  }

  A->resize(num_A_rows, prog.num_vars());
  A->setFromTriplets(A_triplets.begin(), A_triplets.end());
}

// Convert an Eigen::SparseMatrix to csc_matrix, to be used by osqp.
// Make sure the input Eigen sparse matrix is compressed, by calling
// makeCompressed() function.
// The caller of this function is responsible for freeing the memory allocated
// here.
OSQPCscMatrix* EigenSparseToCSC(const Eigen::SparseMatrix<OSQPFloat>& mat) {
  // A csc matrix is in the compressed column major.
  OSQPFloat* values =
      static_cast<OSQPFloat*>(malloc(sizeof(OSQPFloat) * mat.nonZeros()));
  OSQPInt* inner_indices =
      static_cast<OSQPInt*>(malloc(sizeof(OSQPInt) * mat.nonZeros()));
  OSQPInt* outer_indices =
      static_cast<OSQPInt*>(malloc(sizeof(OSQPInt) * (mat.cols() + 1)));
  for (int i = 0; i < mat.nonZeros(); ++i) {
    values[i] = *(mat.valuePtr() + i);
    inner_indices[i] = static_cast<OSQPInt>(*(mat.innerIndexPtr() + i));
  }
  for (int i = 0; i < mat.cols() + 1; ++i) {
    outer_indices[i] = static_cast<OSQPInt>(*(mat.outerIndexPtr() + i));
  }
  OSQPCscMatrix* result =
      OSQPCscMatrix_new(mat.rows(), mat.cols(), mat.nonZeros(), values,
                        inner_indices, outer_indices);
  result->owned = 1;
  return result;
}

bool SameSettings(const OSQPSettings& a, const OSQPSettings& b) {
  return a.device == b.device && a.allocate_solution == b.allocate_solution &&
         a.verbose == b.verbose && a.profiler_level == b.profiler_level &&
         a.warm_starting == b.warm_starting && a.scaling == b.scaling &&
         a.polishing == b.polishing && a.rho == b.rho &&
         a.rho_is_vec == b.rho_is_vec && a.sigma == b.sigma &&
         a.alpha == b.alpha && a.cg_max_iter == b.cg_max_iter &&
         a.cg_tol_reduction == b.cg_tol_reduction &&
         a.cg_tol_fraction == b.cg_tol_fraction &&
         a.adaptive_rho == b.adaptive_rho &&
         a.adaptive_rho_interval == b.adaptive_rho_interval &&
         a.adaptive_rho_fraction == b.adaptive_rho_fraction &&
         a.adaptive_rho_tolerance == b.adaptive_rho_tolerance &&
         a.max_iter == b.max_iter && a.eps_abs == b.eps_abs &&
         a.eps_rel == b.eps_rel && a.eps_prim_inf == b.eps_prim_inf &&
         a.eps_dual_inf == b.eps_dual_inf &&
         a.scaled_termination == b.scaled_termination &&
         a.check_termination == b.check_termination &&
         a.check_dualgap == b.check_dualgap && a.time_limit == b.time_limit &&
         a.delta == b.delta && a.polish_refine_iter == b.polish_refine_iter;
}

// Own both Drake's translation buffers and the native solver workspace.
struct OsqpProblemData final : SolverDataCache {
  bool DoSolve(const MathematicalProgram& prog, const Eigen::VectorXd& guess,
               internal::SpecificOptions* options,
               MathematicalProgramResult* result) final {
    SolveWithCache(prog, guess, options, result, this);
    return true;
  }
  static void SolveWithCache(const MathematicalProgram& prog,
                             const Eigen::VectorXd& initial_guess,
                             internal::SpecificOptions* options,
                             MathematicalProgramResult* result,
                             OsqpProblemData* cached);

  explicit OsqpProblemData(const MathematicalProgram& prog, bool retain)
      : SolverDataCache(prog, OsqpSolver::id()), q(prog.num_vars(), 0) {
    if (retain) snapshot.emplace(prog);
    ParseQuadraticCosts(prog, &P_upper_sparse, &q, &constant_cost_term);
    ParseLinearCosts(prog, &q, &constant_cost_term);
    ParseAllLinearConstraints(prog, &A_sparse, &l, &u, &constraint_start_row);
    if (retain) InitializeMatrixContributions(prog);
  }

  ~OsqpProblemData() {
    if (solver != nullptr) osqp_cleanup(solver);
  }

  OSQPInt Setup() {
    const auto* P = EigenSparseToCSC(P_upper_sparse);
    const auto* A = EigenSparseToCSC(A_sparse);
    ScopeExit guard([P, A]() {
      OSQPCscMatrix_free(const_cast<OSQPCscMatrix*>(P));
      OSQPCscMatrix_free(const_cast<OSQPCscMatrix*>(A));
    });
    SolverCachePhaseScope phase(SolverCachePhase::kSetup);
    return osqp_setup(&solver, P, q.data(), A, l.data(), u.data(),
                      A_sparse.rows(), q.size(), &settings);
  }

  // Find existing CSC slots without inserting entries or changing sparsity.
  static int FindEntry(const Eigen::SparseMatrix<OSQPFloat>& matrix, int row,
                       int col) {
    const auto* begin = matrix.innerIndexPtr() + matrix.outerIndexPtr()[col];
    const auto* end = matrix.innerIndexPtr() + matrix.outerIndexPtr()[col + 1];
    const auto* found = std::lower_bound(begin, end, row);
    return found != end && *found == row ? found - matrix.innerIndexPtr() : -1;
  }

  struct Contribution {
    const internal::SolverBindingSnapshot* binding;
    int row, col;
    double factor;
  };

  void InitializeMatrixContributions(const MathematicalProgram& prog) {
    pending_p_changes = internal::SolverSparseUpdate(P_upper_sparse.nonZeros());
    pending_a_changes = internal::SolverSparseUpdate(A_sparse.nonZeros());
    p_contributions.resize(P_upper_sparse.nonZeros());
    a_contributions.resize(A_sparse.nonZeros());
    const auto scale = [&](int variable) {
      const auto it = prog.GetVariableScaling().find(variable);
      return it == prog.GetVariableScaling().end() ? 1.0 : it->second;
    };
    for (const auto& entry : snapshot->costs) {
      const auto* Q = entry.current_Q();
      if (!Q) continue;
      for (int col = 0; col < Q->cols(); ++col) {
        for (int row = 0; row <= col; ++row) {
          const int vi = entry.variable_indices[row];
          const int vj = entry.variable_indices[col];
          const int slot =
              FindEntry(P_upper_sparse, std::min(vi, vj), std::max(vi, vj));
          if (slot < 0) continue;
          const double factor = row != col && vi == vj ? 2 : 1;
          p_contributions[slot].push_back(
              {&entry, row, col, factor * scale(vi) * scale(vj)});
        }
      }
    }
    int start = 0;
    for (const auto& entry : snapshot->constraints) {
      for (int col = 0; col < entry.A.cols(); ++col) {
        const int variable = entry.variable_indices[col];
        const auto* begin =
            A_sparse.innerIndexPtr() + A_sparse.outerIndexPtr()[variable];
        const auto* end =
            A_sparse.innerIndexPtr() + A_sparse.outerIndexPtr()[variable + 1];
        // Include zero local coefficients whenever another contribution keeps
        // the native slot present, including repeated variables in a binding.
        for (auto* row = std::lower_bound(begin, end, start);
             row != end && *row < start + entry.A.rows(); ++row) {
          const int slot = row - A_sparse.innerIndexPtr();
          a_contributions[slot].push_back(
              {&entry, *row - start, col, scale(variable)});
        }
      }
      start += entry.A.rows();
    }
  }

  std::string CollectMatrixChanges(
      const MathematicalProgram& prog, internal::SolverSparseUpdate* p_changes,
      internal::SolverSparseUpdate* a_changes) const {
    std::string reason;
    const auto scale = [&](int index) {
      const auto it = prog.GetVariableScaling().find(index);
      return it == prog.GetVariableScaling().end() ? 1.0 : it->second;
    };
    const auto add = [&](const auto& matrix, int row, int col, double delta,
                         auto* changes) {
      if (delta == 0) return;
      const int index = FindEntry(matrix, row, col);
      if (index < 0)
        reason = "matrix sparsity changed";
      else
        changes->Mark(index);
    };
    for (const auto& entry : snapshot->costs) {
      if (!entry.matrix_changed) continue;
      const auto& Q = *entry.current_Q();
      for (int j = 0; j < Q.cols(); ++j) {
        for (int i = 0; i <= j; ++i) {
          const int vi = entry.variable_indices[i];
          const int vj = entry.variable_indices[j];
          const double factor = i != j && vi == vj ? 2 : 1;
          add(P_upper_sparse, std::min(vi, vj), std::max(vi, vj),
              factor * scale(vi) * scale(vj) * (Q(i, j) - entry.Q(i, j)),
              p_changes);
        }
      }
    }
    int row = 0;
    for (const auto& entry : snapshot->constraints) {
      if (entry.matrix_changed) {
        const auto add_matrix = [&](const auto& A, double sign) {
          for (int j = 0; j < A.outerSize(); ++j) {
            const int col = entry.variable_indices[j];
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it;
                 ++it) {
              add(A_sparse, row + it.row(), col, sign * scale(col) * it.value(),
                  a_changes);
            }
          }
        };
        add_matrix(entry.A, -1);
        add_matrix(*entry.current_A(), 1);
      }
      row += entry.v.size();
    }
    return reason;
  }

  std::string CheckUpdates(const MathematicalProgram& prog) {
    pending_p_changes.Reset();
    pending_a_changes.Reset();
    return CollectMatrixChanges(prog, &pending_p_changes, &pending_a_changes);
  }

  OSQPInt UpdateMatrices() {
    pending_p_changes.Apply(P_upper_sparse.valuePtr(), [&](int slot) {
      double value = 0;
      for (const auto& c : p_contributions[slot])
        value += c.factor * (*c.binding->current_Q())(c.row, c.col);
      return value;
    });
    pending_a_changes.Apply(A_sparse.valuePtr(), [&](int slot) {
      double value = 0;
      for (const auto& c : a_contributions[slot])
        value += c.factor * c.binding->current_A()->coeff(c.row, c.col);
      return value;
    });
    const auto& pi = pending_p_changes.indices();
    const auto& ai = pending_a_changes.indices();
    const auto& px = pending_p_changes.values();
    const auto& ax = pending_a_changes.values();
    if (px.empty() && ax.empty()) return 0;
    SolverCachePhaseScope phase(SolverCachePhase::kNativeUpdate);
    return osqp_update_data_mat(
        solver, px.empty() ? nullptr : px.data(), pi.data(), pi.size(),
        ax.empty() ? nullptr : ax.data(), ai.data(), ai.size());
  }

  OSQPInt UpdateVectors(const MathematicalProgram& prog) {
    bool cost_changed = false;
    bool bounds_changed = false;
    for (const auto& entry : snapshot->costs) {
      cost_changed |= entry.linear_cost_changed;
    }
    if (cost_changed) std::fill(q.begin(), q.end(), 0);
    constant_cost_term = 0;
    // Match the original quadratic-then-linear aggregation order. Recompute
    // from absolute coefficients to avoid cancellation across updates.
    for (bool quadratic : {true, false}) {
      for (const auto& entry : snapshot->costs) {
        if ((entry.current_Q() != nullptr) != quadratic) continue;
        constant_cost_term += entry.current_constant();
        if (!cost_changed) continue;
        const auto& b = *entry.current_v();
        for (int i = 0; i < b.size(); ++i) q[entry.variable_indices[i]] += b(i);
      }
    }
    if (cost_changed) {
      for (const auto& [index, scale] : prog.GetVariableScaling())
        q[index] *= scale;
    }
    int row = 0;
    for (auto& entry : snapshot->constraints) {
      if (entry.vectors_changed) {
        for (int i = 0; i < entry.v.size(); ++i) {
          l[row + i] = ConvertInfinity((*entry.current_v())(i));
          u[row + i] = ConvertInfinity((*entry.current_w())(i));
        }
        bounds_changed = true;
      }
      row += entry.v.size();
    }
    OSQPInt error = 0;
    if (cost_changed || bounds_changed) {
      SolverCachePhaseScope phase(SolverCachePhase::kNativeUpdate);
      error = osqp_update_data_vec(solver, cost_changed ? q.data() : nullptr,
                                   bounds_changed ? l.data() : nullptr,
                                   bounds_changed ? u.data() : nullptr);
    }
    return error;
  }

  internal::SolverSparseUpdate pending_p_changes, pending_a_changes;
  std::vector<std::vector<Contribution>> p_contributions, a_contributions;
  Eigen::SparseMatrix<OSQPFloat> P_upper_sparse;
  Eigen::SparseMatrix<OSQPFloat> A_sparse;
  std::vector<OSQPFloat> q, l, u;
  double constant_cost_term{};
  std::unordered_map<Binding<Constraint>, int, internal::BindingHash,
                     internal::BindingEqual>
      constraint_start_row;
  OSQPSettings settings{};
  OSQPSolver* solver{};
  std::optional<internal::SolverProgramSnapshot> snapshot;
};

template <typename C>
void SetDualSolution(
    const std::vector<Binding<C>>& constraints,
    const Eigen::VectorXd& all_dual_solution,
    const std::unordered_map<Binding<Constraint>, int, internal::BindingHash,
                             internal::BindingEqual>& constraint_start_row,
    MathematicalProgramResult* result) {
  for (const auto& constraint : constraints) {
    // OSQP uses the dual variable `y` as the negation of the shadow price, so
    // we need to negate `all_dual_solution` as Drake interprets dual solution
    // as the shadow price.
    result->set_dual_solution(constraint,
                              -all_dual_solution.segment(
                                  constraint_start_row.find(constraint)->second,
                                  constraint.evaluator()->num_constraints()));
  }
}
}  // namespace

bool OsqpSolver::is_available() {
  return true;
}

void OsqpSolver::DoSolve2(const MathematicalProgram& prog,
                          const Eigen::VectorXd& initial_guess,
                          internal::SpecificOptions* options,
                          MathematicalProgramResult* result) const {
  OsqpProblemData::SolveWithCache(prog, initial_guess, options, result,
                                  nullptr);
}

void OsqpProblemData::SolveWithCache(const MathematicalProgram& prog,
                                     const Eigen::VectorXd& initial_guess,
                                     internal::SpecificOptions* options,
                                     MathematicalProgramResult* result,
                                     OsqpProblemData* cached) {
  auto& solver_details = result->SetSolverDetailsType<OsqpSolverDetails>();

  const internal::SolverCacheOptions cache_options(options, cached != nullptr);
  OSQPSettings new_settings{};
  auto* settings = &new_settings;
  osqp_set_default_settings(settings);
  // Customize the defaults for Drake.
  // - Default polishing to true, to get an accurate solution.
  // - Disable adaptive rho, for determinism.
  settings->polishing = 1;
  settings->adaptive_rho_interval = OSQP_ADAPTIVE_RHO_FIXED;
  // Apply the user's additional options (if any).
  options->Respell([](const auto& common, auto* respelled) {
    respelled->emplace("verbose", common.print_to_console ? 1 : 0);
    // OSQP does not support setting the number of threads so we ignore the
    // kMaxThreads option.
  });
  options->CopyToSerializableStruct(settings);

  SolverCachePhaseScope validation_phase(SolverCachePhase::kValidation);
  std::string rebuild_reason;
  if (cached != nullptr) {
    rebuild_reason = cached->snapshot->AnalyzeChanges(prog);
    if (rebuild_reason.empty() && !SameSettings(cached->settings, new_settings))
      rebuild_reason = "solver settings changed";
    if (rebuild_reason.empty()) rebuild_reason = cached->CheckUpdates(prog);
    cache_options.CheckRebuild(rebuild_reason);
  }
  std::unique_ptr<OsqpProblemData> fresh;
  if (cached == nullptr || !rebuild_reason.empty()) {
    SolverCachePhaseScope phase(SolverCachePhase::kSetup);
    fresh = std::make_unique<OsqpProblemData>(prog, cache_options.retain);
    fresh->settings = new_settings;
  }
  auto& data = fresh ? *fresh : *cached;
  const auto& constraint_start_row = data.constraint_start_row;
  const auto& constant_cost_term = data.constant_cost_term;
  const int m = data.A_sparse.rows();
  auto*& solver = data.solver;
  if (cache_options.retain) {
    solver_details.cache.status = cached == nullptr
                                      ? SolverCacheStatus::kCreated
                                  : fresh ? SolverCacheStatus::kRebuilt
                                          : SolverCacheStatus::kReused;
    solver_details.cache.rebuild_reason = rebuild_reason;
  }

  bool completed = false;
  ScopeExit invalidate_on_exception([&]() {
    if (!completed && cached) cached->Invalidate();
  });
  // If any step fails, it will set the solution_result and skip other steps.
  std::optional<SolutionResult> solution_result;

  if (fresh) {
    const OSQPInt osqp_setup_err = data.Setup();
    if (osqp_setup_err != 0) {
      solution_result = SolutionResult::kInvalidInput;
    }
  }

  if (!fresh && data.snapshot->changed()) {
    SolverCachePhaseScope phase(SolverCachePhase::kUpdatePrepare);
    if (data.UpdateMatrices() != 0 || data.UpdateVectors(prog) != 0) {
      solution_result = SolutionResult::kInvalidInput;
    } else {
      data.snapshot->CommitChanges();
      solver_details.cache.status = SolverCacheStatus::kUpdated;
    }
  }

  if (!solution_result && !fresh && !cache_options.warm_start) {
    osqp_cold_start(solver);
  }
  if (!solution_result && initial_guess.array().isFinite().all() &&
      (!cache_options.retain || new_settings.warm_starting)) {
    Eigen::VectorXd guess = initial_guess;
    for (const auto& [index, scale] : prog.GetVariableScaling())
      guess(index) /= scale;
    const OSQPInt error = osqp_warm_start(solver, guess.data(), nullptr);
    if (error != 0) solution_result = SolutionResult::kInvalidInput;
  }

  // Solve problem.
  if (!solution_result) {
    DRAKE_THROW_UNLESS(solver != nullptr);
    SolverCachePhaseScope phase(SolverCachePhase::kNativeSolve);
    const OSQPInt osqp_solve_err = osqp_solve(solver);
    if (osqp_solve_err != 0) {
      solution_result = SolutionResult::kInvalidInput;
    }
  }

  SolverCachePhaseScope extraction_phase(SolverCachePhase::kExtraction);
  // Extract results.
  if (!solution_result) {
    DRAKE_THROW_UNLESS(solver->info != nullptr);

    solver_details.iter = solver->info->iter;
    solver_details.status_val = solver->info->status_val;
    solver_details.primal_res = solver->info->prim_res;
    solver_details.dual_res = solver->info->dual_res;
    solver_details.setup_time = solver->info->setup_time;
    solver_details.solve_time = solver->info->solve_time;
    solver_details.polish_time = solver->info->polish_time;
    solver_details.run_time = solver->info->run_time;
    solver_details.rho_updates = solver->info->rho_updates;

    // We set the primal and dual variables as long as osqp_solve() is finished.
    const Eigen::Map<Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1>> osqp_sol(
        solver->solution->x, prog.num_vars());

    // Scale solution back if `scale_map` is not empty.
    const auto& scale_map = prog.GetVariableScaling();
    if (!scale_map.empty()) {
      drake::VectorX<double> scaled_sol = osqp_sol.cast<double>();
      for (const auto& [index, scale] : scale_map) {
        scaled_sol(index) *= scale;
      }
      result->set_x_val(scaled_sol);
    } else {
      result->set_x_val(osqp_sol.cast<double>());
    }
    if (auto* previous =
            internal::SolverResultAccess::PreviousDetails<OsqpSolverDetails>(
                result))
      solver_details.y.swap(previous->y);
    solver_details.y = Eigen::Map<Eigen::VectorXd>(solver->solution->y, m);
    SetDualSolution(prog.linear_constraints(), solver_details.y,
                    constraint_start_row, result);
    SetDualSolution(prog.linear_equality_constraints(), solver_details.y,
                    constraint_start_row, result);
    SetDualSolution(prog.bounding_box_constraints(), solver_details.y,
                    constraint_start_row, result);

    switch (solver->info->status_val) {
      case OSQP_SOLVED:
      case OSQP_SOLVED_INACCURATE: {
        result->set_optimal_cost(solver->info->obj_val + constant_cost_term);
        solution_result = SolutionResult::kSolutionFound;
        break;
      }
      case OSQP_PRIMAL_INFEASIBLE:
      case OSQP_PRIMAL_INFEASIBLE_INACCURATE: {
        solution_result = SolutionResult::kInfeasibleConstraints;
        result->set_optimal_cost(MathematicalProgram::kGlobalInfeasibleCost);
        break;
      }
      case OSQP_DUAL_INFEASIBLE:
      case OSQP_DUAL_INFEASIBLE_INACCURATE: {
        solution_result = SolutionResult::kDualInfeasible;
        break;
      }
      case OSQP_MAX_ITER_REACHED: {
        solution_result = SolutionResult::kIterationLimit;
        break;
      }
      default: {
        solution_result = SolutionResult::kSolverSpecificError;
        break;
      }
    }
  }
  result->set_solution_result(solution_result.value());
  if (solution_result == SolutionResult::kInvalidInput) {
    if (cached) cached->Invalidate();
  } else {
    if (solution_result != SolutionResult::kSolutionFound)
      osqp_cold_start(solver);
    if (cache_options.retain && fresh) result->SetSolverCache(std::move(fresh));
  }
  completed = true;
}

}  // namespace solvers
}  // namespace drake
