#include <vector>

#include <osqp.h>

#include "drake/solvers/aggregate_costs_constraints.h"
#include "drake/solvers/benchmarking/native_mpc.h"
#include "drake/solvers/osqp_solver.h"

namespace drake {
namespace solvers {
namespace benchmarking {
namespace {

class NativeOsqp {
 public:
  using Solver = OsqpSolver;
  struct Data {
    Eigen::SparseMatrix<double> P, A;
    std::vector<double> q, lower, upper;
    double constant{};
  };
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(NativeOsqp);
  NativeOsqp(int mode, bool adaptive) : mode_(mode) {
    osqp_set_default_settings(&settings_);
    settings_.verbose = 0;
    settings_.polishing = 0;
    settings_.warm_starting = mode == 2;
    settings_.adaptive_rho_interval = OSQP_ADAPTIVE_RHO_FIXED;
    if (!adaptive) settings_.adaptive_rho = 0;
  }
  ~NativeOsqp() {
    if (solver_) osqp_cleanup(solver_);
  }
  static SolverOptions Options(int mode, bool adaptive) {
    SolverOptions options;
    options.SetOption(Solver::id(), "retain_solver_cache", mode != 0 ? 1 : 0);
    options.SetOption(Solver::id(), "warm_start_from_cache", mode == 2 ? 1 : 0);
    options.SetOption(Solver::id(), "warm_starting", mode == 2 ? 1 : 0);
    options.SetOption(Solver::id(), "polishing", 0);
    if (!adaptive) options.SetOption(Solver::id(), "adaptive_rho", 0);
    return options;
  }
  static Data Parse(const MathematicalProgram& prog) {
    Data data;
    data.q.resize(prog.num_vars());
    std::vector<Eigen::Triplet<double>> hessian;
    internal::ParseQuadraticCosts(prog, &hessian, &data.q, &data.constant);
    internal::ParseLinearCosts(prog, &data.q, &data.constant);
    data.P.resize(prog.num_vars(), prog.num_vars());
    data.P.setFromTriplets(hessian.begin(), hessian.end());
    std::vector<Eigen::Triplet<double>> entries;
    int rows = 0;
    const auto append = [&](const auto& bindings) {
      for (const auto& binding : bindings) {
        const auto indices =
            prog.FindDecisionVariableIndices(binding.variables());
        const auto& A = binding.evaluator()->get_sparse_A();
        for (int col = 0; col < A.outerSize(); ++col) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(A, col); it; ++it)
            entries.emplace_back(rows + it.row(), indices[col], it.value());
        }
        for (int i = 0; i < A.rows(); ++i) {
          data.lower.push_back(binding.evaluator()->lower_bound()(i));
          data.upper.push_back(binding.evaluator()->upper_bound()(i));
        }
        rows += A.rows();
      }
    };
    // Keep the same canonical row order as OsqpSolver.
    append(prog.linear_constraints());
    append(prog.linear_equality_constraints());
    append(prog.bounding_box_constraints());
    data.A.resize(rows, prog.num_vars());
    data.A.setFromTriplets(entries.begin(), entries.end());
    return data;
  }
  void Setup(const Data& data) {
    if (solver_) osqp_cleanup(solver_);
    solver_ = nullptr;
    // osqp_setup copies these borrowed CSC views into its workspace.
    num_vars_ = data.P.cols();
    auto P = View(data.P);
    auto A = View(data.A);
    DRAKE_DEMAND(osqp_setup(&solver_, &P, data.q.data(), &A, data.lower.data(),
                            data.upper.data(), data.A.rows(), data.A.cols(),
                            &settings_) == 0);
  }
  void Update(const Data& data, int update) {
    if (mode_ == 0) {
      Setup(data);
      return;
    }
    if (update == 2) {
      DRAKE_DEMAND(osqp_update_data_mat(solver_, nullptr, nullptr, 0,
                                        data.A.valuePtr(), nullptr,
                                        data.A.nonZeros()) == 0);
    } else {
      DRAKE_DEMAND(
          osqp_update_data_vec(solver_, update == 1 ? data.q.data() : nullptr,
                               update == 0 ? data.lower.data() : nullptr,
                               update == 0 ? data.upper.data() : nullptr) == 0);
    }
    if (mode_ == 1) osqp_cold_start(solver_);
  }
  void Solve() {
    DRAKE_DEMAND(osqp_solve(solver_) == 0);
    DRAKE_DEMAND(solver_->info->status_val == OSQP_SOLVED);
  }
  Eigen::Map<const Eigen::VectorXd> x() const {
    return Eigen::Map<const Eigen::VectorXd>(solver_->solution->x, num_vars_);
  }
  double objective() const { return solver_->info->obj_val; }
  int iterations() const { return solver_->info->iter; }
  double primal_residual() const { return solver_->info->prim_res; }

 private:
  static OSQPCscMatrix View(const Eigen::SparseMatrix<double>& matrix) {
    OSQPCscMatrix result{};
    OSQPCscMatrix_set_data(&result, matrix.rows(), matrix.cols(),
                           matrix.nonZeros(),
                           const_cast<double*>(matrix.valuePtr()),
                           const_cast<int*>(matrix.innerIndexPtr()),
                           const_cast<int*>(matrix.outerIndexPtr()));
    return result;
  }
  const int mode_;
  int num_vars_{};
  OSQPSettings settings_{};
  OSQPSolver* solver_{};
};

BENCHMARK_TEMPLATE(NativeMpc, NativeOsqp)
    ->ArgsProduct({{50, 100}, {0, 1, 2}, {0, 1, 2}, {0, 1}});

}  // namespace
}  // namespace benchmarking
}  // namespace solvers
}  // namespace drake
