#include <vector>

#include <scs.h>

#include "drake/solvers/aggregate_costs_constraints.h"
#include "drake/solvers/benchmarking/native_mpc.h"
#include "drake/solvers/scs_solver.h"

namespace drake {
namespace solvers {
namespace benchmarking {
namespace {

class NativeScs {
 public:
  using Solver = ScsSolver;
  struct Data {
    Eigen::SparseMatrix<double> P, A;
    std::vector<double> c, b;
    std::vector<int> second_order_cones;
    int equalities{};
    int inequalities{};
    double constant{};
  };
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(NativeScs);
  NativeScs(int mode, bool adaptive) : mode_(mode) {
    scs_set_default_settings(&settings_);
    settings_.verbose = 0;
    settings_.eps_abs = 1e-5;
    settings_.eps_rel = 1e-5;
    settings_.warm_start = mode == 2;
    if (!adaptive) settings_.adaptive_scale = 0;
  }
  ~NativeScs() {
    if (work_) scs_finish(work_);
  }
  static SolverOptions Options(int mode, bool adaptive) {
    SolverOptions options;
    options.SetOption(Solver::id(), "retain_solver_cache", mode != 0 ? 1 : 0);
    options.SetOption(Solver::id(), "warm_start_from_cache", mode == 2 ? 1 : 0);
    options.SetOption(Solver::id(), "warm_start", mode == 2 ? 1 : 0);
    if (!adaptive) options.SetOption(Solver::id(), "adaptive_scale", 0);
    return options;
  }
  static Data Parse(const MathematicalProgram& prog) {
    Data data;
    int num_vars = prog.num_vars();
    data.c.resize(num_vars);
    internal::ParseLinearCosts(prog, &data.c, &data.constant);
    std::vector<Eigen::Triplet<double>> entries;
    int rows = 0;
    std::vector<int> starts;
    internal::ParseLinearEqualityConstraints(prog, &entries, &data.b, &rows,
                                             &starts, &data.equalities);
    // Match SCS's upper-then-lower bounding-box rows, before general
    // inequalities. In particular, Drake represents fixed bounds this way.
    for (const auto& binding : prog.bounding_box_constraints()) {
      const auto indices =
          prog.FindDecisionVariableIndices(binding.variables());
      for (int i = 0; i < binding.GetNumElements(); ++i) {
        entries.emplace_back(rows++, indices[i], 1);
        data.b.push_back(binding.evaluator()->upper_bound()(i));
        entries.emplace_back(rows++, indices[i], -1);
        data.b.push_back(-binding.evaluator()->lower_bound()(i));
      }
    }
    std::vector<std::vector<std::pair<int, int>>> dual_indices;
    int linear_rows{};
    internal::ParseLinearConstraints(prog, &entries, &data.b, &rows,
                                     &dual_indices, &linear_rows);
    data.inequalities = rows - data.equalities;
    std::vector<int> cone_starts, slack_indices;
    internal::ParseL2NormCosts(prog, &num_vars, &entries, &data.b, &rows,
                               &data.second_order_cones, &cone_starts, &data.c,
                               &slack_indices);
    std::vector<Eigen::Triplet<double>> hessian;
    internal::ParseQuadraticCosts(prog, &hessian, &data.c, &data.constant);
    data.P.resize(num_vars, num_vars);
    data.P.setFromTriplets(hessian.begin(), hessian.end());
    data.A.resize(rows, num_vars);
    data.A.setFromTriplets(entries.begin(), entries.end());
    return data;
  }
  void Setup(const Data& data) {
    if (work_) scs_finish(work_);
    work_ = nullptr;
    auto P = View(data.P);
    auto A = View(data.A);
    ScsData problem{};
    problem.m = data.A.rows();
    problem.n = data.A.cols();
    problem.A = &A;
    problem.P = &P;
    problem.b = const_cast<double*>(data.b.data());
    problem.c = const_cast<double*>(data.c.data());
    ScsCone cone{};
    cone.z = data.equalities;
    cone.l = data.inequalities;
    cone.q = const_cast<int*>(data.second_order_cones.data());
    cone.qsize = data.second_order_cones.size();
    work_ = scs_init(&problem, &cone, &settings_);
    DRAKE_DEMAND(work_ != nullptr);
    // SCS accepts caller-owned solution storage. Clear it when starting a
    // fresh workspace; retained warm solves keep all three native iterates.
    x_.setZero(problem.n);
    y_.setZero(problem.m);
    slack_.setZero(problem.m);
    solution_.x = x_.data();
    solution_.y = y_.data();
    solution_.s = slack_.data();
    can_warm_start_ = false;
  }
  void Update(const Data& data, int update) {
    if (mode_ == 0) {
      Setup(data);
      return;
    }
    DRAKE_DEMAND(
        scs_update(
            work_, update == 1 ? nullptr : const_cast<double*>(data.b.data()),
            update == 1 ? const_cast<double*>(data.c.data()) : nullptr) == 0);
  }
  void Solve() {
    const int status =
        scs_solve(work_, &solution_, &info_, mode_ == 2 && can_warm_start_);
    DRAKE_DEMAND(status == SCS_SOLVED);
    can_warm_start_ = true;
  }
  const Eigen::VectorXd& x() const { return x_; }
  double objective() const { return info_.pobj; }
  int iterations() const { return info_.iter; }
  double primal_residual() const { return info_.res_pri; }

 private:
  static ScsMatrix View(const Eigen::SparseMatrix<double>& matrix) {
    ScsMatrix result{};
    result.m = matrix.rows();
    result.n = matrix.cols();
    result.x = const_cast<double*>(matrix.valuePtr());
    result.i = const_cast<int*>(matrix.innerIndexPtr());
    result.p = const_cast<int*>(matrix.outerIndexPtr());
    return result;
  }
  const int mode_;
  bool can_warm_start_{};
  ScsSettings settings_{};
  ScsInfo info_{};
  ScsWork* work_{};
  ScsSolution solution_{};
  Eigen::VectorXd x_, y_, slack_;
};

BENCHMARK_TEMPLATE(NativeMpc, NativeScs)
    ->ArgsProduct({{50, 100}, {0, 1, 2}, {0, 1, 3}, {0, 1}});

}  // namespace
}  // namespace benchmarking
}  // namespace solvers
}  // namespace drake
