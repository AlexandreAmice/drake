#include <optional>
#include <type_traits>
#include <vector>

#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/tools/performance/fixture_common.h"

namespace drake {
namespace solvers {
namespace {

// mode: 0 = fresh, 1 = cached/cold, 2 = cached/warm.
// update: 0 = initial-state bound, 1 = linear objective, 2 = dynamics matrix,
// 3 = affine L2-norm offset (SCS).
template <typename Solver>
void RepeatedMpc(benchmark::State& state) {  // NOLINT
  Solver solver;
  if (!solver.available()) {
    state.SkipWithError("Solver unavailable");
    return;
  }
  const int horizon = state.range(0);
  const int mode = state.range(1);
  const int update = state.range(2);
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables(horizon + 1);
  const auto u = prog.NewContinuousVariables(horizon);
  prog.AddBoundingBoxConstraint(-5, 5, x);
  prog.AddBoundingBoxConstraint(-1, 1, u);
  auto initial = prog.AddBoundingBoxConstraint(1, 1, x.head<1>());
  std::vector<Binding<LinearEqualityConstraint>> dynamics;
  Eigen::RowVector3d A(1, -0.95, -1);
  for (int i = 0; i < horizon; ++i) {
    VectorXDecisionVariable vars(3);
    vars << x(i + 1), x(i), u(i);
    dynamics.push_back(prog.AddLinearEqualityConstraint(A, 0, vars));
    prog.AddQuadraticCost(x(i) * x(i) + 0.1 * u(i) * u(i));
  }
  prog.AddQuadraticCost(5 * x(horizon) * x(horizon));
  Eigen::VectorXd linear = Eigen::VectorXd::Zero(horizon + 1);
  auto objective = prog.AddLinearCost(linear, 0, x);
  std::optional<Binding<L2NormCost>> norm;
  if (update == 3) {
    norm = prog.AddL2NormCost(Eigen::Matrix2d::Identity(),
                              Eigen::Vector2d::Zero(), x.head<2>());
  }
  SolverOptions options;
  options.SetOption(solver.id(), "retain_solver_cache", mode != 0 ? 1 : 0);
  options.SetOption(solver.id(), "warm_start_from_cache", mode == 2 ? 1 : 0);
  if constexpr (std::is_same_v<Solver, OsqpSolver>) {
    options.SetOption(solver.id(), "polishing", 0);
    options.SetOption(solver.id(), "warm_starting", mode == 2 ? 1 : 0);
  } else {
    options.SetOption(solver.id(), "warm_start", mode == 2 ? 1 : 0);
  }
  auto result = solver.Solve(prog, {}, options);
  int step = 0;
  double solve_seconds = 0;
  double setup_seconds = 0;
  for (auto _ : state) {
    const double delta = 0.001 * (++step % 20);
    if (update == 0) {
      initial.evaluator()->set_bounds(Eigen::VectorXd::Constant(1, 1 + delta),
                                      Eigen::VectorXd::Constant(1, 1 + delta));
    } else if (update == 1) {
      linear.setConstant(delta);
      objective.evaluator()->UpdateCoefficients(linear);
    } else if (update == 2) {
      A(1) = -0.95 + delta;
      dynamics[0].evaluator()->UpdateCoefficients(A, Eigen::VectorXd::Zero(1));
    } else {
      norm->evaluator()->UpdateCoefficients(Eigen::Matrix2d::Identity(),
                                            Eigen::Vector2d(delta, -delta));
    }
    solver.Solve(prog, {}, options, &result);
    if (!result.is_success()) {
      state.SkipWithError("Solve failed");
      break;
    }
    if constexpr (std::is_same_v<Solver, OsqpSolver>) {
      const auto& details = result.template get_solver_details<Solver>();
      solve_seconds += details.solve_time;
      if (mode == 0 || details.cache.status == SolverCacheStatus::kRebuilt)
        setup_seconds += details.setup_time;
    } else {
      const auto& details = result.template get_solver_details<Solver>();
      solve_seconds += details.scs_solve_time / 1000;
      if (mode == 0 || details.cache.status == SolverCacheStatus::kRebuilt)
        setup_seconds += details.scs_setup_time / 1000;
    }
    benchmark::DoNotOptimize(result.get_optimal_cost());
  }
  state.counters["native_setup_us"] = 1e6 * setup_seconds / state.iterations();
  state.counters["native_solve_us"] = 1e6 * solve_seconds / state.iterations();
}

BENCHMARK_TEMPLATE(RepeatedMpc, OsqpSolver)
    ->ArgsProduct({{20, 100}, {0, 1, 2}, {0, 1, 2}});
BENCHMARK_TEMPLATE(RepeatedMpc, ScsSolver)
    ->ArgsProduct({{20, 100}, {0, 1, 2}, {0, 1, 3}});

}  // namespace
}  // namespace solvers
}  // namespace drake
