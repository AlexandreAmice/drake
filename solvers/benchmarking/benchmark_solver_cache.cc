#include <optional>
#include <type_traits>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/tools/performance/fixture_common.h"

namespace drake {
namespace solvers {
namespace {

// Planar point-mass MPC with x = [px, py, vx, vy], u = [ax, ay], and a
// nominal 0.1 s sample period. Track a stationary target using quadratic
// state/control costs and a terminal state cost. Enforce workspace, speed,
// acceleration, and acceleration slew limits, including the previous input.
// Horizon H gives 6H+4 variables, 12H+8 scalar constraint rows (counting each
// two-sided bound once), and 24H+6 constraint matrix nonzeros. The quadratic
// Hessian is diagonal. Horizons 50/100 give 304/604 variables and 608/1208
// rows, with constraint matrix density below 1%.
// mode: 0 = fresh, 1 = cached/cold, 2 = cached/warm.
// update: 0 = measured state, 1 = tracking reference, 2 = sample period,
// 3 = terminal L2-norm tracking reference (SCS).
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
  const auto x = prog.NewContinuousVariables<4>(4, horizon + 1, "x");
  const auto u = prog.NewContinuousVariables<2>(2, horizon, "u");
  Eigen::Vector4d measured_state(-2, -1, 0, 0);
  Eigen::Vector4d target(2, 1, 0, 0);
  const Eigen::Vector4d state_limit(5, 5, 2, 2);
  const Eigen::Matrix4d Q = Eigen::Vector4d(10, 10, 1, 1).asDiagonal();
  auto initial =
      prog.AddBoundingBoxConstraint(measured_state, measured_state, x.col(0));
  // The previously applied acceleration is zero; limit each change to 0.25.
  prog.AddBoundingBoxConstraint(-0.25, 0.25, u.col(0));
  const auto make_dynamics = [](double dt) {
    Eigen::Matrix<double, 4, 10> A = Eigen::Matrix<double, 4, 10>::Zero();
    // x[k+1] - F x[k] - G u[k] = 0, with zero-order-held acceleration.
    A.leftCols<4>().setIdentity();
    A.block<4, 4>(0, 4) = -Eigen::Matrix4d::Identity();
    A.block<2, 2>(0, 6) = -dt * Eigen::Matrix2d::Identity();
    A.block<2, 2>(0, 8) = -0.5 * dt * dt * Eigen::Matrix2d::Identity();
    A.block<2, 2>(2, 8) = -dt * Eigen::Matrix2d::Identity();
    return A;
  };
  std::vector<Binding<LinearEqualityConstraint>> dynamics;
  std::vector<Binding<QuadraticCost>> tracking;
  for (int i = 0; i <= horizon; ++i) {
    prog.AddBoundingBoxConstraint(-state_limit, state_limit, x.col(i));
    const Eigen::Matrix4d weight = (i == horizon ? 10 : 1) * Q;
    tracking.push_back(prog.AddQuadraticErrorCost(weight, target, x.col(i)));
    if (i == horizon) break;
    prog.AddBoundingBoxConstraint(-1, 1, u.col(i));
    prog.AddQuadraticErrorCost(0.1, Eigen::Vector2d::Zero(), u.col(i));
    VectorXDecisionVariable vars(10);
    vars << x.col(i + 1), x.col(i), u.col(i);
    dynamics.push_back(prog.AddLinearEqualityConstraint(
        make_dynamics(0.1), Eigen::Vector4d::Zero(), vars));
    if (i > 0) {
      VectorXDecisionVariable controls(4);
      controls << u.col(i), u.col(i - 1);
      Eigen::Matrix<double, 2, 4> difference;
      difference << Eigen::Matrix2d::Identity(), -Eigen::Matrix2d::Identity();
      prog.AddLinearConstraint(difference, Eigen::Vector2d::Constant(-0.25),
                               Eigen::Vector2d::Constant(0.25), controls);
    }
  }
  std::optional<Binding<L2NormCost>> norm;
  if (update == 3) {
    norm = prog.AddL2NormCost(Eigen::Matrix2d::Identity(), -target.head<2>(),
                              x.col(horizon).head<2>());
  }
  int constraint_rows = 0;
  int constraint_nnz = 0;
  for (const auto& binding : prog.GetAllConstraints()) {
    const auto* constraint =
        dynamic_cast<const LinearConstraint*>(binding.evaluator().get());
    DRAKE_DEMAND(constraint != nullptr);
    constraint_rows += constraint->num_constraints();
    constraint_nnz += constraint->get_sparse_A().nonZeros();
  }
  const double constraint_density =
      static_cast<double>(constraint_nnz) / (constraint_rows * prog.num_vars());
  DRAKE_DEMAND(prog.num_vars() >= 100);
  DRAKE_DEMAND(constraint_rows >= 250);
  DRAKE_DEMAND(constraint_density < 0.01);
  state.counters["variables"] = prog.num_vars();
  state.counters["constraint_rows"] = constraint_rows;
  state.counters["constraint_nnz"] = constraint_nnz;
  state.counters["constraint_density"] = constraint_density;
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
      measured_state(0) = -2 + delta;
      initial.evaluator()->set_bounds(measured_state, measured_state);
    } else if (update == 1) {
      target(0) = 2 + delta;
      for (int i = 0; i <= horizon; ++i) {
        const Eigen::Matrix4d weight = (i == horizon ? 10 : 1) * Q;
        tracking[i].evaluator()->UpdateCoefficients(
            2 * weight, -2 * weight * target, target.dot(weight * target));
      }
    } else if (update == 2) {
      const auto A = make_dynamics(0.1 + delta);
      for (const auto& binding : dynamics)
        binding.evaluator()->UpdateCoefficients(A, Eigen::Vector4d::Zero());
    } else {
      target(0) = 2 + delta;
      norm->evaluator()->UpdateCoefficients(Eigen::Matrix2d::Identity(),
                                            -target.head<2>());
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
    ->ArgsProduct({{50, 100}, {0, 1, 2}, {0, 1, 2}});
BENCHMARK_TEMPLATE(RepeatedMpc, ScsSolver)
    ->ArgsProduct({{50, 100}, {0, 1, 2}, {0, 1, 3}});

}  // namespace
}  // namespace solvers
}  // namespace drake
