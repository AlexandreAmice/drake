#include <algorithm>
#include <chrono>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/solvers/benchmarking/mpc_fixture.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/solver_cache_profiler.h"
#include "drake/tools/performance/fixture_common.h"
#include "drake/tools/performance/fixture_memory.h"

DEFINE_bool(solver_cache_profile, false, "Record solver cache phase times");

namespace drake {
namespace solvers {
namespace {

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
  benchmarking::MpcFixture fixture(horizon, update);
  auto& prog = fixture.prog;
  fixture.Report(&state);
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
  for (int i = 0; i < 100; ++i) {
    fixture.Update();
    solver.Solve(prog, {}, options, &result);
    DRAKE_DEMAND(result.is_success());
  }
  internal::SolverCacheProfiler profiler(FLAGS_solver_cache_profile);
  std::vector<double> latencies;
  latencies.reserve(state.max_iterations);
  tools::performance::TareMemoryManager();
  double iterations = 0;
  double solve_seconds = 0;
  double setup_seconds = 0;
  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    fixture.Update();
    solver.Solve(prog, {}, options, &result);
    latencies.push_back(std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - start)
                            .count());
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
    iterations += result.template get_solver_details<Solver>().iter;
    benchmark::DoNotOptimize(result.get_optimal_cost());
  }
  state.counters["native_iterations"] = iterations / state.iterations();
  const auto phase_seconds = profiler.seconds();
  const char* phase_names[] = {
      "other",         "result_prepare", "validation", "update_prepare",
      "native_update", "native_solve",   "extraction", "setup"};
  if (FLAGS_solver_cache_profile) {
    for (int i = 1; i < internal::SolverCacheProfiler::kNumPhases; ++i)
      state.counters[phase_names[i]] =
          1e6 * phase_seconds[i] / state.iterations();
  }
  std::sort(latencies.begin(), latencies.end());
  if (!latencies.empty()) {
    for (const auto& [name, fraction] :
         {std::pair{"median_us", 0.5}, {"p95_us", 0.95}, {"p99_us", 0.99}}) {
      state.counters[name] =
          latencies[static_cast<size_t>(fraction * (latencies.size() - 1))];
    }
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
