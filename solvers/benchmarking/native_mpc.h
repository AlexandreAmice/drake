#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "drake/solvers/benchmarking/mpc_fixture.h"
#include "drake/tools/performance/fixture_memory.h"

namespace drake {
namespace solvers {
namespace benchmarking {

// The native baseline excludes Drake coefficient mutation and result
// extraction. Precompute the same twenty updates used by RepeatedMpc, then
// verify the entire warm-up against Drake with identical solver settings.
// mode: fresh / retained cold / retained warm; adaptive: disabled / default.
template <typename Native>
void NativeMpc(benchmark::State& state) {  // NOLINT
  typename Native::Solver solver;
  if (!solver.available()) {
    state.SkipWithError("Solver unavailable");
    return;
  }
  const int mode = state.range(1);
  const int update = state.range(2);
  const bool adaptive = state.range(3);
  MpcFixture fixture(state.range(0), update);
  fixture.Report(&state);
  auto initial = Native::Parse(fixture.prog);
  std::vector<typename Native::Data> changes;
  changes.reserve(20);
  for (int i = 0; i < 20; ++i) {
    fixture.Update();
    changes.push_back(Native::Parse(fixture.prog));
  }
  Native native(mode, adaptive);
  native.Setup(initial);
  native.Solve();
  auto options = Native::Options(mode, adaptive);
  auto result = solver.Solve(fixture.prog, {}, options);
  for (int i = 0; i < 100; ++i) {
    fixture.Update();
    native.Update(changes[i % 20], update);
    native.Solve();
    solver.Solve(fixture.prog, {}, options, &result);
    DRAKE_DEMAND(result.is_success());
    DRAKE_DEMAND((native.x().head(fixture.prog.num_vars()) - result.get_x_val())
                     .template lpNorm<Eigen::Infinity>() < 1e-7);
    const double objective = native.objective() + changes[i % 20].constant;
    DRAKE_DEMAND(std::abs(objective - result.get_optimal_cost()) <
                 1e-7 * (1 + std::abs(objective)));
  }
  std::vector<double> latencies;
  latencies.reserve(state.max_iterations);
  tools::performance::TareMemoryManager();
  int step = 0;
  double iterations = 0;
  double primal_residual = 0;
  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    native.Update(changes[step++ % 20], update);
    native.Solve();
    latencies.push_back(std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - start)
                            .count());
    iterations += native.iterations();
    primal_residual = std::max(primal_residual, native.primal_residual());
    benchmark::DoNotOptimize(native.objective());
  }
  state.counters["native_iterations"] = iterations / state.iterations();
  state.counters["max_primal_residual"] = primal_residual;
  std::sort(latencies.begin(), latencies.end());
  if (!latencies.empty()) {
    state.counters["median_us"] = latencies[latencies.size() / 2];
    state.counters["p95_us"] = latencies[(latencies.size() - 1) * 95 / 100];
    state.counters["p99_us"] = latencies[(latencies.size() - 1) * 99 / 100];
  }
}

}  // namespace benchmarking
}  // namespace solvers
}  // namespace drake
