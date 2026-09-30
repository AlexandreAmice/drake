#include "drake/solvers/solver_cache_profiler.h"

#include <thread>

#include <gtest/gtest.h>

namespace drake {
namespace solvers {
namespace internal {
namespace {
GTEST_TEST(SolverCacheProfilerTest, ScopeAndThreadIsolation) {
  SolverCacheProfiler profile(true);
  {
    SolverCachePhaseScope phase(SolverCachePhase::kValidation);
    std::thread worker([] {
      SolverCachePhaseScope other(SolverCachePhase::kNativeSolve);
    });
    worker.join();
    {
      SolverCachePhaseScope inner(SolverCachePhase::kNativeUpdate);
    }
  }
  const auto seconds = profile.seconds();
  EXPECT_GT(seconds[static_cast<int>(SolverCachePhase::kValidation)], 0);
  EXPECT_GT(seconds[static_cast<int>(SolverCachePhase::kNativeUpdate)], 0);
  EXPECT_EQ(seconds[static_cast<int>(SolverCachePhase::kNativeSolve)], 0);
}

GTEST_TEST(SolverCacheProfilerTest, Disabled) {
  SolverCacheProfiler profile(false);
  {
    SolverCachePhaseScope phase(SolverCachePhase::kValidation);
  }
  for (double seconds : profile.seconds()) EXPECT_EQ(seconds, 0);
}
}  // namespace
}  // namespace internal
}  // namespace solvers
}  // namespace drake
