#pragma once

#include <array>
#include <chrono>

#include "drake/common/drake_copyable.h"

namespace drake {
namespace solvers {
namespace internal {

// Optional, per-thread phase accounting for benchmark diagnostics. Nested
// scopes attribute time to the innermost phase. Normal solves do not read the
// clock; latency comparisons should run with this recorder disabled.
enum class SolverCachePhase {
  kOther,
  kResultPrepare,
  kValidation,
  kUpdatePrepare,
  kNativeUpdate,
  kNativeSolve,
  kExtraction,
  kSetup,
};

class SolverCacheProfiler {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SolverCacheProfiler);
  static constexpr int kNumPhases = 8;
  explicit SolverCacheProfiler(bool enabled)
      : previous_(active_), enabled_(enabled) {
    if (enabled_) {
      last_ = Clock::now();
      active_ = this;
    }
  }
  ~SolverCacheProfiler() {
    if (enabled_) active_ = previous_;
  }
  std::array<double, kNumPhases> seconds() {
    if (enabled_) ChangePhase(phase_);
    return seconds_;
  }

 private:
  friend class SolverCachePhaseScope;
  using Clock = std::chrono::steady_clock;
  SolverCachePhase ChangePhase(SolverCachePhase phase) {
    const auto now = Clock::now();
    seconds_[static_cast<int>(phase_)] +=
        std::chrono::duration<double>(now - last_).count();
    last_ = now;
    const auto previous = phase_;
    phase_ = phase;
    return previous;
  }
  inline static thread_local SolverCacheProfiler* active_{};
  SolverCacheProfiler* const previous_;
  const bool enabled_;
  Clock::time_point last_;
  SolverCachePhase phase_{SolverCachePhase::kOther};
  std::array<double, kNumPhases> seconds_{};
};

class SolverCachePhaseScope {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SolverCachePhaseScope);
  explicit SolverCachePhaseScope(SolverCachePhase phase)
      : profiler_(SolverCacheProfiler::active_) {
    if (profiler_) previous_ = profiler_->ChangePhase(phase);
  }
  ~SolverCachePhaseScope() {
    if (profiler_) profiler_->ChangePhase(previous_);
  }

 private:
  SolverCacheProfiler* const profiler_;
  SolverCachePhase previous_{};
};

}  // namespace internal
}  // namespace solvers
}  // namespace drake
