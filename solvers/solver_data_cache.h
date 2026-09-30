#pragma once

#include <memory>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "drake/common/drake_copyable.h"
#include "drake/common/identifier.h"
#include "drake/solvers/solver_id.h"

namespace drake {
namespace solvers {

class MathematicalProgram;
class MathematicalProgramResult;
namespace internal {
class SpecificOptions;
}

/** Opaque native solver state owned by a MathematicalProgramResult.
The cache belongs to one solver and one program instance. It may outlive
either, but cannot be used with another program, including a Clone(). */
class SolverDataCache {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SolverDataCache);
  virtual ~SolverDataCache();

  const SolverId& solver_id() const { return solver_id_; }
  /** Throws if the solver or program does not match this cache. */
  void CheckCompatibility(const MathematicalProgram& prog,
                          const SolverId& solver_id) const;

 protected:
  SolverDataCache(const MathematicalProgram& prog, SolverId solver_id);

  // Native operations invalidate the cache until they have succeeded. A
  // preflight rejection leaves it valid, so the caller can restore ownership.
  void Invalidate() { valid_ = false; }

 private:
  friend class SolverBase;
  // Returns false for caches whose solver implements its own dispatch.
  virtual bool DoSolve(const MathematicalProgram&, const Eigen::VectorXd&,
                       internal::SpecificOptions*, MathematicalProgramResult*) {
    return false;
  }
  bool valid_{true};
  Identifier<MathematicalProgram> program_id_;
  SolverId solver_id_;
};

/** What happened to the native workspace during this solve. */
enum class SolverCacheStatus {
  kNotUsed,
  kCreated,
  kReused,
  kUpdated,
  kRebuilt,
};

/** Cache diagnostics for one solve; independent of the retained workspace. */
struct SolverCacheDetails {
  SolverCacheStatus status{SolverCacheStatus::kNotUsed};
  std::string rebuild_reason;
};

namespace internal {
// Preserve result value semantics without copying opaque native resources.
template <typename T>
struct SolverScratchStorage {
  SolverScratchStorage() = default;
  SolverScratchStorage(const SolverScratchStorage&) {}
  SolverScratchStorage& operator=(const SolverScratchStorage& other) {
    if (this != &other) value.reset();
    return *this;
  }
  SolverScratchStorage(SolverScratchStorage&&) = default;
  SolverScratchStorage& operator=(SolverScratchStorage&&) = default;
  std::unique_ptr<T> value;
};
using SolverDataCacheStorage = SolverScratchStorage<SolverDataCache>;
}  // namespace internal
}  // namespace solvers
}  // namespace drake
