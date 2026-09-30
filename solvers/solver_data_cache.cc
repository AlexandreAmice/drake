#include "drake/solvers/solver_data_cache.h"

#include <stdexcept>
#include <utility>

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {

SolverDataCache::SolverDataCache(const MathematicalProgram& prog,
                                 SolverId solver_id)
    : program_id_(prog.solver_cache_id_), solver_id_(std::move(solver_id)) {}

SolverDataCache::~SolverDataCache() = default;

void SolverDataCache::CheckCompatibility(const MathematicalProgram& prog,
                                         const SolverId& solver_id) const {
  if (program_id_ != prog.solver_cache_id_) {
    throw std::invalid_argument(
        "Solver cache belongs to a different MathematicalProgram. "
        "Use a fresh result or disable retain_solver_cache.");
  }
  if (solver_id_ != solver_id) {
    throw std::invalid_argument("Solver cache belongs to a different solver.");
  }
}

}  // namespace solvers
}  // namespace drake
