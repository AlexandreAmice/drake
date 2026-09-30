#pragma once

#include <stdexcept>
#include <string>

#include "drake/solvers/specific_options.h"

namespace drake {
namespace solvers {
namespace internal {

// Drake options consumed before forwarding options to a native solver.
struct SolverCacheOptions {
  bool retain{};
  bool allow_rebuild{true};
  bool warm_start{true};

  SolverCacheOptions(SpecificOptions* options, bool has_cache) {
    const auto read_bool = [&](const char* name, bool default_value) {
      const int value = options->Pop<int>(name).value_or(default_value);
      if (value != 0 && value != 1) {
        throw std::invalid_argument(std::string(name) + " must be 0 or 1");
      }
      return value != 0;
    };
    retain = read_bool("retain_solver_cache", has_cache);
    warm_start = read_bool("warm_start_from_cache", true);
    const std::string policy =
        options->Pop<std::string>("solver_cache_rebuild_policy")
            .value_or("allow");
    if (policy != "allow" && policy != "error") {
      throw std::invalid_argument(
          "solver_cache_rebuild_policy must be 'allow' or 'error'");
    }
    allow_rebuild = policy == "allow";
  }

  void CheckRebuild(const std::string& reason) const {
    if (!reason.empty() && !allow_rebuild) {
      throw std::runtime_error("Solver cache requires rebuild: " + reason);
    }
  }
};

}  // namespace internal
}  // namespace solvers
}  // namespace drake
