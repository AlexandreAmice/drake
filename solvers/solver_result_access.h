#pragma once

#include "drake/solvers/mathematical_program_result.h"

namespace drake {
namespace solvers {
namespace internal {

// Only successful native extraction borrows the previous vector storage.
// Until then, public solver details retain their default, empty values.
class SolverResultAccess {
 public:
  template <typename Details>
  static Details* PreviousDetails(MathematicalProgramResult* result) {
    auto* previous = result->previous_details_.value.get();
    return previous ? previous->maybe_get_mutable_value<Details>() : nullptr;
  }
};

}  // namespace internal
}  // namespace solvers
}  // namespace drake
