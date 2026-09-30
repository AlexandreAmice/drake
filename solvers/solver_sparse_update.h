#pragma once

#include <algorithm>
#include <vector>

#include "drake/common/drake_assert.h"

namespace drake {
namespace solvers {
namespace internal {

// Scratch space for updating existing CSC slots. Reserve all storage at setup;
// marking and absolute reaggregation allocate nothing on the repeated path.
class SolverSparseUpdate {
 public:
  explicit SolverSparseUpdate(int slots = 0) : marked_(slots, false) {
    touched_.reserve(slots);
    indices_.reserve(slots);
    values_.reserve(slots);
  }
  void Reset() {
    for (int slot : touched_) marked_[slot] = false;
    touched_.clear();
    indices_.clear();
    values_.clear();
  }
  void Mark(int slot) {
    DRAKE_ASSERT(slot >= 0 && slot < static_cast<int>(marked_.size()));
    if (!marked_[slot]) {
      marked_[slot] = true;
      touched_.push_back(slot);
    }
  }
  // Compute each touched value from its current contributions, not by adding
  // a difference to its old value. Keep the native update indices sorted.
  template <typename Accumulate>
  void Apply(double* matrix_values, const Accumulate& accumulate) {
    std::sort(touched_.begin(), touched_.end());
    for (int slot : touched_) {
      const double value = accumulate(slot);
      if (matrix_values[slot] == value) continue;
      matrix_values[slot] = value;
      indices_.push_back(slot);
      values_.push_back(value);
    }
  }
  const std::vector<int>& indices() const { return indices_; }
  const std::vector<double>& values() const { return values_; }

 private:
  std::vector<bool> marked_;
  std::vector<int> touched_, indices_;
  std::vector<double> values_;
};

}  // namespace internal
}  // namespace solvers
}  // namespace drake
