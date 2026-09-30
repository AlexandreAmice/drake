#include "drake/solvers/solver_sparse_update.h"

#include <array>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/limit_malloc.h"

namespace drake {
namespace solvers {
namespace internal {
namespace {
GTEST_TEST(SolverSparseUpdateTest, RepeatedAbsoluteUpdatesWithoutAllocation) {
  SolverSparseUpdate update(3);
  std::array<double, 3> matrix{1e16, 2, 3};
  test::LimitMalloc guard;
  for (double value : {1.0, -1.0, 0.0}) {
    update.Reset();
    update.Mark(2);
    update.Mark(0);
    update.Mark(0);
    update.Apply(matrix.data(), [&](int slot) {
      return slot == 0 ? value : 3.0;
    });
    ASSERT_EQ(update.indices().size(), 1);
    EXPECT_EQ(update.indices()[0], 0);
    EXPECT_EQ(update.values()[0], value);
    EXPECT_EQ(matrix[0], value);
    EXPECT_EQ(matrix[1], 2);
    EXPECT_EQ(matrix[2], 3);
  }
}
}  // namespace
}  // namespace internal
}  // namespace solvers
}  // namespace drake
