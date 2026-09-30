#pragma once

#include <optional>
#include <vector>

#include "drake/solvers/mathematical_program.h"
#include "drake/tools/performance/fixture_common.h"

namespace drake {
namespace solvers {
namespace benchmarking {

// Planar point-mass MPC with x = [px, py, vx, vy], u = [ax, ay], and a
// nominal 0.1 s sample period. Track a stationary target using quadratic
// state/control costs and a terminal state cost. Enforce workspace, speed,
// acceleration, and acceleration slew limits, including the previous input.
// Horizon H gives 6H+4 variables, 12H+8 scalar constraint rows (counting each
// two-sided bound once), and 24H+6 constraint matrix nonzeros. The quadratic
// Hessian is diagonal. Horizons 50/100 give 304/604 variables and 608/1208
// rows, with constraint matrix density below 1%.
// mode: 0 = fresh, 1 = cached/cold, 2 = cached/warm.
// update: 0 = measured state, 1 = tracking reference, 2 = sample period,
// 3 = terminal L2-norm tracking reference (SCS).
class MpcFixture {
 public:
  explicit MpcFixture(int horizon_in, int update_in);
  void Update();
  void Report(benchmark::State* state) const;
  static Eigen::Matrix<double, 4, 10> MakeDynamics(double dt);

  const int horizon;
  const int update;
  MathematicalProgram prog;
  MatrixXDecisionVariable x, u;
  Eigen::Vector4d measured_state{-2, -1, 0, 0};
  Eigen::Vector4d target{2, 1, 0, 0};
  Eigen::Matrix4d Q{Eigen::Vector4d(10, 10, 1, 1).asDiagonal()};
  std::optional<Binding<BoundingBoxConstraint>> initial;
  std::vector<Binding<LinearEqualityConstraint>> dynamics;
  std::vector<Binding<QuadraticCost>> tracking;
  std::optional<Binding<L2NormCost>> norm;
  int step{};
  int constraint_rows{};
  int constraint_nnz{};
  double constraint_density{};
};

}  // namespace benchmarking
}  // namespace solvers
}  // namespace drake
