#include "drake/solvers/benchmarking/mpc_fixture.h"

namespace drake {
namespace solvers {
namespace benchmarking {

Eigen::Matrix<double, 4, 10> MpcFixture::MakeDynamics(double dt) {
  Eigen::Matrix<double, 4, 10> A = Eigen::Matrix<double, 4, 10>::Zero();
  // x[k+1] - F x[k] - G u[k] = 0, with zero-order-held acceleration.
  A.leftCols<4>().setIdentity();
  A.block<4, 4>(0, 4) = -Eigen::Matrix4d::Identity();
  A.block<2, 2>(0, 6) = -dt * Eigen::Matrix2d::Identity();
  A.block<2, 2>(0, 8) = -0.5 * dt * dt * Eigen::Matrix2d::Identity();
  A.block<2, 2>(2, 8) = -dt * Eigen::Matrix2d::Identity();
  return A;
}

MpcFixture::MpcFixture(int horizon_in, int update_in)
    : horizon(horizon_in), update(update_in) {
  x = prog.NewContinuousVariables<4>(4, horizon + 1, "x");
  u = prog.NewContinuousVariables<2>(2, horizon, "u");
  const Eigen::Vector4d state_limit(5, 5, 2, 2);
  initial =
      prog.AddBoundingBoxConstraint(measured_state, measured_state, x.col(0));
  // The previously applied acceleration is zero; limit each change to 0.25.
  prog.AddBoundingBoxConstraint(-0.25, 0.25, u.col(0));

  for (int i = 0; i <= horizon; ++i) {
    prog.AddBoundingBoxConstraint(-state_limit, state_limit, x.col(i));
    const Eigen::Matrix4d weight = (i == horizon ? 10 : 1) * Q;
    tracking.push_back(prog.AddQuadraticErrorCost(weight, target, x.col(i)));
    if (i == horizon) break;
    prog.AddBoundingBoxConstraint(-1, 1, u.col(i));
    prog.AddQuadraticErrorCost(0.1, Eigen::Vector2d::Zero(), u.col(i));
    VectorXDecisionVariable vars(10);
    vars << x.col(i + 1), x.col(i), u.col(i);
    dynamics.push_back(prog.AddLinearEqualityConstraint(
        MakeDynamics(0.1), Eigen::Vector4d::Zero(), vars));
    if (i > 0) {
      VectorXDecisionVariable controls(4);
      controls << u.col(i), u.col(i - 1);
      Eigen::Matrix<double, 2, 4> difference;
      difference << Eigen::Matrix2d::Identity(), -Eigen::Matrix2d::Identity();
      prog.AddLinearConstraint(difference, Eigen::Vector2d::Constant(-0.25),
                               Eigen::Vector2d::Constant(0.25), controls);
    }
  }
  if (update == 3) {
    norm = prog.AddL2NormCost(Eigen::Matrix2d::Identity(), -target.head<2>(),
                              x.col(horizon).head<2>());
  }
  constraint_rows = 0;
  constraint_nnz = 0;
  const auto count_rows = [&](const auto& bindings) {
    for (const auto& binding : bindings) {
      constraint_rows += binding.evaluator()->num_constraints();
      constraint_nnz += binding.evaluator()->get_sparse_A().nonZeros();
    }
  };
  count_rows(prog.linear_constraints());
  count_rows(prog.linear_equality_constraints());
  count_rows(prog.bounding_box_constraints());
  constraint_density =
      static_cast<double>(constraint_nnz) / (constraint_rows * prog.num_vars());
  DRAKE_DEMAND(prog.num_vars() >= 100);
  DRAKE_DEMAND(constraint_rows >= 250);
  DRAKE_DEMAND(constraint_density < 0.01);
}

void MpcFixture::Update() {
  const double delta = 0.001 * (++step % 20);
  if (update == 0) {
    measured_state(0) = -2 + delta;
    initial->evaluator()->set_bounds(measured_state, measured_state);
  } else if (update == 1) {
    target(0) = 2 + delta;
    for (int i = 0; i <= horizon; ++i) {
      const Eigen::Matrix4d weight = (i == horizon ? 10 : 1) * Q;
      tracking[i].evaluator()->UpdateCoefficients(
          2 * weight, -2 * weight * target, target.dot(weight * target));
    }
  } else if (update == 2) {
    const auto A = MakeDynamics(0.1 + delta);
    for (const auto& binding : dynamics)
      binding.evaluator()->UpdateCoefficients(A, Eigen::Vector4d::Zero());
  } else {
    target(0) = 2 + delta;
    norm->evaluator()->UpdateCoefficients(Eigen::Matrix2d::Identity(),
                                          -target.head<2>());
  }
}

void MpcFixture::Report(benchmark::State* state) const {
  state->counters["variables"] = prog.num_vars();
  state->counters["constraint_rows"] = constraint_rows;
  state->counters["constraint_nnz"] = constraint_nnz;
  state->counters["constraint_density"] = constraint_density;
}

}  // namespace benchmarking
}  // namespace solvers
}  // namespace drake
