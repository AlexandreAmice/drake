#include "drake/multibody/rational_forward_kinematics/redundant_inequality_pruning.h"

#include <limits>
#include <list>

#include "drake/solvers/solve.h"

namespace drake {
namespace multibody {
const double kInf = std::numeric_limits<double>::infinity();

std::vector<int> FindRedundantInequalitiesInHPolyhedronByIndex(
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d, double tighten) {
  const int num_inequalities = C.rows();
  const int num_vars = C.cols();

  std::unordered_set<int> kept_indices;
  for (int i = 0; i < num_inequalities; i++) {
    kept_indices.emplace(i);
  }

  std::vector<int> excluded_indices(0);

  for (int excluded_index = 0; excluded_index < num_inequalities;
       excluded_index++) {
    solvers::MathematicalProgram prog;
    solvers::VectorXDecisionVariable x =
        prog.NewContinuousVariables(num_vars, "x");
    std::unordered_set<int> cur_kept_indices = kept_indices;
    cur_kept_indices.erase(excluded_index);

    // constraint c^Tx <= d+1
    prog.AddLinearConstraint(
        C.row(excluded_index), Eigen::VectorXd::Constant(1, -kInf),
        d.row(excluded_index) + Eigen::VectorXd::Ones(1), x);

    // constraint Ax <= b
    for (const int i : cur_kept_indices) {
      prog.AddLinearConstraint(C.row(i), Eigen::VectorXd::Constant(1, -kInf),
                               d.row(i), x);
    }

    prog.AddLinearCost(-C.row(excluded_index), 0, x);

    auto result = solvers::Solve(prog);

    if (!result.is_success()) {
      // polyhedron is empty so exit as redundancy is ill-defined here
      return std::vector<int>{};
    }

    if (-result.get_optimal_cost() <= d(excluded_index) - tighten) {
      excluded_indices.push_back(excluded_index);
      kept_indices.erase(excluded_index);
    }
  }
  return excluded_indices;
}

}  // namespace multibody
}  // namespace drake
