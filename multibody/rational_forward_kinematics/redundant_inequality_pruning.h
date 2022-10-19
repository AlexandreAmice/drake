#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"

namespace drake {
namespace multibody {

/**
 * Given the C-space free region candidate C*t<=d,
 * find all the inequalities which are redundant by solving
 * a series of linear programs.
 * @param tighten We remove the i'th row C.row(i) * t <= d(i) iff C.row(i) * t
 * <= d(i) - tighten is redundant (namely the other rows of inequality
 * constraints would imply C.row(i) * t <= d(i) - tighten).
 * @return the indices of the redundant inequalites
 */
std::vector<int> FindRedundantInequalitiesInHPolyhedronByIndex(
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d, double tighten = 0.);

}  // namespace multibody
}  // namespace drake
