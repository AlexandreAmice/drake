#pragma once

#include <map>
#include <memory>

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {
namespace internal {

std::pair<std::unique_ptr<MathematicalProgram>, MatrixX<symbolic::Variable>>
MakeSemidefiniteRelaxationLinearConstraints(
    const MathematicalProgram& prog,
    std::optional<std::set<symbolic::Variables>*> variable_dependence_cliques);

}  // namespace internal
}  // namespace solvers
}  // namespace drake