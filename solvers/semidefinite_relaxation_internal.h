#pragma once

#include <map>
#include <memory>

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {
namespace internal {

// Creates the linear constraints in semidefinite relaxation and computes which
// of the original program variables appear in constraints together. This does
// NOT add the semidefinite constraint on the aggregated variables. That must
// occur afterwards. Returns the semidefinite relaxation mathematical program
// without the semidefinite constraint added, as well as the variable X.
// Throughout this method use y = prog.decision_vars(), x = [y, 1], Y = yyᵀ, and
// X = xxᵀ.
std::pair<std::unique_ptr<MathematicalProgram>, MatrixX<symbolic::Variable>>
MakeSemidefiniteRelaxationLinearConstraintsAndComputeMinorCliques(
    const MathematicalProgram& prog,
    std::optional<std::set<symbolic::Variables>*> term_sparsity);

// If any of the variable groups are subsets of vars, remove them from the set.
// If vars is not a subset of any of the variable groups, add it to the set.
void InsertIfNotSubsetOrReplaceIfSuperset(
    const symbolic::Variables vars,
    std::set<symbolic::Variables>* variable_groups);

}  // namespace internal
}  // namespace solvers
}  // namespace drake