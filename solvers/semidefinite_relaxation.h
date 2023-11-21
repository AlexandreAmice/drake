#pragma once

#include <map>
#include <memory>

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {

// TODO(russt): Add an option for using diagonal dominance and/or
// scaled-diagonal dominance instead of the PSD constraint.

// TODO(russt): Consider adding Y as an optional argument return value, to help
// users know the decision variables associated with Y.

/** Constructs a new MathematicalProgram which represents the semidefinite
 programming convex relaxation of the (likely nonconvex) program `prog`. This
 method currently supports only linear and quadratic costs and constraints, but
 may be extended in the future with broader support.

 See https://underactuated.mit.edu/optimization.html#sdp_relaxation for
 references and examples.

 Note: Currently, programs using LinearEqualityConstraint will give tighter
 relaxations than programs using LinearConstraint or BoundingBoxConstraint,
 even if lower_bound == upper_bound. Prefer LinearEqualityConstraint.

 // TODO(Alexandre.Amice) fix this comment.
 If use_term_sparsity is set to true, then only the minors corresponding
 to variables which appear in the same constraint are enforced to be PSD.
 Setting this option to true in general makes the semidefinite program faster to
 solver, at the expense of making it weaker (i.e. a looser relaxation).

 @throws std::exception if `prog` has costs and constraints which are not
 linear nor quadratic.
 */
std::unique_ptr<MathematicalProgram> MakeSemidefiniteRelaxation(
    const MathematicalProgram& prog, bool use_term_sparsity = false);

/**
 * A variant of MakeSemidefiniteRelaxation which only enforces that the minor
 * corresponding to the outer products of variables appearing in each Variables
 * object are PSD. This in general makes a faster PSD program, at the expense of
 * making it weaker (i.e. a looser relaxation). If the second entry of the pair
 * is true, include the constant terms.
 *
 * If variables_to_enforce_sparsity = std::nullopt then enforce that the outer
 * product of all the variables be PSD.
 *
 * Throws if any key in variables to enforce sparsity is not a subset of the
 * decision variables in prog.
 */
std::unique_ptr<MathematicalProgram> MakeSemidefiniteRelaxation(
    const MathematicalProgram& prog,
    const std::map<symbolic::Variables, bool>& variables_to_enforce_sparsity);

}  // namespace solvers
}  // namespace drake
