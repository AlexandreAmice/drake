#pragma once

#include "drake/common/symbolic/expression.h"
#include "drake/common/symbolic/polynomial.h"

namespace drake {
namespace multibody {
namespace rational_old {
/**
 * Generate all the monomials of t_angles, such that the order for t_angles(i)
 * is no larger than 1.
 * The link pose is a rational function of t_angles. In both the numerator and
 * denominator polynomials of this rational function, the monomials in these
 * polynomials has the form ∏ tᵢᵐⁱ, where tᵢ is a term in t_angles, and the
 * order mᵢ <= 2.
 * Hence if we compute the monomial basis z for this polynomial, such that the
 * polynomial can be written as zᵀHz, then z should contain all the monomials
 * of form ∏tᵢⁿⁱ, where nᵢ <= 1.
 */
drake::VectorX<drake::symbolic::Monomial> GenerateMonomialBasisWithOrderUpToOne(
    const drake::symbolic::Variables& t_angles);

/**
 * Generate all the monomials of t_angles, such that ∃j, t_angles(j)'s order
 * is no larger than 2, and for all i ≠ j, t_angles(i)'s order is no larger
 * than 1.
 */
drake::VectorX<drake::symbolic::Monomial>
GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(
    const drake::symbolic::Variables& t_angles);
}  // namespace rational_old
}  // namespace multibody
}  // namespace drake
