#include "drake/geometry/optimization/dev/polynomial_positive_on_path.h"

#include <limits>
#include <utility>

#include "drake/common/symbolic/monomial_util.h"

namespace drake {
namespace geometry {
namespace optimization {

ParametrizedPolynomialPositiveOnUnitInterval::
    ParametrizedPolynomialPositiveOnUnitInterval(
        const symbolic::Polynomial& poly,
        const symbolic::Variable& interval_variable,
        const symbolic::Variables& parameters)
    : mu_(interval_variable),
      p_(poly),
      parameters_(parameters),
      psatz_variables_and_psd_constraints_() {
  psatz_variables_and_psd_constraints_.AddIndeterminates(poly.indeterminates());

  const int deg = poly.Degree(mu_);
  if (poly.TotalDegree() == 0) {
    // If poly is of degree 0, then it is a scalar, and we just need to
    // constraint that p_ >= 0.
    const solvers::VectorXDecisionVariable lambda{
        psatz_variables_and_psd_constraints_.NewContinuousVariables(1, "Sl")};
    psatz_variables_and_psd_constraints_.AddBoundingBoxConstraint(
        0, std::numeric_limits<double>::infinity(), lambda);
    lambda_ = symbolic::Polynomial{lambda(0), symbolic::Variables()};
    p_ -= lambda(0);
  } else {
    const int d = static_cast<int>(std::floor(deg / 2));
    const solvers::MathematicalProgram::NonnegativePolynomial type =
        solvers::MathematicalProgram::NonnegativePolynomial::kSos;

    // This basis is [μᵈ, ... μ, 1, y₁, ..., yₙ]
    const int num_y_indets =
        poly.indeterminates().size() - (poly.Degree(mu_) > 0 ? 1 : 0);

    VectorX<symbolic::Monomial> multiplier_basis_d{d + 1 + num_y_indets};

    multiplier_basis_d.head(d + 1) = symbolic::MonomialBasis({mu_}, d);
    int i = d + 1;
    for (const auto& var : poly.indeterminates()) {
      if (!var.equal_to(mu_)) {
        DRAKE_DEMAND(poly.Degree(var) <= 2);
        multiplier_basis_d(i) = symbolic::Monomial(var);
        ++i;
      }
    }

    // Constructs the multiplier polynomials and their associated Gram matrices
    // as well as the polynomial p_. Recall that p_ has already been initialized
    // to poly(μ,y).
    auto [lambda, Q_lambda] =
        psatz_variables_and_psd_constraints_.NewSosPolynomial(
            multiplier_basis_d, type, "Sl");
    lambda_ = std::move(lambda);
    if (deg == 0) {
      p_ -= lambda_;  // interval variable doesn't exist in the program, so we
                      // can ignore it.
    } else if (deg % 2 == 0) {
      auto [nu, Q_nu] = psatz_variables_and_psd_constraints_.NewSosPolynomial(
          multiplier_basis_d.tail(multiplier_basis_d.size() -
                                  1),  // exclude μᵈ monomial
          type, "Sv");
      nu_ = std::move(nu);
      p_ -= lambda_ + nu_ * mu_ * (symbolic::Polynomial(1, {mu_}) - mu_);
    } else {
      auto [nu, Q_nu] = psatz_variables_and_psd_constraints_.NewSosPolynomial(
          multiplier_basis_d, type, "Sv");
      nu_ = std::move(nu);
      p_ -= lambda_ * mu_ + nu_ * (symbolic::Polynomial(1, {mu_}) - mu_);
    }
  }
}

void ParametrizedPolynomialPositiveOnUnitInterval::
    AddPositivityConstraintToProgram(const symbolic::Environment& env,
                                     solvers::MathematicalProgram* prog) const {
  DRAKE_DEMAND(env.size() == parameters_.size());
  for (const auto& parameter : parameters_) {
    DRAKE_DEMAND(env.find(parameter) != env.cend());
  }
  for (int i = 0;
       i < psatz_variables_and_psd_constraints_.indeterminates().size(); ++i) {
    // Check that prog contains the indeterminates of this program.
    DRAKE_DEMAND(
        prog->indeterminates_index().count(
            psatz_variables_and_psd_constraints_.indeterminates()(i).get_id()) >
        0);
  }

  prog->AddDecisionVariables(
      psatz_variables_and_psd_constraints_.decision_variables());
  for (const auto& binding :
       psatz_variables_and_psd_constraints_.GetAllConstraints()) {
    prog->AddConstraint(binding);
  }
  prog->AddEqualityConstraintBetweenPolynomials(p_.EvaluatePartial(env),
                                                symbolic::Polynomial());
}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
