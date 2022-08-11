#include "drake/multibody/rational_forward_kinematics/cspace_free_line.h"

#include <execution>
#include <utility>

#include "drake/common/symbolic/decompose.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace multibody {
using solvers::MathematicalProgram;

CspaceFreeLine::CspaceFreeLine(
    const systems::Diagram<double>& diagram,
    const multibody::MultibodyPlant<double>* plant,
    const geometry::SceneGraph<double>* scene_graph,
    SeparatingPlaneOrder plane_order, std::optional<Eigen::VectorXd> q_star,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const VerificationOption& option)
    : CspaceFreeRegion(diagram, plant, scene_graph, plane_order,
                       CspaceRegionType::kAxisAlignedBoundingBox),
      mu_{symbolic::Variable("mu")},
      filtered_collision_pairs_{filtered_collision_pairs},
      option_{option} {
  if (q_star.has_value()) {
    DRAKE_DEMAND(q_star.value().size() == plant->num_positions());
    q_star_ = q_star.value();
  } else {
    q_star_ = Eigen::VectorXd::Zero(plant->num_positions());
  }

  InitializeLineVariables(plant->num_positions());
  allocated_certification_program_ = AllocateCertificationProgram();
}

bool CspaceFreeLine::CertifyTangentConfigurationSpaceLine(
    const Eigen::Ref<const Eigen::VectorXd>& s0,
    const Eigen::Ref<const Eigen::VectorXd>& s1,
    const solvers::SolverOptions& solver_options) {
  symbolic::Environment env;
  DRAKE_DEMAND(s0.size() == s0_.size());
  DRAKE_DEMAND(s1.size() == s1_.size());
  auto is_in_joint_limits = [this](const Eigen::Ref<const Eigen::VectorXd>& s) {
    Eigen::VectorXd q_lower =
        rational_forward_kinematics().plant().GetPositionLowerLimits();
    Eigen::VectorXd s_lower =
        rational_forward_kinematics().ComputeTValue(q_lower, q_star_);
    Eigen::VectorXd q_upper =
        rational_forward_kinematics().plant().GetPositionUpperLimits();
    Eigen::VectorXd s_upper =
        rational_forward_kinematics().ComputeTValue(q_upper, q_star_);
    for (int i = 0; i < s.size(); ++i) {
      if (s(i) < s_lower(i) || s(i) > s_upper(i)) {
        throw std::invalid_argument(fmt::format(
            "s = {} is not in the joint limits\n lower limit = {}\n "
            "upper limit = {}",
            s, s_lower, s_upper));
      }
    }
  };
  is_in_joint_limits(s0);
  is_in_joint_limits(s1);

  for (int i = 0; i < s0_.size(); ++i) {
    env.insert(s0_(i), s0(i));
    env.insert(s1_(i), s1(i));
  }
  auto clock_start = std::chrono::system_clock::now();
  allocated_certification_program_.EvaluatePolynomialsAndUpdateProgram(env);
  auto clock_now = std::chrono::system_clock::now();
  drake::log()->info(fmt::format(
      "Bindings updated in {} s",
      static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(
                             clock_now - clock_start)
                             .count()) /
          1000));
  clock_start = std::chrono::system_clock::now();
  const auto result = allocated_certification_program_.solve(solver_options);
  clock_now = std::chrono::system_clock::now();
  drake::log()->info(fmt::format(
      "Certification checked in  {} s",
      static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(
                             clock_now - clock_start)
                             .count()) /
          1000));
  return result.is_success();
}

std::vector<LinkOnPlaneSideRational>
CspaceFreeLine::GenerateRationalsForLinkOnOneSideOfPlane(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs)
    const {
  std::vector<LinkOnPlaneSideRational> generic_rationals =
      CspaceFreeRegion::GenerateRationalsForLinkOnOneSideOfPlane(
          q_star, filtered_collision_pairs);
  std::vector<LinkOnPlaneSideRational> rationals;

  symbolic::Expression numerator_expr;
  symbolic::Polynomial numerator_poly;
  symbolic::Expression denominator_expr;
  symbolic::Polynomial denominator_poly;
  for (const auto& rational : generic_rationals) {
    numerator_expr = rational.rational.numerator().ToExpression();
    numerator_poly =
        symbolic::Polynomial(numerator_expr.Substitute(t_to_line_subs_map_),
                             symbolic::Variables{mu_});

    denominator_expr = rational.rational.denominator().ToExpression();
    denominator_poly =
        symbolic::Polynomial(denominator_expr.Substitute(t_to_line_subs_map_),
                             symbolic::Variables{mu_});

    rationals.emplace_back(
        symbolic::RationalFunction(numerator_poly, denominator_poly),
        rational.link_geometry, rational.expressed_body_index,
        rational.other_side_link_geometry, rational.a_A, rational.b,
        rational.plane_side, rational.plane_order,
        rational.lorentz_cone_constraints);
  }
  return rationals;
}

void CspaceFreeLine::GenerateTuplesForCertification(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs,
    std::vector<CspaceFreeRegion::CspacePolytopeTuple>* alternation_tuples,
    VectorX<symbolic::Variable>* lagrangian_gram_vars,
    VectorX<symbolic::Variable>* verified_gram_vars,
    VectorX<symbolic::Variable>* separating_plane_vars,
    std::vector<std::vector<int>>* separating_plane_to_tuples,
    std::vector<std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>*
        separating_plane_to_lorentz_cone_constraints) const {
  ConstructTuplesInMemory(
      q_star, filtered_collision_pairs, 0, 1, alternation_tuples,
      lagrangian_gram_vars, verified_gram_vars, separating_plane_vars,
      separating_plane_to_tuples, separating_plane_to_lorentz_cone_constraints);
}

internal::AllocatedCertificationProgram
CspaceFreeLine::AllocateCertificationProgram() const {
  std::vector<CspaceFreeRegion::CspacePolytopeTuple> alternation_tuples;
  VectorX<symbolic::Variable> lagrangian_gram_vars, verified_gram_vars,
      separating_plane_vars;
  std::vector<std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>
      separating_plane_to_lorentz_cone_constraints;
  std::vector<std::vector<int>> separating_plane_to_tuples;
  GenerateTuplesForCertification(q_star_, filtered_collision_pairs_,
                                 &alternation_tuples, &lagrangian_gram_vars,
                                 &verified_gram_vars, &separating_plane_vars,
                                 &separating_plane_to_tuples,
                                 &separating_plane_to_lorentz_cone_constraints);

  auto prog = std::make_unique<solvers::MathematicalProgram>();

  // Adds decision variables.
  prog->AddDecisionVariables(separating_plane_vars);
  symbolic::Variables mu_variables{mu_};
  prog->AddIndeterminates(solvers::VectorIndeterminate<1>(mu_));

  // a map to store polynomials which will have s0 and s1 in its coefficients
  // for rapidly updating the coefficients for every new certification program
  std::unordered_map<
      symbolic::Polynomial,
      std::unordered_map<symbolic::Monomial,
                         solvers::Binding<solvers::LinearEqualityConstraint>>,
      std::hash<symbolic::Polynomial>, internal::ComparePolynomials>
      polynomial_to_monomial_to_binding_map;

  symbolic::Expression coefficient_expression;
  // For each rational numerator, add the constraint that the Lagrangian
  // polynomials >= 0, and the verified polynomial >= 0.
  //
  // Recall that a univariate polynomial p(μ) is nonnegative on [0, 1] if and
  // only if p(μ) = λ(μ) + ν(μ)*μ*(1-μ) if deg(p) = 2d with deg(λ) ≤ 2d and
  // deg(ν) ≤ 2d - 2 p(μ) = λ(μ)*μ + ν(μ)*(1-μ) if deg(p) = 2d + 1 with
  // deg(λ) ≤ 2d and deg(ν) ≤ 2d and λ, ν are SOS
  for (const auto& tuple : alternation_tuples) {
    symbolic::Polynomial verified_polynomial = tuple.rational_numerator;
    // In this case we just need positivity
    if (verified_polynomial.TotalDegree() == 0) {
      for (const auto& [monomial, coeff] :
           verified_polynomial.monomial_to_coefficient_map()) {
        prog->AddLinearConstraint(coeff >= 0);
      }
    } else {
      int d = verified_polynomial.TotalDegree() / 2;
      auto [l, Ql] =
          prog->NewSosPolynomial(mu_variables, 2 * d, option_.lagrangian_type);
      if (verified_polynomial.TotalDegree() % 2 == 0) {
        auto [v, Qv] = prog->NewSosPolynomial(mu_variables, 2 * d - 2,
                                              option_.lagrangian_type);
        verified_polynomial -=
            symbolic::Polynomial(l + v * mu_ * (symbolic::Polynomial(1) - mu_));
      } else {
        auto [v, Qv] = prog->NewSosPolynomial(mu_variables, 2 * d,
                                              option_.lagrangian_type);
        verified_polynomial -=
            symbolic::Polynomial(l * mu_ + v * (symbolic::Polynomial(1) - mu_));
      }
      // preallocate linear equality constraints for the zero equality awaiting
      // substitution of s0 and s1 also create a map from the monomial to the
      // appropriate linear equality constraint
      std::unordered_map<symbolic::Monomial,
                         solvers::Binding<solvers::LinearEqualityConstraint>>
          monomial_to_equality_constraint;
      verified_polynomial.SetIndeterminates({mu_});
      for (const auto& [monomial, coefficient] :
           verified_polynomial.monomial_to_coefficient_map()) {
        // construct dummy expression to ensure that the binding has the
        // appropriate variables
        coefficient_expression = 0;
        for (const auto& v : coefficient.GetVariables()) {
          coefficient_expression += v;
        }
        monomial_to_equality_constraint.insert(
            {monomial, (prog->AddLinearEqualityConstraint(0, 0))});
      }

      polynomial_to_monomial_to_binding_map.insert_or_assign(
          verified_polynomial, monomial_to_equality_constraint);
    }
  }
  return {std::move(prog), polynomial_to_monomial_to_binding_map};
}

void CspaceFreeLine::InitializeLineVariables(int num_positions) {
  s0_.resize(num_positions);
  s1_.resize(num_positions);
  for (int i = 0; i < num_positions; ++i) {
    s0_[i] = drake::symbolic::Variable("s0[" + std::to_string(i) + "]");
    s1_[i] = drake::symbolic::Variable("s1[" + std::to_string(i) + "]");
  }

  const drake::VectorX<drake::symbolic::Variable> t =
      (this->rational_forward_kinematics()).t();
  for (int i = 0;
       i < ((this->rational_forward_kinematics()).plant().num_positions());
       ++i) {
    t_to_line_subs_map_[t[i]] = mu_ * s0_[i] + (1 - mu_) * s1_[i];
  }
}

namespace internal {
AllocatedCertificationProgram::AllocatedCertificationProgram(
    std::unique_ptr<solvers::MathematicalProgram> prog,
    std::unordered_map<
        symbolic::Polynomial,
        std::unordered_map<symbolic::Monomial,
                           solvers::Binding<solvers::LinearEqualityConstraint>>,
        std::hash<symbolic::Polynomial>, internal::ComparePolynomials>
        polynomial_to_monomial_to_bindings_map)
    : prog_{std::move(prog)},
      polynomial_to_monomial_to_bindings_map_{
          polynomial_to_monomial_to_bindings_map} {};

void internal::AllocatedCertificationProgram::
    EvaluatePolynomialsAndUpdateProgram(symbolic::Environment env) {
  symbolic::Expression coefficient;
  for (auto& [polynomial, monomial_to_binding] :
       polynomial_to_monomial_to_bindings_map_) {
    symbolic::Polynomial evaluated_polynomial = polynomial.EvaluatePartial(env);
    // why do i need to do this??
    evaluated_polynomial.SetIndeterminates(polynomial.indeterminates());

    for (const auto& [monomial, binding] : monomial_to_binding) {
      prog_->RemoveConstraint(binding);
      // note that we do not explicitly check that evaluated polynomial and
      // monomial_to_bindings contain the same monomials which could be
      // dangerous.
      coefficient = evaluated_polynomial.monomial_to_coefficient_map()
                        .at(monomial)
                        .Expand();
      monomial_to_binding.insert_or_assign(
          monomial, prog_->AddLinearEqualityConstraint(coefficient, 0));
    }
  }
}

// void internal::AllocatedCertificationProgram::
//    EvaluatePolynomialsAndUpdateProgram(symbolic::Environment env) {
//  auto clock_start = std::chrono::system_clock::now();
//  //  symbolic::Expression coefficient;
//  std::for_each(
//      std::execution::par, polynomial_to_monomial_to_bindings_map_.begin(),
//      polynomial_to_monomial_to_bindings_map_.end(),
//      [this, env](auto poly_to_monom_to_binding_item) {
//        symbolic::Polynomial evaluated_polynomial =
//            poly_to_monom_to_binding_item.first.EvaluatePartial(env);
//        // why do i need to do this??
//        evaluated_polynomial.SetIndeterminates(
//            poly_to_monom_to_binding_item.first.indeterminates());
//        symbolic::Expression coefficient;
//        for (auto& [monomial, binding] : poly_to_monom_to_binding_item.second)
//        { prog_->RemoveConstraint(binding);
//        // note that we do not explicitly check that evaluated polynomial and
//        // monomial_to_bindings contain the same monomials which could be
//        // dangerous.
//        coefficient = evaluated_polynomial.monomial_to_coefficient_map()
//                          .at(monomial)
//                          .Expand();
//        try {
//          poly_to_monom_to_binding_item.second.insert_or_assign(
//              monomial, prog_->AddLinearEqualityConstraint(coefficient, 0));
//        } catch (...) {
//          std::cout << monomial << "\n";
//          std::cout << evaluated_polynomial.monomial_to_coefficient_map()
//                           .at(monomial)
//                           .Expand()
//                    << std::endl;
//          std::cout << evaluated_polynomial.decision_variables() << "\n";
//          std::cout << evaluated_polynomial.indeterminates() << "\n" <<
//          std::endl; throw;
//        }
//      }
//        // TODO(Alex.Amice) change the bindings in parallel as well
////        std::for_each(
////            std::execution::par,
////            poly_to_monom_to_binding_item.second.begin(),
////            poly_to_monom_to_binding_item.second.end(),
////            [this, evaluated_polynomial,
////             poly_to_monom_to_binding_item](auto monomial_to_binding) {
////              this->prog_->RemoveConstraint(monomial_to_binding.second);
////              symbolic::Expression coefficient =
////                  evaluated_polynomial.monomial_to_coefficient_map()
////                      .at(monomial_to_binding.first)
////                      .Expand();
////              poly_to_monom_to_binding_item.second.insert_or_assign(
////                  monomial_to_binding.first,
////                  prog_->AddLinearEqualityConstraint(coefficient, 0));
////            });
//      });
//  auto clock_now = std::chrono::system_clock::now();
//  drake::log() -> info(fmt::format(
//      "Bindings updated in {} s",
//      static_cast<float>(
//      std::chrono::duration_cast<std::chrono::milliseconds>(clock_now -
//                                                                  clock_start)
//                .count()) /
//            1000)
//      );
//}
}  // namespace internal

solvers::MathematicalProgramResult
internal::AllocatedCertificationProgram::solve(
    const solvers::SolverOptions& solver_options) {
  return solvers::Solve(*(prog_.get()), std::nullopt, solver_options);
}

}  // namespace multibody
}  // namespace drake
