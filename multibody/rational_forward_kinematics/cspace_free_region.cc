#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"

#include <algorithm>
#include <limits>
#include <optional>

#include <fmt/format.h>

#include "drake/geometry/optimization/vpolytope.h"
#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace multibody {
const double kInf = std::numeric_limits<double>::infinity();

namespace {
struct DirectedKinematicsChain {
  DirectedKinematicsChain(BodyIndex m_start, BodyIndex m_end)
      : start(m_start), end(m_end) {}

  bool operator==(const DirectedKinematicsChain& other) const {
    return start == other.start && end == other.end;
  }

  BodyIndex start;
  BodyIndex end;
};

struct DirectedKinematicsChainHash {
  size_t operator()(const DirectedKinematicsChain& p) const {
    return p.start * 100 + p.end;
  }
};

// map the kinematics chain to monomial_basis.
// If the separating plane has affine order, then the polynomial we want to
// verify (in the numerator of 1 - aᵀ(x−c) or aᵀ(x−c)-1) only contains the
// monomials tⱼ * ∏(tᵢ, dᵢ), where tᵢ is a t on the "half chain" from the
// expressed frame to either the link or the obstacle, and dᵢ<=2. Namely at
// most one variable has degree 3, all the other variables have degree <= 2.
// If the separating plane has constant order, then the polynomial we want to
// verify only contains the monomials ∏(tᵢ, dᵢ) with dᵢ<=2. In both cases, the
// monomial basis for the SOS polynomial contains all monomials that each
// variable has degree at most 1, and each variable is on the "half chain" from
// the expressed body to this link (either the robot link or the obstacle).
void FindMonomialBasisForPolytopicRegion(
    const RationalForwardKinematics& rational_forward_kinematics,
    const LinkVertexOnPlaneSideRational& rational,
    std::unordered_map<SortedPair<multibody::BodyIndex>,
                       VectorX<drake::symbolic::Monomial>>*
        map_chain_to_monomial_basis,
    VectorX<drake::symbolic::Monomial>* monomial_basis_halfchain) {
  // First check if the monomial basis for this kinematics chain has been
  // computed.
  const SortedPair<multibody::BodyIndex> kinematics_chain(
      rational.link_polytope->body_index(), rational.expressed_body_index);
  const auto it = map_chain_to_monomial_basis->find(kinematics_chain);
  if (it == map_chain_to_monomial_basis->end()) {
    const auto t_halfchain = rational_forward_kinematics.FindTOnPath(
        rational.link_polytope->body_index(), rational.expressed_body_index);
    *monomial_basis_halfchain = GenerateMonomialBasisWithOrderUpToOne(
        drake::symbolic::Variables(t_halfchain));
    map_chain_to_monomial_basis->emplace_hint(
        it, std::make_pair(kinematics_chain, *monomial_basis_halfchain));
  } else {
    *monomial_basis_halfchain = it->second;
  }
}

/**
 * For a polyhedron C * x <= d, find the lower and upper bound for x(i).
 */
void BoundPolyhedronByBox(const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
                          Eigen::VectorXd* x_lower, Eigen::VectorXd* x_upper) {
  solvers::MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables(C.cols());
  prog.AddLinearConstraint(C, Eigen::VectorXd::Constant(d.rows(), -kInf), d, x);
  Eigen::VectorXd cost_coeff = Eigen::VectorXd::Zero(x.rows());
  auto cost = prog.AddLinearCost(cost_coeff, x);
  x_lower->resize(x.rows());
  x_upper->resize(x.rows());
  for (int i = 0; i < x.rows(); ++i) {
    // Compute x_lower(i).
    cost_coeff.setZero();
    cost_coeff(i) = 1;
    cost.evaluator()->UpdateCoefficients(cost_coeff);
    auto result = solvers::Solve(prog);
    (*x_lower)(i) = result.get_optimal_cost();
    // Compute x_upper(i).
    cost_coeff(i) = -1;
    cost.evaluator()->UpdateCoefficients(cost_coeff);
    result = solvers::Solve(prog);
    (*x_upper)(i) = -result.get_optimal_cost();
  }
}

template <typename T>
void CalcDminusCt(const Eigen::Ref<const MatrixX<T>>& C,
                  const Eigen::Ref<const VectorX<T>>& d,
                  const std::vector<symbolic::Monomial>& t_monomials,
                  VectorX<symbolic::Polynomial>* d_minus_Ct) {
  // Now build the polynomials d(i) - C.row(i) * t
  DRAKE_DEMAND(C.rows() == d.rows() &&
               C.cols() == static_cast<int>(t_monomials.size()));
  d_minus_Ct->resize(C.rows());
  const symbolic::Monomial monomial_one{};
  symbolic::Polynomial::MapType d_minus_Ct_poly_map;
  for (int i = 0; i < C.rows(); ++i) {
    for (int j = 0; j < static_cast<int>(t_monomials.size()); ++j) {
      auto it = d_minus_Ct_poly_map.find(t_monomials[j]);
      if (it == d_minus_Ct_poly_map.end()) {
        d_minus_Ct_poly_map.emplace_hint(it, t_monomials[j], -C(i, j));
      } else {
        it->second = -C(i, j);
      }
    }
    auto it = d_minus_Ct_poly_map.find(monomial_one);
    if (it == d_minus_Ct_poly_map.end()) {
      d_minus_Ct_poly_map.emplace_hint(it, monomial_one, d(i));
    } else {
      it->second = d(i);
    }
    (*d_minus_Ct)(i) = symbolic::Polynomial(d_minus_Ct_poly_map);
  }
}
}  // namespace

CspaceFreeRegion::CspaceFreeRegion(
    const systems::Diagram<double>& diagram,
    const multibody::MultibodyPlant<double>* plant,
    const geometry::SceneGraph<double>* scene_graph,
    SeparatingPlaneOrder plane_order, CspaceRegionType cspace_region_type)
    : rational_forward_kinematics_(*plant),
      scene_graph_{scene_graph},
      polytope_geometries_{GetConvexPolytopes(diagram, plant, scene_graph)},
      plane_order_{plane_order},
      cspace_region_type_{cspace_region_type} {
  // Now create the separating planes.
  std::map<SortedPair<BodyIndex>,
           std::vector<std::pair<const ConvexPolytope*, const ConvexPolytope*>>>
      collision_pairs;
  int num_collision_pairs = 0;
  const auto& model_inspector = scene_graph->model_inspector();
  for (const auto& [link1, polytopes1] : polytope_geometries_) {
    for (const auto& [link2, polytopes2] : polytope_geometries_) {
      if (link1 < link2) {
        // link_collision_pairs stores all the pair of collision geometry on
        // (link1, link2).
        std::vector<std::pair<const ConvexPolytope*, const ConvexPolytope*>>
            link_collision_pairs;
        for (const auto& polytope1 : polytopes1) {
          for (const auto& polytope2 : polytopes2) {
            if (!model_inspector.CollisionFiltered(polytope1.get_id(),
                                                   polytope2.get_id())) {
              num_collision_pairs++;
              link_collision_pairs.emplace_back(&polytope1, &polytope2);
            }
          }
        }
        collision_pairs.emplace_hint(collision_pairs.end(),
                                     SortedPair<BodyIndex>(link1, link2),
                                     link_collision_pairs);
      }
    }
  }
  separating_planes_.reserve(num_collision_pairs);
  // If we verify an axis-algined bounding box in the C-space, then for each
  // pair of (link, obstacle), then we only need to consider variables t for the
  // joints on the kinematics chain between the link and the obstacle; if we
  // verify a generic bounding box, then we need to consider t for all joint
  // angles.
  std::unordered_map<SortedPair<BodyIndex>, VectorX<symbolic::Variable>>
      map_link_obstacle_to_t;
  for (const auto& [link_pair, polytope_pairs] : collision_pairs) {
    for (const auto& polytope_pair : polytope_pairs) {
      Vector3<symbolic::Expression> a;
      symbolic::Expression b;
      const symbolic::Monomial monomial_one{};
      VectorX<symbolic::Variable> plane_decision_vars;
      if (plane_order_ == SeparatingPlaneOrder::kConstant) {
        plane_decision_vars.resize(4);
        for (int i = 0; i < 3; ++i) {
          plane_decision_vars(i) = symbolic::Variable(
              "a" + std::to_string(separating_planes_.size() * 3 + i));
          plane_decision_vars(3) = symbolic::Variable(
              "b" + std::to_string(separating_planes_.size()));
        }
        a = plane_decision_vars.head<3>().cast<symbolic::Expression>();
        b = plane_decision_vars(3);
      } else if (plane_order_ == SeparatingPlaneOrder::kAffine) {
        VectorX<symbolic::Variable> t_for_plane;
        if (cspace_region_type_ == CspaceRegionType::kGenericPolytope) {
          t_for_plane = rational_forward_kinematics_.t();
        } else if (cspace_region_type_ ==
                   CspaceRegionType::kAxisAlignedBoundingBox) {
          auto it = map_link_obstacle_to_t.find(link_pair);
          if (it == map_link_obstacle_to_t.end()) {
            t_for_plane = rational_forward_kinematics_.FindTOnPath(
                link_pair.first(), link_pair.second());
            map_link_obstacle_to_t.emplace_hint(it, link_pair, t_for_plane);
          } else {
            t_for_plane = it->second;
          }
        }
        // Now create the variable a_coeff, a_constant, b_coeff, b_constant,
        // such that a = a_coeff * t_for_plane + a_constant, and b = b_coeff *
        // t_for_plane + b_constant.
        Matrix3X<symbolic::Variable> a_coeff(3, t_for_plane.rows());
        Vector3<symbolic::Variable> a_constant;
        for (int i = 0; i < 3; ++i) {
          a_constant(i) = symbolic::Variable(
              fmt::format("a_constant{}({})", separating_planes_.size(), i));
          for (int j = 0; j < t_for_plane.rows(); ++j) {
            a_coeff(i, j) = symbolic::Variable(fmt::format(
                "a_coeff{}({}, {})", separating_planes_.size(), i, j));
          }
        }
        VectorX<symbolic::Variable> b_coeff(t_for_plane.rows());
        for (int i = 0; i < t_for_plane.rows(); ++i) {
          b_coeff(i) = symbolic::Variable(
              fmt::format("b_coeff{}({})", separating_planes_.size(), i));
        }
        symbolic::Variable b_constant(
            fmt::format("b_constant{}", separating_planes_.size()));
        // Now construct a(i) = a_coeff.row(i) * t_for_plane + a_constant(i).
        a = a_coeff * t_for_plane + a_constant;
        b = b_coeff.cast<symbolic::Expression>().dot(t_for_plane) + b_constant;
        // Now put a_coeff, a_constant, b_coeff, b_constant to
        // plane_decision_vars.
        plane_decision_vars.resize(4 * t_for_plane.rows() + 4);
        int var_count = 0;
        for (int i = 0; i < 3; ++i) {
          plane_decision_vars.segment(var_count, t_for_plane.rows()) =
              a_coeff.row(i);
          var_count += t_for_plane.rows();
        }
        plane_decision_vars.segment<3>(var_count) = a_constant;
        var_count += 3;
        plane_decision_vars.segment(var_count, t_for_plane.rows()) = b_coeff;
        var_count += t_for_plane.rows();
        plane_decision_vars(var_count) = b_constant;
        var_count++;
      }
      separating_planes_.emplace_back(
          a, b, polytope_pair.first, polytope_pair.second,
          internal::FindBodyInTheMiddleOfChain(*plant, link_pair.first(),
                                               link_pair.second()),
          plane_order_, plane_decision_vars);
      map_polytopes_to_separating_planes_.emplace(
          SortedPair<ConvexGeometry::Id>(polytope_pair.first->get_id(),
                                         polytope_pair.second->get_id()),
          &(separating_planes_[separating_planes_.size() - 1]));
    }
  }
}

std::vector<LinkVertexOnPlaneSideRational>
CspaceFreeRegion::GenerateLinkOnOneSideOfPlaneRationals(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs)
    const {
  std::unordered_map<DirectedKinematicsChain,
                     RationalForwardKinematics::Pose<symbolic::Polynomial>,
                     DirectedKinematicsChainHash>
      body_pair_to_X_AB_multilinear;
  std::vector<LinkVertexOnPlaneSideRational> rationals;
  for (const auto& separating_plane : separating_planes_) {
    if (!IsGeometryPairCollisionIgnored(
            separating_plane.positive_side_polytope->get_id(),
            separating_plane.negative_side_polytope->get_id(),
            filtered_collision_pairs)) {
      // First compute X_AB for both side of the polytopes.
      for (const PlaneSide plane_side :
           {PlaneSide::kPositive, PlaneSide::kNegative}) {
        const ConvexPolytope* polytope;
        const ConvexPolytope* other_side_polytope;
        if (plane_side == PlaneSide::kPositive) {
          polytope = separating_plane.positive_side_polytope;
          other_side_polytope = separating_plane.negative_side_polytope;
        } else {
          polytope = separating_plane.negative_side_polytope;
          other_side_polytope = separating_plane.positive_side_polytope;
        }
        const DirectedKinematicsChain expressed_to_link(
            separating_plane.expressed_link, polytope->body_index());
        auto it = body_pair_to_X_AB_multilinear.find(expressed_to_link);
        if (it == body_pair_to_X_AB_multilinear.end()) {
          body_pair_to_X_AB_multilinear.emplace_hint(
              it, expressed_to_link,
              rational_forward_kinematics_.CalcLinkPoseAsMultilinearPolynomial(
                  q_star, polytope->body_index(),
                  separating_plane.expressed_link));
        }
        it = body_pair_to_X_AB_multilinear.find(expressed_to_link);
        const RationalForwardKinematics::Pose<symbolic::Polynomial>&
            X_AB_multilinear = it->second;
        const std::vector<LinkVertexOnPlaneSideRational>
            rationals_expressed_to_link =
                GenerateLinkOnOneSideOfPlaneRationalFunction(
                    rational_forward_kinematics_, polytope, other_side_polytope,
                    X_AB_multilinear, separating_plane.a, separating_plane.b,
                    plane_side, separating_plane.order);
        // I cannot use "insert" function to append vectors, since
        // LinkVertexOnPlaneSideRational contains const members, hence it does
        // not have an assignment operator.
        std::copy(rationals_expressed_to_link.begin(),
                  rationals_expressed_to_link.end(),
                  std::back_inserter(rationals));
      }
    }
  }
  return rationals;
}

namespace {
// Given t[i], t_lower and t_upper, construct the polynomial t - t_lower and
// t_upper - t.
void ConstructTBoundsPolynomial(
    const std::vector<symbolic::Monomial>& t_monomial,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    VectorX<symbolic::Polynomial>* t_minus_t_lower,
    VectorX<symbolic::Polynomial>* t_upper_minus_t) {
  const symbolic::Monomial monomial_one{};
  t_minus_t_lower->resize(t_monomial.size());
  t_upper_minus_t->resize(t_monomial.size());
  for (int i = 0; i < static_cast<int>(t_monomial.size()); ++i) {
    const symbolic::Polynomial::MapType map_lower{
        {{t_monomial[i], 1}, {monomial_one, -t_lower(i)}}};
    (*t_minus_t_lower)(i) = symbolic::Polynomial(map_lower);
    const symbolic::Polynomial::MapType map_upper{
        {{t_monomial[i], -1}, {monomial_one, t_upper(i)}}};
    (*t_upper_minus_t)(i) = symbolic::Polynomial(map_upper);
  }
}

/**
 * Impose the constraint
 * l_polytope(t) >= 0
 * l_lower(t)>=0
 * l_upper(t)>=0
 * p(t) - l_polytope(t)ᵀ(d - C*t) - l_lower(t)ᵀ(t-t_lower) -
 * l_upper(t)ᵀ(t_upper-t) >=0
 * where l_polytope, l_lower, l_upper are Lagrangian
 * multipliers. p(t) is the numerator of polytope_on_one_side_rational
 * @param monomial_basis The monomial basis for all non-negative polynomials
 * above.
 * @param t_lower_needs_lagrangian If t_lower_needs_lagrangian[i]=false, then
 * lagrangian_lower(i) = 0
 * @param t_upper_needs_lagrangian If t_upper_needs_lagrangian[i]=false, then
 * lagrangian_upper(i) = 0
 * @param[out] lagrangian_polytope l_polytope(t).
 * @param[out] lagrangian_lower l_lower(t).
 * @param[out] lagrangian_upper l_upper(t).
 * @param[out] verified_polynomial p(t) - l_polytope(t)ᵀ(d - C*t) -
 * l_lower(t)ᵀ(t-t_lower) - l_upper(t)ᵀ(t_upper-t)
 */
void AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
    solvers::MathematicalProgram* prog,
    const symbolic::RationalFunction& polytope_on_one_side_rational,
    const VectorX<symbolic::Polynomial>& d_minus_Ct,
    const VectorX<symbolic::Polynomial>& t_minus_t_lower,
    const VectorX<symbolic::Polynomial>& t_upper_minus_t,
    const VectorX<symbolic::Monomial>& monomial_basis,
    const VerificationOption& verification_option,
    const std::vector<bool>& t_lower_needs_lagrangian,
    const std::vector<bool>& t_upper_needs_lagrangian,
    VectorX<symbolic::Polynomial>* lagrangian_polytope,
    VectorX<symbolic::Polynomial>* lagrangian_lower,
    VectorX<symbolic::Polynomial>* lagrangian_upper,
    symbolic::Polynomial* verified_polynomial) {
  lagrangian_polytope->resize(d_minus_Ct.rows());
  lagrangian_lower->resize(t_minus_t_lower.rows());
  lagrangian_upper->resize(t_upper_minus_t.rows());
  *verified_polynomial = polytope_on_one_side_rational.numerator();
  for (int i = 0; i < d_minus_Ct.rows(); ++i) {
    (*lagrangian_polytope)(i) =
        prog->NewNonnegativePolynomial(monomial_basis,
                                       verification_option.lagrangian_type)
            .first;
    *verified_polynomial -= (*lagrangian_polytope)(i)*d_minus_Ct(i);
  }
  for (int i = 0; i < t_minus_t_lower.rows(); ++i) {
    if (t_lower_needs_lagrangian[i]) {
      (*lagrangian_lower)(i) =
          prog->NewNonnegativePolynomial(monomial_basis,
                                         verification_option.lagrangian_type)
              .first;
      *verified_polynomial -= (*lagrangian_lower)(i)*t_minus_t_lower(i);
    } else {
      (*lagrangian_lower)(i) = symbolic::Polynomial();
    }
  }
  for (int i = 0; i < t_upper_minus_t.rows(); ++i) {
    if (t_upper_needs_lagrangian[i]) {
      (*lagrangian_upper)(i) =
          prog->NewNonnegativePolynomial(monomial_basis,
                                         verification_option.lagrangian_type)
              .first;
      *verified_polynomial -= (*lagrangian_upper)(i)*t_upper_minus_t(i);
    } else {
      (*lagrangian_upper)(i) = symbolic::Polynomial();
    }
  }

  const symbolic::Polynomial verified_polynomial_expected =
      prog->NewNonnegativePolynomial(monomial_basis,
                                     verification_option.link_polynomial_type)
          .first;
  const symbolic::Polynomial poly_diff{*verified_polynomial -
                                       verified_polynomial_expected};
  for (const auto& term : poly_diff.monomial_to_coefficient_map()) {
    prog->AddLinearEqualityConstraint(term.second, 0);
  }
}
}  // namespace

CspaceFreeRegion::CspacePolytopeProgramReturn
CspaceFreeRegion::ConstructProgramForCspacePolytope(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const std::vector<LinkVertexOnPlaneSideRational>& rationals,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const VerificationOption& verification_option) const {
  DRAKE_DEMAND(cspace_region_type_ == CspaceRegionType::kGenericPolytope);
  CspaceFreeRegion::CspacePolytopeProgramReturn ret(rationals.size());
  // Add t as indeterminates
  const auto& t = rational_forward_kinematics_.t();
  ret.prog->AddIndeterminates(t);
  // Add separating planes as decision variables.
  for (const auto& separating_plane : separating_planes_) {
    if (!IsGeometryPairCollisionIgnored(
            separating_plane.positive_side_polytope->get_id(),
            separating_plane.negative_side_polytope->get_id(),
            filtered_collision_pairs)) {
      ret.prog->AddDecisionVariables(separating_plane.decision_variables);
    }
  }
  // Now build the polynomials d(i) - C.row(i) * t
  VectorX<symbolic::Polynomial> d_minus_Ct_polynomial(C.rows());
  std::vector<symbolic::Monomial> t_monomials;
  t_monomials.reserve(t.rows());
  for (int i = 0; i < t.rows(); ++i) {
    t_monomials.emplace_back(t(i));
  }
  CalcDminusCt<double>(C, d, t_monomials, &d_minus_Ct_polynomial);

  // Build the polynomial for t-t_lower and t_upper-t
  Eigen::VectorXd t_lower, t_upper;
  ComputeBoundsOnT(
      q_star, rational_forward_kinematics_.plant().GetPositionLowerLimits(),
      rational_forward_kinematics_.plant().GetPositionUpperLimits(), &t_lower,
      &t_upper);
  // If C * t <= d already implies t(i) <= t_upper(i) or t(i) >= t_lower(i) for
  // some t, then we don't need to add the lagrangian multiplier for that
  // t_upper(i) - t(i) or t(i) - t_lower(i).
  Eigen::VectorXd t_lower_from_polytope, t_upper_from_polytope;
  BoundPolyhedronByBox(C, d, &t_lower_from_polytope, &t_upper_from_polytope);
  std::vector<bool> t_lower_needs_lagrangian(t.rows(), true);
  std::vector<bool> t_upper_needs_lagrangian(t.rows(), true);
  for (int i = 0; i < t.rows(); ++i) {
    if (t_lower(i) < t_lower_from_polytope(i)) {
      t_lower_needs_lagrangian[i] = false;
    }
    if (t_upper(i) > t_upper_from_polytope(i)) {
      t_upper_needs_lagrangian[i] = false;
    }
  }
  VectorX<symbolic::Polynomial> t_minus_t_lower_poly(t.rows());
  VectorX<symbolic::Polynomial> t_upper_minus_t_poly(t.rows());
  ConstructTBoundsPolynomial(t_monomials, t_lower, t_upper,
                             &t_minus_t_lower_poly, &t_upper_minus_t_poly);

  // Get the monomial basis for each kinematics chain.
  std::unordered_map<SortedPair<multibody::BodyIndex>,
                     VectorX<symbolic::Monomial>>
      map_chain_to_monomial_basis;
  for (int i = 0; i < static_cast<int>(rationals.size()); ++i) {
    VectorX<symbolic::Monomial> monomial_basis_chain;
    FindMonomialBasisForPolytopicRegion(
        rational_forward_kinematics_, rationals[i],
        &map_chain_to_monomial_basis, &monomial_basis_chain);
    // Now add the constraint that C*t<=d and t_lower <= t <= t_upper implies
    // the rational being nonnegative.
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        ret.prog.get(), rationals[i].rational, d_minus_Ct_polynomial,
        t_minus_t_lower_poly, t_upper_minus_t_poly, monomial_basis_chain,
        verification_option, t_lower_needs_lagrangian, t_upper_needs_lagrangian,
        &(ret.polytope_lagrangians[i]), &(ret.t_lower_lagrangians[i]),
        &(ret.t_upper_lagrangians[i]), &(ret.verified_polynomials[i]));
  }
  return ret;
}

bool CspaceFreeRegion::IsPostureInCollision(
    const systems::Context<double>& context) const {
  const auto& plant = rational_forward_kinematics_.plant();
  drake::math::RigidTransform<double> X_WB1;
  drake::math::RigidTransform<double> X_WB2;
  const auto& model_inspector = scene_graph_->model_inspector();
  for (const auto& [link1, polytopes1] : polytope_geometries_) {
    X_WB1 = plant.EvalBodyPoseInWorld(context, plant.get_body(link1));
    for (const auto& [link2, polytopes2] : polytope_geometries_) {
      X_WB2 = plant.EvalBodyPoseInWorld(context, plant.get_body(link2));
      for (const auto& polytope1 : polytopes1) {
        for (const auto& polytope2 : polytopes2) {
          if (!model_inspector.CollisionFiltered(polytope1.get_id(),
                                                 polytope2.get_id()) &&
              polytope1.IsInCollision(polytope2, X_WB1, X_WB2)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void CspaceFreeRegion::GenerateTuplesForBilinearAlternation(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs, int C_rows,
    std::vector<CspaceFreeRegion::CspacePolytopeTuple>* alternation_tuples,
    VectorX<symbolic::Polynomial>* d_minus_Ct, Eigen::VectorXd* t_lower,
    Eigen::VectorXd* t_upper, VectorX<symbolic::Polynomial>* t_minus_t_lower,
    VectorX<symbolic::Polynomial>* t_upper_minus_t,
    MatrixX<symbolic::Variable>* C, VectorX<symbolic::Variable>* d,
    VectorX<symbolic::Variable>* lagrangian_gram_vars,
    VectorX<symbolic::Variable>* verified_gram_vars,
    VectorX<symbolic::Variable>* separating_plane_vars) const {
  // Create variables C and d.
  const auto& t = rational_forward_kinematics_.t();
  C->resize(C_rows, t.rows());
  d->resize(C_rows);
  for (int i = 0; i < C_rows; ++i) {
    for (int j = 0; j < t.rows(); ++j) {
      (*C)(i, j) = symbolic::Variable(fmt::format("C({}, {})", i, j));
    }
    (*d)(i) = symbolic::Variable(fmt::format("d({})", i));
  }
  std::vector<symbolic::Monomial> t_monomials;
  t_monomials.reserve(t.rows());
  for (int i = 0; i < t.rows(); ++i) {
    t_monomials.emplace_back(t(i));
  }
  CalcDminusCt<symbolic::Variable>(*C, *d, t_monomials, d_minus_Ct);

  // Build the polynomial for t-t_lower and t_upper-t
  ComputeBoundsOnT(
      q_star, rational_forward_kinematics_.plant().GetPositionLowerLimits(),
      rational_forward_kinematics_.plant().GetPositionUpperLimits(), t_lower,
      t_upper);
  ConstructTBoundsPolynomial(t_monomials, *t_lower, *t_upper, t_minus_t_lower,
                             t_upper_minus_t);

  // Build tuples.
  const auto rationals =
      GenerateLinkOnOneSideOfPlaneRationals(q_star, filtered_collision_pairs);
  alternation_tuples->reserve(rationals.size());
  // Get the monomial basis for each kinematics chain.
  std::unordered_map<SortedPair<multibody::BodyIndex>,
                     VectorX<symbolic::Monomial>>
      map_chain_to_monomial_basis;
  // Count the total number of variables for all Gram matrices and allocate the
  // memory for once. It is time consuming to allocate each Gram matrix
  // variables within a for loop.
  // Also count the total number of variables for all separating plane decision
  // variables.
  int lagrangian_gram_vars_count = 0;
  int verified_gram_vars_count = 0;
  for (int i = 0; i < static_cast<int>(rationals.size()); ++i) {
    VectorX<symbolic::Monomial> monomial_basis_chain;
    FindMonomialBasisForPolytopicRegion(
        rational_forward_kinematics_, rationals[i],
        &map_chain_to_monomial_basis, &monomial_basis_chain);
    std::vector<int> polytope_lagrangian_gram_lower_start(C->rows());
    const int gram_lower_size =
        monomial_basis_chain.rows() * (monomial_basis_chain.rows() + 1) / 2;
    for (int j = 0; j < C->rows(); ++j) {
      polytope_lagrangian_gram_lower_start[j] =
          lagrangian_gram_vars_count + j * gram_lower_size;
    }
    std::vector<int> t_lower_lagrangian_gram_lower_start(t.rows());
    for (int j = 0; j < t.rows(); ++j) {
      t_lower_lagrangian_gram_lower_start[j] =
          lagrangian_gram_vars_count + (C->rows() + j) * gram_lower_size;
    }
    std::vector<int> t_upper_lagrangian_gram_lower_start(t.rows());
    for (int j = 0; j < t.rows(); ++j) {
      t_upper_lagrangian_gram_lower_start[j] =
          lagrangian_gram_vars_count +
          (C->rows() + t.rows() + j) * gram_lower_size;
    }
    alternation_tuples->emplace_back(
        rationals[i].rational.numerator(), polytope_lagrangian_gram_lower_start,
        t_lower_lagrangian_gram_lower_start,
        t_upper_lagrangian_gram_lower_start, verified_gram_vars_count,
        monomial_basis_chain);
    // Each Gram matrix is of size monomial_basis_chain.rows() *
    // (monomial_basis_chain.rows() + 1) / 2. Each rational needs C.rows() + 2 *
    // t.rows() Lagrangians.
    lagrangian_gram_vars_count += gram_lower_size * (C_rows + 2 * t.rows());
    verified_gram_vars_count +=
        monomial_basis_chain.rows() * (monomial_basis_chain.rows() + 1) / 2;
  }
  lagrangian_gram_vars->resize(lagrangian_gram_vars_count);
  for (int i = 0; i < lagrangian_gram_vars_count; ++i) {
    (*lagrangian_gram_vars)(i) =
        symbolic::Variable(fmt::format("l_gram({})", i));
  }
  verified_gram_vars->resize(verified_gram_vars_count);
  for (int i = 0; i < verified_gram_vars_count; ++i) {
    (*verified_gram_vars)(i) =
        symbolic::Variable(fmt::format("verified_gram({})", i));
  }
  // Set separating_plane_vars.
  int separating_plane_vars_count = 0;
  for (const auto& separating_plane : separating_planes_) {
    separating_plane_vars_count += separating_plane.decision_variables.rows();
  }
  separating_plane_vars->resize(separating_plane_vars_count);
  separating_plane_vars_count = 0;
  for (const auto& separating_plane : separating_planes_) {
    separating_plane_vars->segment(separating_plane_vars_count,
                                   separating_plane.decision_variables.rows()) =
        separating_plane.decision_variables;
    separating_plane_vars_count += separating_plane.decision_variables.rows();
  }
}

std::unique_ptr<solvers::MathematicalProgram>
CspaceFreeRegion::ConstructLagrangianProgram(
    const std::vector<CspacePolytopeTuple>& alternation_tuples,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const VectorX<symbolic::Variable>& lagrangian_gram_vars,
    const VectorX<symbolic::Variable>& verified_gram_vars,
    const VectorX<symbolic::Variable>& separating_plane_vars,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    const VerificationOption& option, MatrixX<symbolic::Variable>* P,
    VectorX<symbolic::Variable>* q) const {
  // TODO(hongkai.dai): support more nonnegative polynomials.
  if (option.lagrangian_type !=
      solvers::MathematicalProgram::NonnegativePolynomial::kSos) {
    throw std::runtime_error("Only support sos polynomial for now");
  }
  if (option.link_polynomial_type !=
      solvers::MathematicalProgram::NonnegativePolynomial::kSos) {
    throw std::runtime_error("Only support sos polynomial for now");
  }
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  // Adds decision variables.
  prog->AddDecisionVariables(lagrangian_gram_vars);
  prog->AddDecisionVariables(verified_gram_vars);
  prog->AddDecisionVariables(separating_plane_vars);

  // Compute d-C*t, t - t_lower and t_upper - t.
  const auto& t = rational_forward_kinematics_.t();
  std::vector<symbolic::Monomial> t_monomials;
  t_monomials.reserve(t.rows());
  for (int i = 0; i < t.rows(); ++i) {
    t_monomials.emplace_back(t(i));
  }
  VectorX<symbolic::Polynomial> d_minus_Ct;
  CalcDminusCt<double>(C, d, t_monomials, &d_minus_Ct);
  VectorX<symbolic::Polynomial> t_minus_t_lower(t.rows());
  VectorX<symbolic::Polynomial> t_upper_minus_t(t.rows());
  ConstructTBoundsPolynomial(t_monomials, t_lower, t_upper, &t_minus_t_lower,
                             &t_upper_minus_t);
  // For each rational numerator, add the constraint that the Lagrangian
  // polynomials >= 0, and the verified polynomial >= 0.
  //
  // Within each rational, all the lagrangians and the verified polynomial has
  // same gram size. This gram size only depends on the number of joints on the
  // kinematics chain, hence we can reuse the same gram matrix without
  // reallocating the memory.
  std::unordered_map<int, MatrixX<symbolic::Variable>> size_to_gram;
  for (const auto& tuple : alternation_tuples) {
    const int gram_rows = tuple.monomial_basis.rows();
    const int gram_lower_size = gram_rows * (gram_rows + 1) / 2;
    symbolic::Polynomial verified_polynomial = tuple.rational_numerator;
    auto it = size_to_gram.find(gram_rows);
    if (it == size_to_gram.end()) {
      it = size_to_gram.emplace_hint(
          it, gram_rows, MatrixX<symbolic::Variable>(gram_rows, gram_rows));
    }
    MatrixX<symbolic::Variable>& gram_mat = it->second;

    // This lambda does three things.
    // 1. Compute the Gram matrix.
    // 2. Constraint the Gram matrix to be PSD.
    // 3. subtract lagrangian(t) * constraint_polynomial from
    // verified_polynomial.
    auto constrain_lagrangian =
        [&gram_mat, &verified_polynomial, &lagrangian_gram_vars, &prog,
         gram_lower_size](int lagrangian_gram_lower_start,
                          const VectorX<symbolic::Monomial>& monomial_basis,
                          const symbolic::Polynomial& constraint_polynomial) {
          SymmetricMatrixFromLower<symbolic::Variable>(
              gram_mat.rows(),
              lagrangian_gram_vars.segment(lagrangian_gram_lower_start,
                                           gram_lower_size),
              &gram_mat);
          prog->AddPositiveSemidefiniteConstraint(gram_mat);
          verified_polynomial -= CalcPolynomialFromGram<symbolic::Variable>(
                                     monomial_basis, gram_mat) *
                                 constraint_polynomial;
        };
    // Handle lagrangian l_polytope(t).
    for (int i = 0; i < C.rows(); ++i) {
      constrain_lagrangian(tuple.polytope_lagrangian_gram_lower_start[i],
                           tuple.monomial_basis, d_minus_Ct(i));
    }
    // Handle lagrangian l_lower(t) and l_upper(t).
    for (int i = 0; i < t.rows(); ++i) {
      constrain_lagrangian(tuple.t_lower_lagrangian_gram_lower_start[i],
                           tuple.monomial_basis, t_minus_t_lower(i));
      constrain_lagrangian(tuple.t_upper_lagrangian_gram_lower_start[i],
                           tuple.monomial_basis, t_upper_minus_t(i));
    }
    // Now constrain that verified_polynomial is non-negative.
    SymmetricMatrixFromLower<symbolic::Variable>(
        gram_rows,
        verified_gram_vars.segment(tuple.verified_polynomial_gram_lower_start,
                                   gram_lower_size),
        &gram_mat);
    prog->AddPositiveSemidefiniteConstraint(gram_mat);
    const symbolic::Polynomial verified_polynomial_expected =
        CalcPolynomialFromGram<symbolic::Variable>(tuple.monomial_basis,
                                                   gram_mat);
    const symbolic::Polynomial poly_diff{verified_polynomial -
                                         verified_polynomial_expected};
    for (const auto& item : poly_diff.monomial_to_coefficient_map()) {
      prog->AddLinearEqualityConstraint(item.second, 0);
    }
  }
  if (P != nullptr && q != nullptr) {
    *P = prog->NewSymmetricContinuousVariables(t.rows(), "P");
    *q = prog->NewContinuousVariables(t.rows(), "q");
    AddInscribedEllipsoid(prog.get(), C, d, t_lower, t_upper, *P, *q, false);
  }
  return prog;
}

std::unique_ptr<solvers::MathematicalProgram>
CspaceFreeRegion::ConstructPolytopeProgram(
    const std::vector<CspaceFreeRegion::CspacePolytopeTuple>&
        alternation_tuples,
    const MatrixX<symbolic::Variable>& C, const VectorX<symbolic::Variable>& d,
    const VectorX<symbolic::Polynomial>& d_minus_Ct,
    const Eigen::VectorXd& lagrangian_gram_var_vals,
    const VectorX<symbolic::Variable>& verified_gram_vars,
    const VectorX<symbolic::Variable>& separating_plane_vars,
    const VectorX<symbolic::Polynomial>& t_minus_t_lower,
    const VectorX<symbolic::Polynomial>& t_upper_minus_t,
    const Eigen::MatrixXd& P, const Eigen::VectorXd& q,
    const VerificationOption& option,
    VectorX<symbolic::Variable>* margin) const {
  if (option.link_polynomial_type !=
      solvers::MathematicalProgram::NonnegativePolynomial::kSos) {
    throw std::runtime_error("Only support sos polynomial for now");
  }
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  // Add the decision variables.
  prog->AddDecisionVariables(Eigen::Map<const VectorX<symbolic::Variable>>(
      C.data(), C.rows() * C.cols()));
  prog->AddDecisionVariables(d);
  prog->AddDecisionVariables(verified_gram_vars);
  prog->AddDecisionVariables(separating_plane_vars);

  // For each rational numerator, we will impose positivity (like PSD matrix)
  // constraint on its Gram matrix. This gram size only depends on the number of
  // joints on the kinematics chain, hence we can reuse the same gram matrix
  // without reallocating the memory.
  std::unordered_map<int, MatrixX<symbolic::Variable>> size_to_gram;
  for (const auto& tuple : alternation_tuples) {
    symbolic::Polynomial verified_polynomial = tuple.rational_numerator;
    const auto& monomial_basis = tuple.monomial_basis;
    const int gram_rows = monomial_basis.rows();
    const int gram_lower_size = gram_rows * (gram_rows + 1) / 2;
    // add_lagrangian adds the term -lagrangian(t) * constraint(t) to
    // verified_polynomial.
    auto add_lagrangian = [&verified_polynomial, &lagrangian_gram_var_vals,
                           &monomial_basis, gram_lower_size](
                              int lagrangian_var_start,
                              const symbolic::Polynomial& constraint) {
      verified_polynomial -=
          CalcPolynomialFromGramLower<double>(
              monomial_basis, lagrangian_gram_var_vals.segment(
                                  lagrangian_var_start, gram_lower_size)) *
          constraint;
    };

    for (int i = 0; i < C.rows(); ++i) {
      add_lagrangian(tuple.polytope_lagrangian_gram_lower_start[i],
                     d_minus_Ct(i));
    }
    for (int i = 0; i < rational_forward_kinematics_.t().rows(); ++i) {
      add_lagrangian(tuple.t_lower_lagrangian_gram_lower_start[i],
                     t_minus_t_lower(i));
      add_lagrangian(tuple.t_upper_lagrangian_gram_lower_start[i],
                     t_upper_minus_t(i));
    }
    auto it = size_to_gram.find(gram_rows);
    if (it == size_to_gram.end()) {
      it = size_to_gram.emplace_hint(
          it, gram_rows, MatrixX<symbolic::Variable>(gram_rows, gram_rows));
    }
    MatrixX<symbolic::Variable>& verified_gram = it->second;
    SymmetricMatrixFromLower<symbolic::Variable>(
        tuple.monomial_basis.rows(),
        verified_gram_vars.segment(tuple.verified_polynomial_gram_lower_start,
                                   gram_lower_size),
        &verified_gram);
    prog->AddPositiveSemidefiniteConstraint(verified_gram);
    const symbolic::Polynomial verified_polynomial_expected =
        CalcPolynomialFromGram<symbolic::Variable>(tuple.monomial_basis,
                                                   verified_gram);
    const symbolic::Polynomial poly_diff{verified_polynomial -
                                         verified_polynomial_expected};
    for (const auto& item : poly_diff.monomial_to_coefficient_map()) {
      prog->AddLinearEqualityConstraint(item.second, 0);
    }
  }

  *margin = prog->NewContinuousVariables(C.rows(), "margin");
  AddOuterPolytope(prog.get(), P, q, C, d, *margin);
  // margin >= 0.
  prog->AddBoundingBoxConstraint(0, kInf, *margin);
  return prog;
}

void CspaceFreeRegion::CspacePolytopeBilinearAlternation(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs,
    const Eigen::Ref<const Eigen::MatrixXd>& C_init,
    const Eigen::Ref<const Eigen::VectorXd>& d_init,
    const CspaceFreeRegion::BilinearAlternationOption&
        bilinear_alternation_option,
    const solvers::SolverOptions& solver_options, Eigen::MatrixXd* C_final,
    Eigen::VectorXd* d_final, Eigen::MatrixXd* P_final,
    Eigen::VectorXd* q_final) const {
  const int C_rows = C_init.rows();
  DRAKE_DEMAND(d_init.rows() == C_rows);
  DRAKE_DEMAND(C_init.cols() == rational_forward_kinematics_.t().rows());
  // First normalize each row of C and d, such that each row of C has a unit
  // norm. This is important as later when we search for polytope, we impose the
  // constraint |C.row(i)|<=1, hence we need to first start with C and d
  // satisfying this constraint.
  Eigen::MatrixXd C_val = C_init;
  Eigen::VectorXd d_val = d_init;
  for (int i = 0; i < C_rows; ++i) {
    const double C_row_norm = C_val.row(i).norm();
    C_val.row(i) /= C_row_norm;
    d_val(i) /= C_row_norm;
  }
  std::vector<CspaceFreeRegion::CspacePolytopeTuple> alternation_tuples;
  VectorX<symbolic::Polynomial> d_minus_Ct;
  Eigen::VectorXd t_lower, t_upper;
  VectorX<symbolic::Polynomial> t_minus_t_lower, t_upper_minus_t;
  MatrixX<symbolic::Variable> C_var;
  VectorX<symbolic::Variable> d_var, lagrangian_gram_vars, verified_gram_vars,
      separating_plane_vars;
  GenerateTuplesForBilinearAlternation(
      q_star, filtered_collision_pairs, C_rows, &alternation_tuples,
      &d_minus_Ct, &t_lower, &t_upper, &t_minus_t_lower, &t_upper_minus_t,
      &C_var, &d_var, &lagrangian_gram_vars, &verified_gram_vars,
      &separating_plane_vars);

  MatrixX<symbolic::Variable> P;
  VectorX<symbolic::Variable> q;
  VectorX<symbolic::Variable> margin;
  int iter_count = 0;
  double cost_improvement = kInf;
  double previous_cost = -kInf;
  VerificationOption verification_option{};
  while (iter_count < bilinear_alternation_option.max_iters &&
         cost_improvement > bilinear_alternation_option.convergence_tol) {
    auto prog_lagrangian = ConstructLagrangianProgram(
        alternation_tuples, C_val, d_val, lagrangian_gram_vars,
        verified_gram_vars, separating_plane_vars, t_lower, t_upper,
        verification_option, &P, &q);
    prog_lagrangian->AddMaximizeLogDeterminantCost(
        P.cast<symbolic::Expression>());
    const auto result_lagrangian =
        solvers::Solve(*prog_lagrangian, std::nullopt, solver_options);
    if (!result_lagrangian.is_success()) {
      throw std::runtime_error(
          fmt::format("Find Lagrangian fails in iter {}", iter_count));
    }
    if (bilinear_alternation_option.verbose) {
      std::cout << fmt::format("Iter: {}, max(log(det(P)))={}\n", iter_count,
                               -result_lagrangian.get_optimal_cost());
    }
    // TODO(hongkai.dai): backoff the lagrangian step result.
    const Eigen::VectorXd lagrangian_gram_var_vals =
        result_lagrangian.GetSolution(lagrangian_gram_vars);
    const auto P_sol = result_lagrangian.GetSolution(P);
    const auto q_sol = result_lagrangian.GetSolution(q);
    *P_final = P_sol;
    *q_final = q_sol;
    // Update the cost.
    const double log_det_P = std::log(P_sol.determinant());
    cost_improvement = log_det_P - previous_cost;
    previous_cost = log_det_P;

    // Now solve the polytope problem (fix Lagrangian).
    auto prog_polytope = ConstructPolytopeProgram(
        alternation_tuples, C_var, d_var, d_minus_Ct, lagrangian_gram_var_vals,
        verified_gram_vars, separating_plane_vars, t_minus_t_lower,
        t_upper_minus_t, P_sol, q_sol, verification_option, &margin);
    auto prog_polytope_cost = prog_polytope->AddLinearCost(
        -Eigen::VectorXd::Ones(margin.rows()), 0., margin);
    auto result_polytope =
        solvers::Solve(*prog_polytope, std::nullopt, solver_options);
    if (!result_polytope.is_success()) {
      throw std::runtime_error(fmt::format(
          "Failed to find the polytope at iteration {}", iter_count));
    }

    if (bilinear_alternation_option.verbose) {
      std::cout << fmt::format("Iter: {}, polytope step cost {}\n", iter_count,
                               result_polytope.get_optimal_cost());
    }
    if (bilinear_alternation_option.backoff_scale > 0) {
      const auto margin_sum_max = -result_polytope.get_optimal_cost();
      prog_polytope->RemoveCost(prog_polytope_cost);
      // Now add the constraint margin.sum() >= (1-backoff_scale) *
      // margin_sum_max.
      prog_polytope->AddLinearConstraint(
          Eigen::VectorXd::Ones(margin.rows()),
          (1 - bilinear_alternation_option.backoff_scale) * margin_sum_max,
          kInf, margin);
      result_polytope =
          solvers::Solve(*prog_polytope, std::nullopt, solver_options);
      if (!result_polytope.is_success()) {
        throw std::runtime_error(fmt::format(
            "Failed to backoff polytope program in iteration {}", iter_count));
      }
    }
    C_val = result_polytope.GetSolution(C_var);
    d_val = result_polytope.GetSolution(d_var);
    *C_final = C_val;
    *d_final = d_val;
    iter_count += 1;
  }
}

void CspaceFreeRegion::CspacePolytopeBinarySearch(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d_init,
    const BinarySearchOption& binary_search_option,
    const solvers::SolverOptions& solver_options,
    Eigen::VectorXd* d_final) const {
  const int C_rows = C.rows();
  DRAKE_DEMAND(d_init.rows() == C_rows);
  DRAKE_DEMAND(C.cols() == rational_forward_kinematics_.t().rows());
  std::vector<CspaceFreeRegion::CspacePolytopeTuple> alternation_tuples;
  VectorX<symbolic::Polynomial> d_minus_Ct;
  Eigen::VectorXd t_lower, t_upper;
  VectorX<symbolic::Polynomial> t_minus_t_lower, t_upper_minus_t;
  MatrixX<symbolic::Variable> C_var;
  VectorX<symbolic::Variable> d_var, lagrangian_gram_vars, verified_gram_vars,
      separating_plane_vars;
  GenerateTuplesForBilinearAlternation(
      q_star, filtered_collision_pairs, C_rows, &alternation_tuples,
      &d_minus_Ct, &t_lower, &t_upper, &t_minus_t_lower, &t_upper_minus_t,
      &C_var, &d_var, &lagrangian_gram_vars, &verified_gram_vars,
      &separating_plane_vars);

  VerificationOption verification_option{};
  // Checks if C*t<=d, t_lower<=t<=t_upper is collision free.
  auto is_polytope_collision_free =
      [this, &alternation_tuples, &C, &lagrangian_gram_vars,
       &verified_gram_vars, &separating_plane_vars, &t_lower, &t_upper,
       &verification_option, &solver_options](const Eigen::VectorXd& d) {
        auto prog = this->ConstructLagrangianProgram(
            alternation_tuples, C, d, lagrangian_gram_vars, verified_gram_vars,
            separating_plane_vars, t_lower, t_upper, verification_option,
            nullptr, nullptr);
        // Now add the constraint that C*t<=d , t_lower <= t <= t_upper is not
        // empty. We find t_nominal satisfying these constraints.
        const auto t_nominal = prog->NewContinuousVariables(
            rational_forward_kinematics_.t().rows());
        prog->AddLinearConstraint(C, Eigen::VectorXd::Constant(d.rows(), -kInf),
                                  d, t_nominal);
        prog->AddBoundingBoxConstraint(t_lower, t_upper, t_nominal);

        const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
        return result.is_success();
      };
  if (is_polytope_collision_free(d_init +
                                 binary_search_option.epsilon_max *
                                     Eigen::VectorXd::Ones(d_init.rows()))) {
    *d_final = d_init + binary_search_option.epsilon_max *
                            Eigen::VectorXd::Ones(d_init.rows());
    return;
  }
  if (!is_polytope_collision_free(d_init +
                                  binary_search_option.epsilon_min *
                                      Eigen::VectorXd::Ones(d_init.rows()))) {
    throw std::runtime_error(
        fmt::format("binary search: the initial epsilon {} is infeasible",
                    binary_search_option.epsilon_min));
  }
  double eps_max = binary_search_option.epsilon_max;
  double eps_min = binary_search_option.epsilon_min;
  while (eps_max - eps_min > binary_search_option.epsilon_tol) {
    const double eps = (eps_max + eps_min) / 2;
    const Eigen::VectorXd d =
        d_init + eps * Eigen::VectorXd::Ones(d_init.rows());
    const bool is_feasible = is_polytope_collision_free(d);
    if (is_feasible) {
      std::cout << fmt::format("epsilon={} is feasible\n", eps);
      eps_min = eps;
    } else {
      std::cout << fmt::format("epsilon={} is infeasible\n", eps);
      eps_max = eps;
    }
    *d_final = d;
  }
}

std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const ConvexPolytope* polytope, const ConvexPolytope* other_side_polytope,
    const RationalForwardKinematics::Pose<symbolic::Polynomial>&
        X_AB_multilinear,
    const drake::Vector3<symbolic::Expression>& a_A,
    const symbolic::Expression& b, PlaneSide plane_side,
    SeparatingPlaneOrder plane_order) {
  std::vector<LinkVertexOnPlaneSideRational> rational_fun;
  rational_fun.reserve(polytope->p_BV().cols());
  const symbolic::Monomial monomial_one{};
  // a_A and b are not polynomial of sinθ or cosθ.
  Vector3<symbolic::Polynomial> a_A_poly;
  for (int i = 0; i < 3; ++i) {
    a_A_poly(i) = symbolic::Polynomial({{monomial_one, a_A(i)}});
  }
  const symbolic::Polynomial b_poly({{monomial_one, b}});
  for (int i = 0; i < polytope->p_BV().cols(); ++i) {
    // Step 1: Compute vertex position.
    const Vector3<drake::symbolic::Polynomial> p_AVi =
        X_AB_multilinear.p_AB + X_AB_multilinear.R_AB * polytope->p_BV().col(i);

    // Step 2: Compute a_A.dot(p_AVi) + b
    const drake::symbolic::Polynomial point_on_hyperplane_side =
        a_A_poly.dot(p_AVi) + b_poly;

    // Step 3: Convert the multilinear polynomial to rational function.
    rational_fun.emplace_back(
        rational_forward_kinematics
            .ConvertMultilinearPolynomialToRationalFunction(
                plane_side == PlaneSide::kPositive
                    ? point_on_hyperplane_side - 1
                    : -1 - point_on_hyperplane_side),
        polytope, X_AB_multilinear.frame_A_index, other_side_polytope, a_A, b,
        plane_side, plane_order);
  }
  return rational_fun;
}

bool IsGeometryPairCollisionIgnored(
    const SortedPair<ConvexGeometry::Id>& geometry_pair,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  return filtered_collision_pairs.count(geometry_pair) > 0;
}

bool IsGeometryPairCollisionIgnored(
    ConvexGeometry::Id id1, ConvexGeometry::Id id2,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  return IsGeometryPairCollisionIgnored(
      drake::SortedPair<ConvexGeometry::Id>(id1, id2),
      filtered_collision_pairs);
}

void ComputeBoundsOnT(const Eigen::Ref<const Eigen::VectorXd>& q_star,
                      const Eigen::Ref<const Eigen::VectorXd>& q_lower,
                      const Eigen::Ref<const Eigen::VectorXd>& q_upper,
                      Eigen::VectorXd* t_lower, Eigen::VectorXd* t_upper) {
  DRAKE_DEMAND((q_upper.array() >= q_lower.array()).all());
  // Currently I require that q_upper - q_star < pi and q_star - q_lower > -pi.
  DRAKE_DEMAND(((q_upper - q_star).array() < M_PI).all());
  DRAKE_DEMAND(((q_star - q_lower).array() > -M_PI).all());
  *t_lower = ((q_lower - q_star) / 2).array().tan();
  *t_upper = ((q_upper - q_star) / 2).array().tan();
}

template <typename T>
symbolic::Polynomial CalcPolynomialFromGram(
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Eigen::Ref<const MatrixX<T>>& gram) {
  const int Q_rows = monomial_basis.rows();
  DRAKE_DEMAND(gram.rows() == Q_rows && gram.cols() == Q_rows);
  symbolic::Polynomial ret{};
  using std::pow;
  for (int i = 0; i < Q_rows; ++i) {
    ret.AddProduct(gram(i, i), pow(monomial_basis(i), 2));
    for (int j = i + 1; j < Q_rows; ++j) {
      ret.AddProduct(gram(i, j) + gram(j, i),
                     monomial_basis(i) * monomial_basis(j));
    }
  }
  return ret;
}

symbolic::Polynomial CalcPolynomialFromGram(
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& gram,
    const solvers::MathematicalProgramResult& result) {
  const int Q_rows = monomial_basis.rows();
  DRAKE_DEMAND(gram.rows() == Q_rows && gram.cols() == Q_rows);
  symbolic::Polynomial ret{};
  using std::pow;
  for (int i = 0; i < Q_rows; ++i) {
    ret.AddProduct(result.GetSolution(gram(i, i)), pow(monomial_basis(i), 2));
    for (int j = i + 1; j < Q_rows; ++j) {
      ret.AddProduct(
          result.GetSolution(gram(i, j)) + result.GetSolution(gram(j, i)),
          monomial_basis(i) * monomial_basis(j));
    }
  }
  return ret;
}

template <typename T>
symbolic::Polynomial CalcPolynomialFromGramLower(
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Eigen::Ref<const VectorX<T>>& gram_lower) {
  // I want to avoid dynamically allocating memory for the gram matrix.
  symbolic::Polynomial ret{};
  const int gram_rows = monomial_basis.rows();
  int gram_count = 0;
  using std::pow;
  for (int j = 0; j < gram_rows; ++j) {
    ret.AddProduct(gram_lower(gram_count++), pow(monomial_basis(j), 2));
    for (int i = j + 1; i < gram_rows; ++i) {
      ret.AddProduct(2 * gram_lower(gram_count++),
                     monomial_basis(i) * monomial_basis(j));
    }
  }
  return ret;
}

symbolic::Polynomial CalcPolynomialFromGramLower(
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& gram_lower,
    const solvers::MathematicalProgramResult& result) {
  const int Q_rows = monomial_basis.rows();
  DRAKE_DEMAND(gram_lower.rows() == Q_rows * (Q_rows + 1) / 2);
  symbolic::Polynomial ret{};
  using std::pow;
  int count = 0;
  for (int j = 0; j < Q_rows; ++j) {
    ret.AddProduct(result.GetSolution(gram_lower(count++)),
                   pow(monomial_basis(j), 2));
    for (int i = j + 1; i < Q_rows; ++i) {
      ret.AddProduct(2 * result.GetSolution(gram_lower(count++)),
                     monomial_basis(i) * monomial_basis(j));
    }
  }
  return ret;
}

template <typename T>
void SymmetricMatrixFromLower(int mat_rows,
                              const Eigen::Ref<const VectorX<T>>& lower,
                              MatrixX<T>* mat) {
  DRAKE_DEMAND(lower.rows() == mat_rows * (mat_rows + 1) / 2);
  mat->resize(mat_rows, mat_rows);
  int count = 0;
  for (int j = 0; j < mat_rows; ++j) {
    (*mat)(j, j) = lower(count++);
    for (int i = j + 1; i < mat_rows; ++i) {
      (*mat)(i, j) = lower(count++);
      (*mat)(j, i) = (*mat)(i, j);
    }
  }
}

void AddInscribedEllipsoid(
    solvers::MathematicalProgram* prog,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& P,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& q,
    bool constrain_P_psd) {
  const int t_size = t_lower.rows();
  DRAKE_DEMAND(C.cols() == t_size && C.rows() == d.rows());
  DRAKE_DEMAND(t_upper.rows() == t_size && P.rows() == t_size &&
               P.cols() == t_size && q.rows() == t_size);
  DRAKE_DEMAND((t_upper.array() >= t_lower.array()).all());
  if (constrain_P_psd) {
    prog->AddPositiveSemidefiniteConstraint(P);
  }
  // Add constraint |cᵢᵀP|₂ ≤ dᵢ−cᵢᵀq
  VectorX<symbolic::Expression> lorentz_cone1(t_size + 1);
  for (int i = 0; i < C.rows(); ++i) {
    lorentz_cone1(0) = d(i) - C.row(i).dot(q);
    lorentz_cone1.tail(t_size) = C.row(i) * P;
    prog->AddLorentzConeConstraint(lorentz_cone1);
  }
  // Add constraint |P.row(i)|₂ + qᵢ ≤ t_upper(i)
  // Namely [t_upper(i) - q(i), P.row(i)]=lorentz_A2 * [q(i);P.row(i)] +
  // lorentz_b2 is in the Lorentz cone.
  Eigen::MatrixXd lorentz_A2 =
      Eigen::MatrixXd::Identity(1 + t_size, 1 + t_size);
  lorentz_A2(0, 0) = -1;
  Eigen::VectorXd lorentz_b2 = Eigen::VectorXd::Zero(1 + t_size);
  VectorX<symbolic::Variable> lorentz_var2(t_size + 1);
  for (int i = 0; i < t_size; ++i) {
    lorentz_b2(0) = t_upper(i);
    lorentz_var2(0) = q(i);
    lorentz_var2.tail(t_size) = P.row(i);
    prog->AddLorentzConeConstraint(lorentz_A2, lorentz_b2, lorentz_var2);
  }
  // Add constraint −|P.row(i)|₂ + qᵢ ≥ t_lower(i)
  // Namely [q(i)-t_lower(i), P.row(i)]=lorentz_A2 * [q(i);P.row(i)] +
  // lorentz_b2 is in the Lorentz cone.
  lorentz_A2 = Eigen::MatrixXd::Identity(1 + t_size, 1 + t_size);
  lorentz_b2 = Eigen::VectorXd::Zero(1 + t_size);
  for (int i = 0; i < t_size; ++i) {
    lorentz_b2(0) = -t_lower(i);
    lorentz_var2(0) = q(i);
    lorentz_var2.tail(t_size) = P.row(i);
    prog->AddLorentzConeConstraint(lorentz_A2, lorentz_b2, lorentz_var2);
  }
}

void AddOuterPolytope(
    solvers::MathematicalProgram* prog,
    const Eigen::Ref<const Eigen::MatrixXd>& P,
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& C,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& d,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& margin) {
  DRAKE_DEMAND(P.rows() == P.cols());
  // Add the constraint |cᵢᵀP|₂ ≤ dᵢ − cᵢᵀq − δᵢ as a Lorentz cone constraint,
  // namely [dᵢ − cᵢᵀq − δᵢ, cᵢᵀP] is in the Lorentz cone.
  // [dᵢ − cᵢᵀq − δᵢ, cᵢᵀP] = A_lorentz1 * [cᵢᵀ, dᵢ, δᵢ] + b_lorentz1
  Eigen::MatrixXd A_lorentz1(P.rows() + 1, 2 + C.cols());
  Eigen::VectorXd b_lorentz1(P.rows() + 1);
  VectorX<symbolic::Variable> lorentz1_vars(2 + C.cols());
  for (int i = 0; i < C.rows(); ++i) {
    A_lorentz1.setZero();
    A_lorentz1(0, C.cols()) = 1;
    A_lorentz1(0, C.cols() + 1) = -1;
    A_lorentz1.block(0, 0, 1, C.cols()) = -q.transpose();
    A_lorentz1.block(1, 0, P.rows(), P.cols()) = P;
    b_lorentz1.setZero();
    lorentz1_vars << C.row(i).transpose(), d(i), margin(i);
    prog->AddLorentzConeConstraint(A_lorentz1, b_lorentz1, lorentz1_vars);
  }
  // Add the constraint |cᵢᵀ|₂ ≤ 1 as a Lorentz cone constraint that [1, cᵢᵀ] is
  // in the Lorentz cone.
  // [1, cᵢᵀ] = A_lorentz2 * cᵢᵀ + b_lorentz2
  Eigen::MatrixXd A_lorentz2 = Eigen::MatrixXd::Zero(1 + C.cols(), C.cols());
  A_lorentz2.bottomRows(C.cols()) =
      Eigen::MatrixXd::Identity(C.cols(), C.cols());
  Eigen::VectorXd b_lorentz2 = Eigen::VectorXd::Zero(1 + C.cols());
  b_lorentz2(0) = 1;
  for (int i = 0; i < C.rows(); ++i) {
    prog->AddLorentzConeConstraint(A_lorentz2, b_lorentz2, C.row(i));
  }
}

std::map<BodyIndex, std::vector<ConvexPolytope>> GetConvexPolytopes(
    const systems::Diagram<double>& diagram,
    const MultibodyPlant<double>* plant,
    const geometry::SceneGraph<double>* scene_graph) {
  std::map<BodyIndex, std::vector<ConvexPolytope>> ret;
  // First generate the query object.
  auto diagram_context = diagram.CreateDefaultContext();
  diagram.Publish(*diagram_context);
  const auto query_object =
      scene_graph->get_query_output_port().Eval<geometry::QueryObject<double>>(
          scene_graph->GetMyContextFromRoot(*diagram_context));
  // Loop through each geometry in the SceneGraph.
  const auto& inspector = scene_graph->model_inspector();

  for (multibody::BodyIndex body_index{0}; body_index < plant->num_bodies();
       ++body_index) {
    const std::optional<geometry::FrameId> frame_id =
        plant->GetBodyFrameIdIfExists(body_index);
    if (frame_id.has_value()) {
      const auto geometry_ids =
          inspector.GetGeometries(frame_id.value(), geometry::Role::kProximity);
      for (const auto& geometry_id : geometry_ids) {
        const geometry::optimization::VPolytope v_polytope(
            query_object, geometry_id, frame_id.value());
        const ConvexPolytope convex_polytope(body_index, geometry_id,
                                             v_polytope.vertices());
        auto it = ret.find(body_index);
        if (it == ret.end()) {
          std::vector<ConvexPolytope> body_polytopes;
          body_polytopes.push_back(convex_polytope);
          ret.emplace_hint(it, body_index, body_polytopes);
        } else {
          it->second.push_back(convex_polytope);
        }
      }
    }
  }
  return ret;
}

// Explicit instantiation.
template symbolic::Polynomial CalcPolynomialFromGram<double>(
    const VectorX<symbolic::Monomial>&,
    const Eigen::Ref<const MatrixX<double>>&);
template symbolic::Polynomial CalcPolynomialFromGram<symbolic::Variable>(
    const VectorX<symbolic::Monomial>&,
    const Eigen::Ref<const MatrixX<symbolic::Variable>>&);

template symbolic::Polynomial CalcPolynomialFromGramLower<double>(
    const VectorX<symbolic::Monomial>&,
    const Eigen::Ref<const VectorX<double>>&);
template symbolic::Polynomial CalcPolynomialFromGramLower<symbolic::Variable>(
    const VectorX<symbolic::Monomial>&,
    const Eigen::Ref<const VectorX<symbolic::Variable>>&);

template void SymmetricMatrixFromLower<double>(
    int mat_rows, const Eigen::Ref<const Eigen::VectorXd>&, Eigen::MatrixXd*);
template void SymmetricMatrixFromLower<symbolic::Variable>(
    int mat_rows, const Eigen::Ref<const VectorX<symbolic::Variable>>&,
    MatrixX<symbolic::Variable>*);

}  // namespace multibody
}  // namespace drake
