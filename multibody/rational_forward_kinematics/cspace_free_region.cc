#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"

#include <algorithm>
#include <limits>
#include <optional>

#include <fmt/format.h>
#include <libqhullcpp/Qhull.h>

#include "drake/geometry/optimization/vpolytope.h"
#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/multibody/rational_forward_kinematics/redundant_inequality_pruning.h"
#include "drake/solvers/mosek_solver.h"
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
// verify (in the numerator of 1-aᵀx-b or aᵀx+b-1) only contains the
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
    if (t_halfchain.rows() > 0) {
      *monomial_basis_halfchain = GenerateMonomialBasisWithOrderUpToOne(
          drake::symbolic::Variables(t_halfchain));
    } else {
      *monomial_basis_halfchain =
          Vector1<symbolic::Monomial>(symbolic::Monomial());
    }
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
        // I need to check if the kinematics chain betwen link 1 and link 2 has
        // length 0.
        std::optional<bool> chain_has_length_zero;
        for (const auto& polytope1 : polytopes1) {
          for (const auto& polytope2 : polytopes2) {
            if (!model_inspector.CollisionFiltered(polytope1.get_id(),
                                                   polytope2.get_id())) {
              if (!chain_has_length_zero.has_value()) {
                chain_has_length_zero =
                    rational_forward_kinematics_.FindTOnPath(link1, link2)
                        .rows() == 0;
                if (chain_has_length_zero.value()) {
                  throw std::runtime_error(fmt::format(
                      "No joint on the kinematic chain from link {} to {}",
                      plant->get_body(link1).name(),
                      plant->get_body(link2).name()));
                }
              }
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
        prog->NewSosPolynomial(monomial_basis,
                               verification_option.lagrangian_type)
            .first;
    *verified_polynomial -= (*lagrangian_polytope)(i)*d_minus_Ct(i);
  }
  for (int i = 0; i < t_minus_t_lower.rows(); ++i) {
    if (t_lower_needs_lagrangian[i]) {
      (*lagrangian_lower)(i) =
          prog->NewSosPolynomial(monomial_basis,
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
          prog->NewSosPolynomial(monomial_basis,
                                 verification_option.lagrangian_type)
              .first;
      *verified_polynomial -= (*lagrangian_upper)(i)*t_upper_minus_t(i);
    } else {
      (*lagrangian_upper)(i) = symbolic::Polynomial();
    }
  }

  const symbolic::Polynomial verified_polynomial_expected =
      prog->NewSosPolynomial(monomial_basis,
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
    const VerificationOption& option, std::optional<double> redundant_tighten,
    MatrixX<symbolic::Variable>* P, VectorX<symbolic::Variable>* q) const {
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
  // find the redundant inequalities.
  std::unordered_set<int> C_redundant_indices, t_lower_redundant_indices,
      t_upper_redundant_indices;
  if (redundant_tighten.has_value()) {
    FindRedundantInequalities(C, d, t_lower, t_upper, redundant_tighten.value(),
                              &C_redundant_indices, &t_lower_redundant_indices,
                              &t_upper_redundant_indices);
  }
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

    // This lambda does these things.
    // If redundant = False, namely this constraint is not redundant, then
    // 1. Compute the Gram matrix.
    // 2. Constraint the Gram matrix to be PSD.
    // 3. subtract lagrangian(t) * constraint_polynomial from
    // verified_polynomial.
    // If redundant = True, then
    // 1. Set the Gram variables to be 0.
    auto constrain_lagrangian =
        [&gram_mat, &verified_polynomial, &lagrangian_gram_vars, &prog,
         gram_lower_size](int lagrangian_gram_lower_start,
                          const VectorX<symbolic::Monomial>& monomial_basis,
                          const symbolic::Polynomial& constraint_polynomial,
                          bool redundant) {
          if (redundant) {
            prog->AddBoundingBoxConstraint(
                0, 0,
                lagrangian_gram_vars.segment(lagrangian_gram_lower_start,
                                             gram_lower_size));

          } else {
            SymmetricMatrixFromLower<symbolic::Variable>(
                gram_mat.rows(),
                lagrangian_gram_vars.segment(lagrangian_gram_lower_start,
                                             gram_lower_size),
                &gram_mat);
            prog->AddPositiveSemidefiniteConstraint(gram_mat);
            verified_polynomial -= CalcPolynomialFromGram<symbolic::Variable>(
                                       monomial_basis, gram_mat) *
                                   constraint_polynomial;
          }
        };
    // Handle lagrangian l_polytope(t).
    for (int i = 0; i < C.rows(); ++i) {
      constrain_lagrangian(tuple.polytope_lagrangian_gram_lower_start[i],
                           tuple.monomial_basis, d_minus_Ct(i),
                           C_redundant_indices.count(i) > 0);
    }
    // Handle lagrangian l_lower(t) and l_upper(t).
    for (int i = 0; i < t.rows(); ++i) {
      constrain_lagrangian(tuple.t_lower_lagrangian_gram_lower_start[i],
                           tuple.monomial_basis, t_minus_t_lower(i),
                           t_lower_redundant_indices.count(i) > 0);
      constrain_lagrangian(tuple.t_upper_lagrangian_gram_lower_start[i],
                           tuple.monomial_basis, t_upper_minus_t(i),
                           t_upper_redundant_indices.count(i) > 0);
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
    const VerificationOption& option) const {
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
      if ((lagrangian_gram_var_vals
               .segment(lagrangian_var_start, gram_lower_size)
               .array() != 0)
              .any()) {
        verified_polynomial -=
            CalcPolynomialFromGramLower<double>(
                monomial_basis, lagrangian_gram_var_vals.segment(
                                    lagrangian_var_start, gram_lower_size)) *
            constraint;
      }
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
  return prog;
}

namespace {
// In bilinear alternation, each conic program (with a linear cost) finds an
// optimal solution at the boundary of the cone. Namely the solution is very
// close to being infeasible, which is a bad starting point for the next conic
// program in alternation, as the next conic program can be regarded infeasible
// due to numerical problems. To resolve this issue, we solve a feasibility
// problem (with no cost) that finds a strictly feasible solution at the
// interior of the cone.
// @param cost_val is the optimal cost of the optimization program (which
// minimizes `linear_cost`).
solvers::MathematicalProgramResult BackoffProgram(
    solvers::MathematicalProgram* prog, double cost_val, double backoff_scale,
    const solvers::SolverOptions& solver_options) {
  // The program has no quadratic cost.
  DRAKE_DEMAND(prog->quadratic_costs().empty());
  // For the moment, assume the program has only one linear cost.
  if (prog->linear_costs().size() != 1) {
    drake::log()->error(
        "Currently only support program with exactly one linear cost");
  }
  const auto linear_cost = prog->linear_costs()[0];
  prog->RemoveCost(linear_cost);
  if (backoff_scale <= 0) {
    throw std::invalid_argument(
        fmt::format("backoff_scale = {}, only back off when backoff_scale > 0",
                    backoff_scale));
  }
  // If cost_val > 0, then we impose the constraint cost(vars) <= (1 +
  // backoff_scale) * cost_val; otherwise we impose the constraint cost(vars) <=
  // (1 - backoff_scale) * cost_val;
  const double upper_bound =
      cost_val > 0
          ? (1 + backoff_scale) * cost_val - linear_cost.evaluator()->b()
          : (1 - backoff_scale) * cost_val - linear_cost.evaluator()->b();
  prog->AddLinearConstraint(linear_cost.evaluator()->a(), -kInf, upper_bound,
                            linear_cost.variables());
  const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
  if (!result.is_success()) {
    throw std::runtime_error("Backoff fails.");
  }
  return result;
}
}  // namespace

void CspaceFreeRegion::CspacePolytopeBilinearAlternation(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs,
    const Eigen::Ref<const Eigen::MatrixXd>& C_init,
    const Eigen::Ref<const Eigen::VectorXd>& d_init,
    const CspaceFreeRegion::BilinearAlternationOption&
        bilinear_alternation_option,
    const solvers::SolverOptions& solver_options,
    const std::optional<Eigen::MatrixXd>& q_inner_pts,
    const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
        inner_polytope,
    Eigen::MatrixXd* C_final, Eigen::VectorXd* d_final,
    Eigen::MatrixXd* P_final, Eigen::VectorXd* q_final) const {
  if (bilinear_alternation_option.lagrangian_backoff_scale < 0) {
    throw std::invalid_argument(
        fmt::format("lagrangian_backoff_scale={}, should be non-negative",
                    bilinear_alternation_option.lagrangian_backoff_scale));
  }
  if (bilinear_alternation_option.polytope_backoff_scale < 0 ||
      bilinear_alternation_option.polytope_backoff_scale > 1) {
    throw std::invalid_argument(
        fmt::format("polytope_backoff_scale={}, should be within [0, 1]",
                    bilinear_alternation_option.polytope_backoff_scale));
  }
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
  if (bilinear_alternation_option.compute_polytope_volume) {
    drake::log()->info(
        fmt::format("Polytope volume {}",
                    CalcCspacePolytopeVolume(C_val, d_val, t_lower, t_upper)));
  }

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
        verification_option, bilinear_alternation_option.redundant_tighten, &P,
        &q);
    auto [log_det_cost, log_det_t, log_det_Z] =
        prog_lagrangian->AddMaximizeLogDeterminantCost(
            P.cast<symbolic::Expression>());
    auto result_lagrangian =
        solvers::Solve(*prog_lagrangian, std::nullopt, solver_options);
    if (!result_lagrangian.is_success()) {
      throw std::runtime_error(
          fmt::format("Find Lagrangian fails in iter {}", iter_count));
    }
    if (bilinear_alternation_option.verbose) {
      drake::log()->info(fmt::format(
          "Iter: {}, max(log(det(P)))={}, solver_time {}", iter_count,
          -result_lagrangian.get_optimal_cost(),
          result_lagrangian.get_solver_details<solvers::MosekSolver>()
              .optimizer_time));
    }
    if (bilinear_alternation_option.lagrangian_backoff_scale > 0) {
      result_lagrangian = BackoffProgram(
          prog_lagrangian.get(), result_lagrangian.get_optimal_cost(),
          bilinear_alternation_option.lagrangian_backoff_scale, solver_options);
      if (bilinear_alternation_option.verbose) {
        drake::log()->info(fmt::format(
            "backoff with log(det(P)) {}, solver time {}",
            -result_lagrangian.EvalBinding(log_det_cost)(0),
            result_lagrangian.get_solver_details<solvers::MosekSolver>()
                .optimizer_time));
      }
    }
    const Eigen::VectorXd lagrangian_gram_var_vals =
        result_lagrangian.GetSolution(lagrangian_gram_vars);
    const auto P_sol = result_lagrangian.GetSolution(P);
    const auto q_sol = result_lagrangian.GetSolution(q);
    *P_final = P_sol;
    *q_final = q_sol;
    // Update the cost.
    const double log_det_P = std::log(P_sol.determinant());
    cost_improvement = log_det_P - previous_cost;
    drake::log()->info(fmt::format("cost improvement {}", cost_improvement));
    previous_cost = log_det_P;

    // Now solve the polytope problem (fix Lagrangian).
    auto prog_polytope = ConstructPolytopeProgram(
        alternation_tuples, C_var, d_var, d_minus_Ct, lagrangian_gram_var_vals,
        verified_gram_vars, separating_plane_vars, t_minus_t_lower,
        t_upper_minus_t, verification_option);
    // Add the constraint that the polytope contains the ellipsoid
    margin = prog_polytope->NewContinuousVariables(C_var.rows(), "margin");
    AddOuterPolytope(prog_polytope.get(), P_sol, q_sol, C_var, d_var, margin);
    prog_polytope->AddBoundingBoxConstraint(0, kInf, margin);
    // Add the constraint that the polytope contains the t_inner_pts.
    if (q_inner_pts.has_value()) {
      Eigen::MatrixXd t_inner_pts(q_inner_pts->rows(), q_inner_pts->cols());
      for (int i = 0; i < q_inner_pts->cols(); ++i) {
        t_inner_pts.col(i) = rational_forward_kinematics_.ComputeTValue(
            q_inner_pts->col(i), q_star);
      }
      for (int i = 0; i < t_inner_pts.cols(); ++i) {
        DRAKE_DEMAND((t_inner_pts.col(i).array() <= t_upper.array()).all());
        DRAKE_DEMAND((t_inner_pts.col(i).array() >= t_lower.array()).all());
      }
      AddCspacePolytopeContainment(prog_polytope.get(), C_var, d_var,
                                   t_inner_pts);
    }
    // Add the constraint that the polytope contains the inner polytope.
    if (inner_polytope.has_value()) {
      AddCspacePolytopeContainment(prog_polytope.get(), C_var, d_var,
                                   inner_polytope->first,
                                   inner_polytope->second, t_lower, t_upper);
    }

    // Maximize ∏ᵢ(margin(i) + epsilon), where epsilon is a
    // small positive number to make sure this geometric mean is always
    // positive, even when some of margin is 0.
    prog_polytope->AddMaximizeGeometricMeanCost(
        Eigen::MatrixXd::Identity(margin.rows(), margin.rows()),
        Eigen::VectorXd::Constant(margin.rows(), 1E-5), margin);
    // TODO(hongkai.dai): remove this line when the Drake PR 16373 is merged.
    auto prog_polytope_cost = prog_polytope->linear_costs()[0];
    auto result_polytope =
        solvers::Solve(*prog_polytope, std::nullopt, solver_options);
    if (!result_polytope.is_success()) {
      throw std::runtime_error(fmt::format(
          "Failed to find the polytope at iteration {}", iter_count));
    }

    if (bilinear_alternation_option.verbose) {
      drake::log()->info(
          fmt::format("Iter: {}, polytope step cost {}, solver time {}",
                      iter_count, result_polytope.get_optimal_cost(),
                      result_polytope.get_solver_details<solvers::MosekSolver>()
                          .optimizer_time));
    }
    if (bilinear_alternation_option.polytope_backoff_scale > 0) {
      result_polytope = BackoffProgram(
          prog_polytope.get(), result_polytope.get_optimal_cost(),
          bilinear_alternation_option.polytope_backoff_scale, solver_options);
      if (bilinear_alternation_option.verbose) {
        drake::log()->info(fmt::format(
            "back off, cost: {}, solver time {}",
            result_polytope.EvalBinding(prog_polytope_cost)(0),
            result_polytope.get_solver_details<solvers::MosekSolver>()
                .optimizer_time));
      }
    }
    C_val = result_polytope.GetSolution(C_var);
    d_val = result_polytope.GetSolution(d_var);
    if (bilinear_alternation_option.compute_polytope_volume) {
      drake::log()->info(fmt::format(
          "Polytope volume {}",
          CalcCspacePolytopeVolume(C_val, d_val, t_lower, t_upper)));
    }
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
    const std::optional<Eigen::MatrixXd>& q_inner_pts,
    const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
        inner_polytope,
    Eigen::VectorXd* d_final) const {
  // The polytope region is C * t <= d_without_epsilon + epsilon. We might
  // change d_without_epsilon during the binary search process.
  Eigen::VectorXd d_without_epsilon = d_init;
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
  std::optional<Eigen::MatrixXd> t_inner_pts;
  if (q_inner_pts.has_value()) {
    t_inner_pts->resize(q_inner_pts->rows(), q_inner_pts->cols());
    for (int i = 0; i < q_inner_pts->cols(); ++i) {
      t_inner_pts->col(i) = rational_forward_kinematics_.ComputeTValue(
          q_inner_pts->col(i), q_star);
    }
  }
  DRAKE_DEMAND(binary_search_option.epsilon_min >=
               FindEpsilonLower(C, d_init, t_lower, t_upper, t_inner_pts,
                                inner_polytope));

  VerificationOption verification_option{};
  // Checks if C*t<=d, t_lower<=t<=t_upper is collision free.
  // Update d_final = d if the program C*t<=d, t_lower <= t <= t_upper is
  // collision free.
  auto is_polytope_collision_free = [this, &alternation_tuples, &C,
                                     &lagrangian_gram_vars, &verified_gram_vars,
                                     &separating_plane_vars, &t_lower, &t_upper,
                                     &verification_option, &solver_options,
                                     &C_var, &d_var, &d_minus_Ct,
                                     &t_minus_t_lower, &t_upper_minus_t,
                                     &t_inner_pts, &inner_polytope](
                                        const Eigen::VectorXd& d, bool search_d, bool compute_polytope_volume,
                                        Eigen::VectorXd* d_sol) {
    const double redundant_tighten = 0.;
    auto prog = this->ConstructLagrangianProgram(
        alternation_tuples, C, d, lagrangian_gram_vars, verified_gram_vars,
        separating_plane_vars, t_lower, t_upper, verification_option,
        redundant_tighten, nullptr, nullptr);
    const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
    if (result.is_success()) {
      *d_sol = d;
      if (search_d) {
        // Now fix the Lagrangian and C, and search for d.
        const auto lagrangian_gram_var_vals =
            result.GetSolution(lagrangian_gram_vars);
        auto prog_polytope = this->ConstructPolytopeProgram(
            alternation_tuples, C_var, d_var, d_minus_Ct,
            lagrangian_gram_var_vals, verified_gram_vars, separating_plane_vars,
            t_minus_t_lower, t_upper_minus_t, verification_option);
        // Calling AddBoundingBoxConstraint(C, C, C_var) for matrix C and
        // C_var might have problem, see Drake issue #16421
        prog_polytope->AddBoundingBoxConstraint(
            Eigen::Map<const Eigen::VectorXd>(C.data(), C.rows() * C.cols()),
            Eigen::Map<const Eigen::VectorXd>(C.data(), C.rows() * C.cols()),
            Eigen::Map<const VectorX<symbolic::Variable>>(C_var.data(),
                                                          C.rows() * C.cols()));
        // d_var >= d
        prog_polytope->AddBoundingBoxConstraint(
            d, Eigen::VectorXd::Constant(d.rows(), kInf), d_var);
        if (t_inner_pts.has_value()) {
          AddCspacePolytopeContainment(prog_polytope.get(), C_var, d_var,
                                       t_inner_pts.value());
        }
        if (inner_polytope.has_value()) {
          AddCspacePolytopeContainment(
              prog_polytope.get(), C_var, d_var, inner_polytope->first,
              inner_polytope->second, t_lower, t_upper);
        }
        // maximize d_var.
        prog_polytope->AddLinearCost(-Eigen::VectorXd::Ones(d_var.rows()), 0,
                                     d_var);
        const auto result_polytope =
            solvers::Solve(*prog_polytope, std::nullopt, solver_options);
        if (result_polytope.is_success()) {
          *d_sol = result_polytope.GetSolution(d_var);
        }
      }
    }
    drake::log()->info(fmt::format(
        "Solver time {}",
        result.get_solver_details<solvers::MosekSolver>().optimizer_time));
    if (compute_polytope_volume) {
drake::log()->info(
              fmt::format("C-space polytope volume {}",
                          CalcCspacePolytopeVolume(C, d, t_lower, t_upper)));
    }
    return result.is_success();
  };
  if (is_polytope_collision_free(
          d_without_epsilon +
              binary_search_option.epsilon_max *
                  Eigen::VectorXd::Ones(d_without_epsilon.rows()),
          binary_search_option.search_d, binary_search_option.compute_polytope_volume, d_final)) {
    return;
  }
  if (!is_polytope_collision_free(
          d_without_epsilon +
              binary_search_option.epsilon_min *
                  Eigen::VectorXd::Ones(d_without_epsilon.rows()),
          false /* don't search for d */, binary_search_option.compute_polytope_volume, d_final)) {
    throw std::runtime_error(
        fmt::format("binary search: the initial epsilon {} is infeasible",
                    binary_search_option.epsilon_min));
  }
  double eps_max = binary_search_option.epsilon_max;
  double eps_min = binary_search_option.epsilon_min;
  int iter_count = 0;
  while (iter_count < binary_search_option.max_iters) {
    const double eps = (eps_max + eps_min) / 2;
    const Eigen::VectorXd d =
        d_without_epsilon +
        eps * Eigen::VectorXd::Ones(d_without_epsilon.rows());
    const bool is_feasible =
        is_polytope_collision_free(d, binary_search_option.search_d, binary_search_option.compute_polytope_volume, d_final);
    if (is_feasible) {
      drake::log()->info(fmt::format("epsilon={} is feasible", eps));
      // Now we need to reset d_without_epsilon. The invariance we want is that
      // C*t<= d_without_epsilon + eps_min * 𝟏 is collision free, while C*t <=
      // d_without_epsilon + eps_max * 𝟏 is not feasible with SOS. We know that
      // d_final + 0 * 𝟏 is collision free. Also (d_without_epsilon + eps_max *
      // 𝟏) is not. So we set d_without_epsilon to d_final, and update eps_min
      // and eps_max accordingly.
      eps_max = (d_without_epsilon +
                 eps_max * Eigen::VectorXd::Ones(d.rows(), 1) - *d_final)
                    .maxCoeff();
      d_without_epsilon = *d_final;
      eps_min = 0;
      drake::log()->info(
          fmt::format("reset eps_min={}, eps_max={}", eps_min, eps_max));
    } else {
      drake::log()->info(fmt::format("epsilon={} is infeasible", eps));
      eps_max = eps;
    }
    iter_count++;
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

void FindRedundantInequalities(
    const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
    const Eigen::VectorXd& t_lower, const Eigen::VectorXd& t_upper,
    double tighten, std::unordered_set<int>* C_redundant_indices,
    std::unordered_set<int>* t_lower_redundant_indices,
    std::unordered_set<int>* t_upper_redundant_indices) {
  C_redundant_indices->clear();
  t_lower_redundant_indices->clear();
  t_upper_redundant_indices->clear();
  // We aggregate the constraint {C*t<=d, t_lower <= t <= t_upper} as C̅t ≤ d̅
  const int nt = t_lower.rows();
  Eigen::MatrixXd C_bar(C.rows() + 2 * nt, nt);
  Eigen::VectorXd d_bar(d.rows() + 2 * nt);
  C_bar << C, Eigen::MatrixXd::Identity(nt, nt),
      -Eigen::MatrixXd::Identity(nt, nt);
  d_bar << d, t_upper, -t_lower;
  const std::vector<int> redundant_indices =
      FindRedundantInequalitiesInHPolyhedronByIndex(C_bar, d_bar, tighten);
  C_redundant_indices->reserve(redundant_indices.size());
  t_lower_redundant_indices->reserve(redundant_indices.size());
  t_upper_redundant_indices->reserve(redundant_indices.size());
  for (const int index : redundant_indices) {
    if (index < C.rows()) {
      C_redundant_indices->emplace_hint(C_redundant_indices->end(), index);
    } else if (index < C.rows() + nt) {
      t_upper_redundant_indices->emplace_hint(t_upper_redundant_indices->end(),
                                              index - C.rows());
    } else {
      t_lower_redundant_indices->emplace_hint(t_lower_redundant_indices->end(),
                                              index - C.rows() - nt);
    }
  }
}

double FindEpsilonLower(
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    const std::optional<Eigen::MatrixXd>& t_inner_pts,
    const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
        inner_polytope) {
  solvers::MathematicalProgram prog{};
  const int nt = t_lower.rows();
  DRAKE_DEMAND(t_upper.rows() == nt);
  DRAKE_DEMAND(C.cols() == nt);
  const auto t = prog.NewContinuousVariables(nt, "t");
  const auto epsilon = prog.NewContinuousVariables<1>("epsilon");
  // Add the constraint C*t<=d+epsilon and t_lower <= t <= t_upper.
  Eigen::MatrixXd A(C.rows(), nt + 1);
  A.leftCols(nt) = C;
  A.rightCols<1>() = -Eigen::VectorXd::Ones(C.rows());
  prog.AddLinearConstraint(A, Eigen::VectorXd::Constant(C.rows(), -kInf), d,
                           {t, epsilon});
  prog.AddBoundingBoxConstraint(t_lower, t_upper, t);
  if (t_inner_pts.has_value()) {
    // epsilon >= C *t_inner_pts - d
    const double eps_min = ((C * t_inner_pts.value()).colwise() - d).maxCoeff();
    prog.AddBoundingBoxConstraint(eps_min, kInf, epsilon);
  }
  if (inner_polytope.has_value()) {
    // This is not the most efficient way to add the constraint that
    // C*t<=d+epsilon contains the inner_polytope.
    const auto C_var = prog.NewContinuousVariables(C.rows(), C.cols());
    prog.AddBoundingBoxConstraint(
        Eigen::Map<const Eigen::VectorXd>(C.data(), C.rows() * C.cols()),
        Eigen::Map<const Eigen::VectorXd>(C.data(), C.rows() * C.cols()),
        Eigen::Map<const VectorX<symbolic::Variable>>(C_var.data(),
                                                      C.rows() * C.cols()));
    // d_var = d+epsilon
    const auto d_var = prog.NewContinuousVariables(d.rows());
    Eigen::MatrixXd coeff(d.rows(), d.rows() + 1);
    coeff << Eigen::MatrixXd::Identity(d.rows(), d.rows()),
        -Eigen::VectorXd::Ones(d.rows());
    prog.AddLinearEqualityConstraint(coeff, d, {d_var, epsilon});
    AddCspacePolytopeContainment(&prog, C_var, d_var, inner_polytope->first,
                                 inner_polytope->second, t_lower, t_upper);
  }
  // minimize epsilon.
  prog.AddLinearCost(Vector1d(1), 0, epsilon);
  const auto result = solvers::Solve(prog);
  DRAKE_DEMAND(result.is_success());
  return result.get_optimal_cost();
}

void GetCspacePolytope(const Eigen::Ref<const Eigen::MatrixXd>& C,
                       const Eigen::Ref<const Eigen::VectorXd>& d,
                       const Eigen::Ref<const Eigen::VectorXd>& t_lower,
                       const Eigen::Ref<const Eigen::VectorXd>& t_upper,
                       Eigen::MatrixXd* C_bar, Eigen::VectorXd* d_bar) {
  const int nt = t_lower.rows();
  DRAKE_DEMAND(C.cols() == nt);
  DRAKE_DEMAND(C_bar != nullptr);
  DRAKE_DEMAND(d_bar != nullptr);
  C_bar->resize(C.rows() + 2 * nt, nt);
  *C_bar << C, Eigen::MatrixXd::Identity(nt, nt),
      -Eigen::MatrixXd::Identity(nt, nt);
  d_bar->resize(C.rows() + 2 * nt);
  *d_bar << d, t_upper, -t_lower;
}

void AddCspacePolytopeContainment(
    solvers::MathematicalProgram* prog, const MatrixX<symbolic::Variable>& C,
    const VectorX<symbolic::Variable>& d,
    const Eigen::Ref<const Eigen::MatrixXd>& C_inner,
    const Eigen::Ref<const Eigen::VectorXd>& d_inner,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper) {
  const int dim = C.cols();
  DRAKE_DEMAND(C_inner.cols() == dim);
  Eigen::MatrixXd C_bar;
  Eigen::VectorXd d_bar;
  GetCspacePolytope(C_inner, d_inner, t_lower, t_upper, &C_bar, &d_bar);
  // According to duality theory, the polytope C*t<=d contains the polytope
  // C_bar*t <= d_bar if and only if there exists variable λᵢ≥ 0, dᵢ−λᵢᵀd̅≥0,
  // C̅ᵀλᵢ=cᵢ for every row of C*t<=d.
  auto lambda = prog->NewContinuousVariables(C_bar.rows(), C.rows());
  prog->AddBoundingBoxConstraint(0, kInf, lambda);
  // Allocate the memory for the constraint coefficients and variables.
  // coefficients for dᵢ−λᵢᵀd̅
  Eigen::RowVectorXd coeff1(C_bar.rows() + 1);
  VectorX<symbolic::Variable> vars1(C_bar.rows() + 1);
  // Coefficients for C̅ᵀλᵢ=cᵢ
  Eigen::MatrixXd coeff2(C_bar.cols(), C_bar.rows() + C_bar.cols());
  coeff2 << C_bar.transpose(),
      -Eigen::MatrixXd::Identity(C_bar.cols(), C_bar.cols());
  VectorX<symbolic::Variable> vars2(C_bar.rows() + C_bar.cols());
  for (int i = 0; i < C.rows(); ++i) {
    // Add the constraint dᵢ−λᵢᵀd̅≥0
    coeff1(0) = 1;
    coeff1.tail(C_bar.rows()) = -d_bar.transpose();
    vars1(0) = d(i);
    vars1.tail(C_bar.rows()) = lambda.col(i);
    prog->AddLinearConstraint(coeff1, 0, kInf, vars1);
    // Add the constraint C̅ᵀλᵢ=cᵢ
    vars2 << lambda.col(i), C.row(i).transpose();
    prog->AddLinearEqualityConstraint(
        coeff2, Eigen::VectorXd::Zero(C_bar.cols()), vars2);
  }
}

void AddCspacePolytopeContainment(
    solvers::MathematicalProgram* prog, const MatrixX<symbolic::Variable>& C,
    const VectorX<symbolic::Variable>& d,
    const Eigen::Ref<const Eigen::MatrixXd>& inner_pts) {
  DRAKE_DEMAND(C.cols() == inner_pts.rows());
  // Add the constraint that C * inner_pts <= d
  // Note this is a constraint on variable C and d.
  // Namely C.row(i) * inner_pts - d(i) * 𝟏ᵀ <= 0.
  Eigen::MatrixXd coeff(inner_pts.cols(), inner_pts.rows() + 1);
  coeff.leftCols(inner_pts.rows()) = inner_pts.transpose();
  coeff.rightCols<1>() = -Eigen::VectorXd::Ones(inner_pts.cols());
  // Allocate memory.
  VectorX<symbolic::Variable> vars(C.cols() + 1);
  const Eigen::VectorXd lb = Eigen::VectorXd::Constant(inner_pts.cols(), -kInf);
  const Eigen::VectorXd ub = Eigen::VectorXd::Zero(inner_pts.cols());
  for (int i = 0; i < C.rows(); ++i) {
    vars << C.row(i).transpose(), d(i);
    prog->AddLinearConstraint(coeff, lb, ub, vars);
  }
}

double CalcCspacePolytopeVolume(const Eigen::MatrixXd& C,
                                const Eigen::VectorXd& d,
                                const Eigen::VectorXd& t_lower,
                                const Eigen::VectorXd& t_upper) {
  const int nt = t_lower.rows();
  Eigen::MatrixXd C_bar(C.rows() + 2 * nt, nt);
  C_bar << C, Eigen::MatrixXd::Identity(nt, nt),
      -Eigen::MatrixXd::Identity(nt, nt);
  Eigen::VectorXd d_bar(C.rows() + 2 * nt);
  d_bar << d, t_upper, -t_lower;
  const geometry::optimization::HPolyhedron h_poly(C_bar, d_bar);
  const geometry::optimization::VPolytope v_poly(h_poly);
  // TODO(hongkai.dai) call v_poly.CalcVolume() when Drake PR 16409 is merged.
  orgQhull::Qhull qhull;
  qhull.runQhull("", nt, v_poly.vertices().cols(), v_poly.vertices().data(),
                 "");
  if (qhull.qhullStatus() != 0) {
    throw std::runtime_error(
        fmt::format("Qhull terminated with status {} and message:\n{}",
                    qhull.qhullStatus(), qhull.qhullMessage()));
  }
  return qhull.volume();
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
