#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <optional>
#include <thread>

#include <drake_vendor/libqhullcpp/Qhull.h>
#include <fmt/format.h>

#include "drake/geometry/optimization/vpolytope.h"
#include "drake/multibody/rational_forward_kinematics/collision_geometry.h"
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
    const LinkOnPlaneSideRational& rational,
    std::unordered_map<SortedPair<multibody::BodyIndex>,
                       VectorX<drake::symbolic::Monomial>>*
        map_chain_to_monomial_basis,
    VectorX<drake::symbolic::Monomial>* monomial_basis_halfchain) {
  // First check if the monomial basis for this kinematics chain has been
  // computed.
  const SortedPair<multibody::BodyIndex> kinematics_chain(
      rational.link_geometry->body_index(), rational.expressed_body_index);
  const auto it = map_chain_to_monomial_basis->find(kinematics_chain);
  if (it == map_chain_to_monomial_basis->end()) {
    const auto t_halfchain = rational_forward_kinematics.FindTOnPath(
        rational.link_geometry->body_index(), rational.expressed_body_index);
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

// Given a matrix X, add the cost to maximize the geometric mean of X's eigen
// values (provided that X is positive semidefinite).
void AddMaximizeEigenValueGeometricMean(
    solvers::MathematicalProgram* prog,
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& X) {
  // See https://docs.mosek.com/modeling-cookbook/sdo.html#log-determinant for
  // the derivation.
  DRAKE_DEMAND(X.rows() == X.cols());
  const int X_rows = X.rows();
  auto Z_lower = prog->NewContinuousVariables(X_rows * (X_rows + 1) / 2);
  MatrixX<symbolic::Expression> Z(X_rows, X_rows);
  Z.setZero();
  // diag_Z is the diagonal matrix that only contains the diagonal entries of Z.
  MatrixX<symbolic::Expression> diag_Z(X_rows, X_rows);
  diag_Z.setZero();
  int Z_lower_index = 0;
  for (int j = 0; j < X_rows; ++j) {
    for (int i = j; i < X_rows; ++i) {
      Z(i, j) = Z_lower(Z_lower_index++);
    }
    diag_Z(j, j) = Z(j, j);
  }

  MatrixX<symbolic::Expression> psd_mat(2 * X_rows, 2 * X_rows);
  // clang-format off
  psd_mat << X,             Z,
             Z.transpose(), diag_Z;
  // clang-format on
  prog->AddPositiveSemidefiniteConstraint(psd_mat);
  // We know that det(X) >= det(Z) = prod(Z(i, i))
  VectorX<symbolic::Variable> Z_diag_vec(X_rows);
  int Z_lower_count = 0;
  for (int i = 0; i < X_rows; ++i) {
    Z_diag_vec(i) = Z_lower(Z_lower_count);
    Z_lower_count += X_rows - i;
  }
  if (Z_diag_vec.rows() > 1) {
    prog->AddMaximizeGeometricMeanCost(Z_diag_vec);
  } else {
    // enables adding this cost on 1d plants
    prog->AddLinearCost(-Eigen::VectorXd::Ones(1), 0, Z_diag_vec);
  }
}

// Return the smallest d >= n where d = power(2, k);
int SmallestPower2(int n) {
  DRAKE_DEMAND(n >= 1);
  if (n == 1) {
    return 1;
  }
  int ret = 1;
  while (true) {
    ret *= 2;
    if (ret >= n) {
      return ret;
    }
  }
}

// Compute the following problem
// max C.row(i).dot(t)
// s.t t_lower <= t <= t_upper
Eigen::VectorXd ComputeMaxD(const Eigen::MatrixXd& C,
                            const Eigen::VectorXd& t_lower,
                            const Eigen::VectorXd& t_upper) {
  DRAKE_DEMAND(C.cols() == t_lower.rows());
  DRAKE_DEMAND(C.cols() == t_upper.rows());
  DRAKE_DEMAND((t_lower.array() <= t_upper.array()).all());
  Eigen::VectorXd d_max(C.rows());
  for (int i = 0; i < C.rows(); ++i) {
    d_max(i) = 0;
    for (int j = 0; j < C.cols(); ++j) {
      d_max(i) += C(i, j) > 0 ? C(i, j) * t_upper(j) : C(i, j) * t_lower(j);
    }
  }
  return d_max;
}

SeparatingPlane<double> GetSeparatingPlaneSolution(
    const SeparatingPlane<symbolic::Variable>& plane,
    const solvers::MathematicalProgramResult& result) {
  symbolic::Environment env;
  const Eigen::VectorXd decision_variables_val =
      result.GetSolution(plane.decision_variables);
  env.insert(plane.decision_variables, decision_variables_val);
  Vector3<symbolic::Expression> a_sol;
  for (int i = 0; i < 3; ++i) {
    a_sol(i) = plane.a(i).EvaluatePartial(env);
  }
  const symbolic::Expression b_sol = plane.b.EvaluatePartial(env);
  return SeparatingPlane<double>(
      a_sol, b_sol, plane.positive_side_geometry, plane.negative_side_geometry,
      plane.expressed_link, plane.order, decision_variables_val);
}

}  // namespace

VectorX<symbolic::Variable> GetTForPlane(
    const BodyIndex positive_side_link, const BodyIndex negative_side_link,
    const RationalForwardKinematics& rat_fk, CspaceRegionType region_type) {
  switch (region_type) {
    case CspaceRegionType::kGenericPolytope: {
      return rat_fk.t();
    }
    case CspaceRegionType::kAxisAlignedBoundingBox: {
      return rat_fk.FindTOnPath(positive_side_link, negative_side_link);
    }
    default: {
      throw std::runtime_error("Unknown region type.");
    }
  }
}

CspaceFreeRegion::CspaceFreeRegion(
    const systems::Diagram<double>& diagram,
    const multibody::MultibodyPlant<double>* plant,
    const geometry::SceneGraph<double>* scene_graph,
    SeparatingPlaneOrder plane_order, CspaceRegionType cspace_region_type,
    double separating_polytope_delta)
    : rational_forward_kinematics_(*plant),
      scene_graph_{scene_graph},
      link_geometries_{GetCollisionGeometries(diagram, plant, scene_graph)},
      plane_order_for_polytope_{plane_order},
      cspace_region_type_{cspace_region_type},
      separating_polytope_delta_{separating_polytope_delta} {
  DRAKE_DEMAND(separating_polytope_delta_ > 0);
  // Now create the separating planes.
  std::map<SortedPair<BodyIndex>,
           std::vector<
               std::pair<const CollisionGeometry*, const CollisionGeometry*>>>
      collision_pairs;
  int num_collision_pairs = 0;
  const auto& model_inspector = scene_graph->model_inspector();
  for (const auto& [link1, geometries1] : link_geometries_) {
    for (const auto& [link2, geometries2] : link_geometries_) {
      if (link1 < link2) {
        // link_collision_pairs stores all the pair of collision geometry on
        // (link1, link2).
        std::vector<
            std::pair<const CollisionGeometry*, const CollisionGeometry*>>
            link_collision_pairs;
        // I need to check if the kinematics chain betwen link 1 and link 2 has
        // length 0.
        std::optional<bool> chain_has_length_zero;
        for (const auto& geometry1 : geometries1) {
          for (const auto& geometry2 : geometries2) {
            if (!model_inspector.CollisionFiltered(geometry1->id(),
                                                   geometry2->id())) {
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
              link_collision_pairs.emplace_back(geometry1.get(),
                                                geometry2.get());
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
  for (const auto& [link_pair, geometry_pairs] : collision_pairs) {
    for (const auto& geometry_pair : geometry_pairs) {
      Vector3<symbolic::Expression> a;
      symbolic::Expression b;
      const symbolic::Monomial monomial_one{};
      VectorX<symbolic::Variable> plane_decision_vars;
      SeparatingPlaneOrder plane_order_geometry_pair =
          SeparatingPlaneOrder::kConstant;
      if (geometry_pair.first->type() == CollisionGeometryType::kPolytope &&
          geometry_pair.second->type() == CollisionGeometryType::kPolytope) {
        plane_order_geometry_pair = plane_order_for_polytope_;
      }
      if (plane_order_geometry_pair == SeparatingPlaneOrder::kConstant) {
        plane_decision_vars.resize(4);
        for (int i = 0; i < 3; ++i) {
          plane_decision_vars(i) = symbolic::Variable(
              "a" + std::to_string(separating_planes_.size() * 3 + i));
          plane_decision_vars(3) = symbolic::Variable(
              "b" + std::to_string(separating_planes_.size()));
        }
        CalcPlane<symbolic::Variable, symbolic::Variable, symbolic::Expression>(
            plane_decision_vars,
            GetTForPlane(link_pair.first(), link_pair.second(),
                         rational_forward_kinematics_, cspace_region_type),
            plane_order_geometry_pair, &a, &b);
      } else if (plane_order_geometry_pair == SeparatingPlaneOrder::kAffine) {
        const VectorX<symbolic::Variable> t_for_plane =
            GetTForPlane(link_pair.first(), link_pair.second(),
                         rational_forward_kinematics_, cspace_region_type);
        plane_decision_vars.resize(4 * t_for_plane.rows() + 4);
        for (int i = 0; i < plane_decision_vars.rows(); ++i) {
          plane_decision_vars(i) =
              symbolic::Variable(fmt::format("plane_var{}", i));
        }
        CalcPlane<symbolic::Variable, symbolic::Variable, symbolic::Expression>(
            plane_decision_vars, t_for_plane, plane_order_geometry_pair, &a,
            &b);
      }
      separating_planes_.emplace_back(
          a, b, geometry_pair.first, geometry_pair.second,
          internal::FindBodyInTheMiddleOfChain(*plant, link_pair.first(),
                                               link_pair.second()),
          plane_order_geometry_pair, plane_decision_vars);
      map_geometries_to_separating_planes_.emplace(
          SortedPair<geometry::GeometryId>(geometry_pair.first->id(),
                                           geometry_pair.second->id()),
          static_cast<int>(separating_planes_.size()) - 1);
    }
  }
}

std::vector<LinkOnPlaneSideRational>
CspaceFreeRegion::GenerateRationalsForLinkOnOneSideOfPlane(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs)
    const {
  std::unordered_map<DirectedKinematicsChain,
                     RationalForwardKinematics::Pose<symbolic::Polynomial>,
                     DirectedKinematicsChainHash>
      body_pair_to_X_AB_multilinear;
  std::vector<LinkOnPlaneSideRational> rationals;
  for (const auto& separating_plane : separating_planes_) {
    if (!IsGeometryPairCollisionIgnored(
            separating_plane.positive_side_geometry->id(),
            separating_plane.negative_side_geometry->id(),
            filtered_collision_pairs)) {
      // First compute X_AB for both side of the geometries.
      for (const PlaneSide plane_side :
           {PlaneSide::kPositive, PlaneSide::kNegative}) {
        const CollisionGeometry* link_geometry;
        const CollisionGeometry* other_side_geometry;
        if (plane_side == PlaneSide::kPositive) {
          link_geometry = separating_plane.positive_side_geometry;
          other_side_geometry = separating_plane.negative_side_geometry;
        } else {
          link_geometry = separating_plane.negative_side_geometry;
          other_side_geometry = separating_plane.positive_side_geometry;
        }
        const DirectedKinematicsChain expressed_to_link(
            separating_plane.expressed_link, link_geometry->body_index());
        auto it = body_pair_to_X_AB_multilinear.find(expressed_to_link);
        if (it == body_pair_to_X_AB_multilinear.end()) {
          body_pair_to_X_AB_multilinear.emplace_hint(
              it, expressed_to_link,
              rational_forward_kinematics_.CalcLinkPoseAsMultilinearPolynomial(
                  q_star, link_geometry->body_index(),
                  separating_plane.expressed_link));
        }
        it = body_pair_to_X_AB_multilinear.find(expressed_to_link);
        const RationalForwardKinematics::Pose<symbolic::Polynomial>&
            X_AB_multilinear = it->second;
        const std::vector<LinkOnPlaneSideRational> rationals_expressed_to_link =
            GenerateLinkOnOneSideOfPlaneRationalFunction(
                rational_forward_kinematics_, link_geometry,
                other_side_geometry, X_AB_multilinear, separating_plane.a,
                separating_plane.b, plane_side, separating_plane.order,
                separating_polytope_delta_);
        // I cannot use "insert" function to append vectors, since
        // LinkOnPlaneSideRational contains const members, hence it does
        // not have an assignment operator.
        std::copy(rationals_expressed_to_link.begin(),
                  rationals_expressed_to_link.end(),
                  std::back_inserter(rationals));
      }
    }
  }
  return rationals;
}

void CspaceFreeRegion::ConstructTBoundsPolynomial(
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
void CspaceFreeRegion::AddNonnegativeConstraintForGeometryOnOneSideOfPlane(
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

CspaceFreeRegion::CspacePolytopeProgramReturn
CspaceFreeRegion::ConstructProgramForCspacePolytope(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const std::vector<LinkOnPlaneSideRational>& rationals,
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
            separating_plane.positive_side_geometry->id(),
            separating_plane.negative_side_geometry->id(),
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
    AddNonnegativeConstraintForGeometryOnOneSideOfPlane(
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
  const auto& query_port = plant.get_geometry_query_input_port();
  const auto& query_object =
      query_port.Eval<geometry::QueryObject<double>>(context);
  const auto& inspector = scene_graph_->model_inspector();
  for (const auto& geometry_pair : inspector.GetCollisionCandidates()) {
    const geometry::SignedDistancePair<double> signed_distance_pair =
        query_object.ComputeSignedDistancePairClosestPoints(
            geometry_pair.first, geometry_pair.second);
    if (signed_distance_pair.distance < 0) {
      return true;
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
    VectorX<symbolic::Variable>* separating_plane_vars,
    std::vector<std::vector<int>>* separating_plane_to_tuples,
    std::vector<std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>*
        separating_plane_to_lorentz_cone_constraints) const {
  DRAKE_DEMAND(separating_plane_to_lorentz_cone_constraints != nullptr);
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
  ConstructTuplesInMemory(
      q_star, filtered_collision_pairs, C_rows, t.rows(), alternation_tuples,
      lagrangian_gram_vars, verified_gram_vars, separating_plane_vars,
      separating_plane_to_tuples, separating_plane_to_lorentz_cone_constraints);
}

void CspaceFreeRegion::ConstructTuplesInMemory(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs, int C_rows,
    int t_rows,
    std::vector<CspaceFreeRegion::CspacePolytopeTuple>* alternation_tuples,
    VectorX<symbolic::Variable>* lagrangian_gram_vars,
    VectorX<symbolic::Variable>* verified_gram_vars,
    VectorX<symbolic::Variable>* separating_plane_vars,
    std::vector<std::vector<int>>* separating_plane_to_tuples,
    std::vector<std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>*
        separating_plane_to_lorentz_cone_constraints) const {
  // Build tuples.
  const auto rationals = GenerateRationalsForLinkOnOneSideOfPlane(
      q_star, filtered_collision_pairs);
  separating_plane_to_tuples->resize(this->separating_planes().size());
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
    std::vector<int> polytope_lagrangian_gram_lower_start(C_rows);
    const int gram_lower_size =
        monomial_basis_chain.rows() * (monomial_basis_chain.rows() + 1) / 2;
    for (int j = 0; j < C_rows; ++j) {
      polytope_lagrangian_gram_lower_start[j] =
          lagrangian_gram_vars_count + j * gram_lower_size;
    }
    std::vector<int> t_lower_lagrangian_gram_lower_start(t_rows);
    for (int j = 0; j < t_rows; ++j) {
      t_lower_lagrangian_gram_lower_start[j] =
          lagrangian_gram_vars_count + (C_rows + j) * gram_lower_size;
    }
    std::vector<int> t_upper_lagrangian_gram_lower_start(t_rows);
    for (int j = 0; j < t_rows; ++j) {
      t_upper_lagrangian_gram_lower_start[j] =
          lagrangian_gram_vars_count + (C_rows + t_rows + j) * gram_lower_size;
    }
    alternation_tuples->emplace_back(
        rationals[i].rational.numerator(), polytope_lagrangian_gram_lower_start,
        t_lower_lagrangian_gram_lower_start,
        t_upper_lagrangian_gram_lower_start, verified_gram_vars_count,
        monomial_basis_chain);
    (*separating_plane_to_tuples)[this->map_geometries_to_separating_planes_.at(
                                      SortedPair<geometry::GeometryId>(
                                          rationals[i].link_geometry->id(),
                                          rationals[i]
                                              .other_side_link_geometry->id()))]
        .push_back(alternation_tuples->size() - 1);
    // Each Gram matrix is of size monomial_basis_chain.rows() *
    // (monomial_basis_chain.rows() + 1) / 2. Each rational needs C.rows() + 2 *
    // t.rows() Lagrangians.
    lagrangian_gram_vars_count += gram_lower_size * (C_rows + 2 * t_rows);
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
  // Set the separating plane lorentz cone constraints.
  separating_plane_to_lorentz_cone_constraints->clear();
  separating_plane_to_lorentz_cone_constraints->resize(
      separating_planes_.size());
  for (const auto& rational : rationals) {
    if (!rational.lorentz_cone_constraints.empty()) {
      const int plane_index = map_geometries_to_separating_planes_.at(
          SortedPair<geometry::GeometryId>(
              rational.link_geometry->id(),
              rational.other_side_link_geometry->id()));
      (*separating_plane_to_lorentz_cone_constraints)[plane_index].insert(
          (*separating_plane_to_lorentz_cone_constraints)[plane_index].end(),
          rational.lorentz_cone_constraints.begin(),
          rational.lorentz_cone_constraints.end());
    }
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
    const std::vector<solvers::Binding<solvers::LorentzConeConstraint>>&
        separating_plane_lorentz_cone_constraints,
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
  VectorX<symbolic::Polynomial> t_minus_t_lower(t.rows());
  VectorX<symbolic::Polynomial> t_upper_minus_t(t.rows());
  ConstructTBoundsPolynomial(t_monomials, t_lower, t_upper, &t_minus_t_lower,
                             &t_upper_minus_t);

  VectorX<symbolic::Polynomial> d_minus_Ct;
  std::unordered_set<int> C_redundant_indices, t_lower_redundant_indices,
      t_upper_redundant_indices;
  if (C.rows() > 0) {
    CalcDminusCt<double>(C, d, t_monomials, &d_minus_Ct);
    // find the redundant inequalities.
    if (redundant_tighten.has_value()) {
      FindRedundantInequalities(C, d, t_lower, t_upper,
                                redundant_tighten.value(), &C_redundant_indices,
                                &t_lower_redundant_indices,
                                &t_upper_redundant_indices);
    }
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
    // 2. Constrain the Gram matrix to be PSD.
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
    for (int i = 0; i < t_minus_t_lower.rows(); ++i) {
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
  // Now add the Lorentz cone constraints for the separting planes.
  for (const auto& binding : separating_plane_lorentz_cone_constraints) {
    prog->AddConstraint(binding);
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
    const std::vector<
        std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>&
        separating_plane_to_lorentz_cone_constraints,
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
  // Add Lorentz cone constraints for separating planes.
  for (const auto& bindings : separating_plane_to_lorentz_cone_constraints) {
    for (const auto& binding : bindings) {
      prog->AddConstraint(binding);
    }
  }
  return prog;
}

namespace internal {
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
}  // namespace internal

namespace {
// Find the largest inscribed ellipsoid in the polytope
// C*t<=d, t_lower <= t <= t_upper. The ellipsoid is parameterized as {P*y+q |
// |y|<=1}. We may need to backoff the inscribed ellipsoid a bit to find a
// strict feasible solution.
void FindLargestInscribedEllipsoid(
    const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
    const Eigen::VectorXd& t_lower, const Eigen::VectorXd& t_upper,
    double backoff_scale, EllipsoidVolume ellipsoid_volume,
    const solvers::SolverOptions& solver_options, bool verbose,
    Eigen::MatrixXd* P_sol, Eigen::VectorXd* q_sol, double* max_cost) {
  solvers::MathematicalProgram prog;
  const int nt = t_lower.rows();
  const auto P = prog.NewSymmetricContinuousVariables(nt, "P");
  const auto q = prog.NewContinuousVariables(nt, "q");
  AddInscribedEllipsoid(&prog, C, d, t_lower, t_upper, P, q, true);
  std::string cost_name;
  switch (ellipsoid_volume) {
    case EllipsoidVolume::kLog: {
      prog.AddMaximizeLogDeterminantCost(P.cast<symbolic::Expression>());
      cost_name = "log(det(P))";
      break;
    }
    case EllipsoidVolume::kNthRoot: {
      AddMaximizeEigenValueGeometricMean(&prog, P);
      cost_name = fmt::format("power(det(P), 1/{})", SmallestPower2(P.rows()));
      break;
    }
  }
  const auto ellipsoid_cost = prog.linear_costs()[0];
  auto result = solvers::Solve(prog, std::nullopt, solver_options);
  *max_cost = -result.get_optimal_cost();
  if (verbose) {
    drake::log()->info(fmt::format(
        "max({})={}, solver_time {}", cost_name, *max_cost,
        result.get_solver_details<solvers::MosekSolver>().optimizer_time));
  }
  if (backoff_scale > 0) {
    result = internal::BackoffProgram(&prog, result.get_optimal_cost(),
                                      backoff_scale, solver_options);
    *max_cost = -result.EvalBinding(ellipsoid_cost)(0);
    if (verbose) {
      drake::log()->info(fmt::format(
          "backoff with {}={}, solver time {}", cost_name, *max_cost,
          result.get_solver_details<solvers::MosekSolver>().optimizer_time));
    }
  }
  *P_sol = result.GetSolution(P);
  *q_sol = result.GetSolution(q);
}
}  // namespace

namespace internal {
// Some of the separating planes will be ignored by filtered_collision_pairs.
// Returns std::vector<bool> to indicate if each plane is active or not.
std::vector<bool> IsPlaneActive(
    const std::vector<SeparatingPlane<symbolic::Variable>>& separating_planes,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  std::vector<bool> is_plane_active(separating_planes.size(), true);
  for (int i = 0; i < static_cast<int>(separating_planes.size()); ++i) {
    if (filtered_collision_pairs.count(SortedPair<geometry::GeometryId>(
            separating_planes[i].positive_side_geometry->id(),
            separating_planes[i].negative_side_geometry->id())) != 0) {
      is_plane_active[i] = false;
    }
  }
  return is_plane_active;
}

// For a given polytopic C-space region C * t <= d, t_lower <= t <= t_upper,
// verify if this region is collision-free by solving the SOS program to find
// the separating planes and the Lagrangian multipliers. Return true if the SOS
// is successful, false otherwise.
bool FindLagrangianAndSeparatingPlanesSingleThread(
    const CspaceFreeRegion& cspace_free_region,
    const std::vector<CspaceFreeRegion::CspacePolytopeTuple>&
        alternation_tuples,
    const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
    const VectorX<symbolic::Variable>& lagrangian_gram_vars,
    const VectorX<symbolic::Variable>& verified_gram_vars,
    const VectorX<symbolic::Variable>& separating_plane_vars,
    const std::vector<
        std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>&
        separating_plane_to_lorentz_cone_constraints,
    const Eigen::VectorXd& t_lower, const Eigen::VectorXd& t_upper,
    const VerificationOption& verification_option,
    std::optional<double> redundant_tighten,
    const solvers::SolverOptions& solver_options, bool verbose,
    const std::vector<bool>& is_plane_active,
    Eigen::VectorXd* lagrangian_gram_var_vals,
    Eigen::VectorXd* verified_gram_var_vals,
    Eigen::VectorXd* separating_plane_var_vals,
    std::vector<SeparatingPlane<double>>* separating_planes_sol) {
  std::vector<solvers::Binding<solvers::LorentzConeConstraint>>
      separating_plane_lorentz_cone_constraints;
  for (const auto& bindings : separating_plane_to_lorentz_cone_constraints) {
    separating_plane_lorentz_cone_constraints.insert(
        separating_plane_lorentz_cone_constraints.end(), bindings.begin(),
        bindings.end());
  }
  auto prog_lagrangian = cspace_free_region.ConstructLagrangianProgram(
      alternation_tuples, C, d, lagrangian_gram_vars, verified_gram_vars,
      separating_plane_vars, separating_plane_lorentz_cone_constraints, t_lower,
      t_upper, verification_option, redundant_tighten, nullptr, nullptr);
  auto result_lagrangian =
      solvers::Solve(*prog_lagrangian, std::nullopt, solver_options);
  if (!result_lagrangian.is_success()) {
    if (verbose) {
      drake::log()->warn(fmt::format("Find Lagrangian failed"));
    }
    return false;
  } else {
    if (verbose) {
      drake::log()->info(fmt::format(
          "Lagrangian SOS takes {} seconds",
          result_lagrangian.get_solver_details<solvers::MosekSolver>()
              .optimizer_time));
    }
  }
  *lagrangian_gram_var_vals =
      result_lagrangian.GetSolution(lagrangian_gram_vars);
  *verified_gram_var_vals = result_lagrangian.GetSolution(verified_gram_vars);
  *separating_plane_var_vals =
      result_lagrangian.GetSolution(separating_plane_vars);
  *separating_planes_sol = internal::GetSeparatingPlanesSolution(
      cspace_free_region, is_plane_active, result_lagrangian);
  return true;
}

namespace {
// Checks if a future has completed execution.
// This function is taken from monte_carlo.cc. It will be used in the "thread
// pool" implementation (which doesn't use openMP).
template <typename T>
bool IsFutureReady(const std::future<T>& future) {
  // future.wait_for() is the only method to check the status of a future
  // without waiting for it to complete.
  const std::future_status status =
      future.wait_for(std::chrono::milliseconds(1));
  return (status == std::future_status::ready);
}
}  // namespace

// Same as FindLagrangianAndSeparatingPlanesSingleThread. But instead of solving
// one SOS with all the separating planes simultaneously, we solve many small
// SOS in parallel, each SOS for one separating plane.
bool FindLagrangianAndSeparatingPlanesMultiThread(
    const CspaceFreeRegion& cspace_free_region,
    const std::vector<CspaceFreeRegion::CspacePolytopeTuple>&
        alternation_tuples,
    const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
    const VectorX<symbolic::Variable>& lagrangian_gram_vars,
    const VectorX<symbolic::Variable>& verified_gram_vars,
    const VectorX<symbolic::Variable>& separating_plane_vars,
    const std::vector<
        std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>&
        separating_plane_to_lorentz_cone_constraints,
    const Eigen::VectorXd& t_lower, const Eigen::VectorXd& t_upper,
    const VerificationOption& verification_option,
    std::optional<double> redundant_tighten,
    const solvers::SolverOptions& solver_options, bool verbose,
    const std::vector<std::vector<int>>& separating_plane_to_tuples,
    int num_threads, Eigen::VectorXd* lagrangian_gram_var_vals,
    Eigen::VectorXd* verified_gram_var_vals,
    Eigen::VectorXd* separating_plane_var_vals,
    std::vector<SeparatingPlane<double>>* separating_planes_sol) {
  // To avoid data race, we allocate memory for lagrangian_gram_var_vals and
  // separating_planes_sol before launching the multiple threads. To
  // allocate memory for separating_planes_sol, we count the number of active
  // planes.
  int num_active_planes = 0;
  for (const auto& tuple_indices : separating_plane_to_tuples) {
    if (!tuple_indices.empty()) {
      num_active_planes++;
    }
  }

  std::vector<int> active_plane_count_to_plane_index(num_active_planes);
  num_active_planes = 0;
  for (int i = 0;
       i < static_cast<int>(cspace_free_region.separating_planes().size());
       ++i) {
    if (!separating_plane_to_tuples[i].empty()) {
      active_plane_count_to_plane_index[num_active_planes] = i;
      num_active_planes++;
    }
  }

  // Allocate memory for lagrangian_gram_var_vals;
  lagrangian_gram_var_vals->resize(lagrangian_gram_vars.rows());
  verified_gram_var_vals->resize(verified_gram_vars.rows());
  separating_plane_var_vals->resize(separating_plane_vars.rows());
  separating_planes_sol->resize(num_active_planes);
  // is_success[i] = std::nullopt means the thread for the i'th separating plane
  // hasn't been dispatched yet.
  std::vector<std::optional<bool>> is_success(num_active_planes, std::nullopt);
  // To set values of separating_plane_vars, I build this map from the variable
  // to its index in separating_plane_vars.
  std::unordered_map<symbolic::Variable::Id, int> separating_plane_var_indices;
  for (int i = 0; i < separating_plane_vars.rows(); ++i) {
    separating_plane_var_indices.emplace(separating_plane_vars(i).get_id(), i);
  }

  // This lambda function formulates and solves a small SOS program for each
  // separating plane. It finds the separating plane and the Lagrangian
  // multiplier for a pair of collision geometries.
  auto solve_small_sos =
      [&cspace_free_region, &alternation_tuples, &C, &d, &lagrangian_gram_vars,
       &verified_gram_vars, &separating_plane_vars,
       &separating_plane_to_lorentz_cone_constraints, &t_lower, &t_upper,
       &verification_option, &redundant_tighten, &solver_options,
       &separating_plane_to_tuples, &separating_plane_var_indices,
       &active_plane_count_to_plane_index, lagrangian_gram_var_vals,
       separating_planes_sol, verified_gram_var_vals, separating_plane_var_vals,
       &is_success](int active_plane_count) {
        const int plane_index =
            active_plane_count_to_plane_index[active_plane_count];
        const std::vector<int>& tuple_indices =
            separating_plane_to_tuples[plane_index];
        std::vector<CspaceFreeRegion::CspacePolytopeTuple> plane_tuples;
        plane_tuples.reserve(tuple_indices.size());
        for (const int tuple_index : tuple_indices) {
          plane_tuples.push_back(alternation_tuples[tuple_index]);
        }
        auto prog = cspace_free_region.ConstructLagrangianProgram(
            plane_tuples, C, d, lagrangian_gram_vars, verified_gram_vars,
            separating_plane_vars,
            separating_plane_to_lorentz_cone_constraints[plane_index], t_lower,
            t_upper, verification_option, redundant_tighten, nullptr, nullptr);
        const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
        is_success[active_plane_count] = result.is_success();
        if (result.is_success()) {
          // Now fill in lagrangian_gram_var_vals;
          for (const int tuple_index : tuple_indices) {
            const int gram_rows =
                alternation_tuples[tuple_index].monomial_basis.rows();
            const int gram_lower_size = gram_rows * (gram_rows + 1) / 2;

            auto fill_lagrangian =
                [gram_lower_size, lagrangian_gram_var_vals, &result,
                 &lagrangian_gram_vars](
                    const std::vector<int>& lagrangian_gram_start) {
                  for (int start : lagrangian_gram_start) {
                    lagrangian_gram_var_vals->segment(start, gram_lower_size) =
                        result.GetSolution(lagrangian_gram_vars.segment(
                            start, gram_lower_size));
                  }
                };
            fill_lagrangian(alternation_tuples[tuple_index]
                                .polytope_lagrangian_gram_lower_start);
            fill_lagrangian(alternation_tuples[tuple_index]
                                .t_lower_lagrangian_gram_lower_start);
            fill_lagrangian(alternation_tuples[tuple_index]
                                .t_upper_lagrangian_gram_lower_start);
            verified_gram_var_vals->segment(
                alternation_tuples[tuple_index]
                    .verified_polynomial_gram_lower_start,
                gram_lower_size) =
                result.GetSolution(verified_gram_vars.segment(
                    alternation_tuples[tuple_index]
                        .verified_polynomial_gram_lower_start,
                    gram_lower_size));
          }
          // Now get the solution for separating_plane.
          (*separating_planes_sol)[active_plane_count] =
              GetSeparatingPlaneSolution(
                  cspace_free_region.separating_planes()[plane_index], result);
          // Set separating_plane_var_vals.
          for (int i = 0;
               i < cspace_free_region.separating_planes()[plane_index]
                       .decision_variables.rows();
               ++i) {
            const symbolic::Variable& var =
                cspace_free_region.separating_planes()[plane_index]
                    .decision_variables(i);
            (*separating_plane_var_vals)(separating_plane_var_indices.at(
                var.get_id())) = result.GetSolution(var);
          }
        }
        return active_plane_count;
      };
  if (num_threads > 0) {
    // We implement the "thread pool" idea here, by following
    // MonteCarloSimulationParallel class. This implementation doesn't use
    // openMP library. Storage for active parallel SOS operations.
    std::list<std::future<int>> active_operations;
    // Keep track of how many SOS have been dispatched already.
    int sos_dispatched = 0;
    // If any SOS is infeasible, then we terminate all other SOS and report
    // failure.
    bool found_infeasible = false;
    while (
        (active_operations.size() > 0 || sos_dispatched < num_active_planes) &&
        !found_infeasible) {
      // Check for completed operations.
      for (auto operation = active_operations.begin();
           operation != active_operations.end();) {
        if (IsFutureReady(*operation)) {
          // This call to future.get() is necessary to propagate any exception
          // thrown during SOS setup/solve.
          const int active_plane_count = operation->get();
          drake::log()->debug("SOS {} completed, is_success {}",
                              active_plane_count,
                              is_success[active_plane_count].value());
          if (!(is_success[active_plane_count].value())) {
            found_infeasible = true;
            break;
          }
          // Erase returns iterator to the next node in the list.
          operation = active_operations.erase(operation);
        } else {
          // Advance to next node in the list.
          ++operation;
        }
      }

      // Dispatch new SOS.
      while (static_cast<int>(active_operations.size()) < num_threads &&
             sos_dispatched < num_active_planes) {
        active_operations.emplace_back(std::async(
            std::launch::async, std::move(solve_small_sos), sos_dispatched));
        drake::log()->debug("SOS {} dispatched", sos_dispatched);
        ++sos_dispatched;
      }

      // Wait a bit before checking for completion.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  } else {
    // Create as many threads as possible and join all threads.
    std::vector<std::thread> threads;
    for (int active_plane_count = 0; active_plane_count < num_active_planes;
         ++active_plane_count) {
      threads.push_back(std::thread(solve_small_sos, active_plane_count));
    }
    for (auto& th : threads) {
      th.join();
    }
  }
  if (std::all_of(is_success.begin(), is_success.end(),
                  [](std::optional<bool> flag) {
                    return flag.has_value() && flag.value();
                  })) {
    if (verbose) {
      drake::log()->info("Found Lagrangian multiplier and separating planes");
    }
    return true;
  } else {
    if (verbose) {
      std::string bad_pairs;
      const auto& inspector =
          cspace_free_region.scene_graph().model_inspector();
      for (int active_plane_count = 0; active_plane_count < num_active_planes;
           ++active_plane_count) {
        const int plane_index =
            active_plane_count_to_plane_index[active_plane_count];
        if (is_success[active_plane_count].has_value() &&
            !(is_success[active_plane_count].value())) {
          bad_pairs.append(fmt::format(
              "({}, {})\n",
              inspector.GetName(
                  cspace_free_region.separating_planes()[plane_index]
                      .positive_side_geometry->id()),
              inspector.GetName(
                  cspace_free_region.separating_planes()[plane_index]
                      .negative_side_geometry->id())));
        }
      }

      drake::log()->warn(fmt::format(
          "Cannot find Lagrangian multiplier and separating planes for \n{}",
          bad_pairs));
    }
    return false;
  }
}

bool FindLagrangianAndSeparatingPlanes(
    const CspaceFreeRegion& cspace_free_region,
    const std::vector<CspaceFreeRegion::CspacePolytopeTuple>&
        alternation_tuples,
    const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
    const VectorX<symbolic::Variable>& lagrangian_gram_vars,
    const VectorX<symbolic::Variable>& verified_gram_vars,
    const VectorX<symbolic::Variable>& separating_plane_vars,
    const std::vector<
        std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>&
        separating_plane_to_lorentz_cone_constraints,
    const Eigen::VectorXd& t_lower, const Eigen::VectorXd& t_upper,
    const VerificationOption& verification_option,
    std::optional<double> redundant_tighten,
    const solvers::SolverOptions& solver_options, bool verbose,
    std::optional<int> num_threads,
    const std::vector<std::vector<int>>& separating_plane_to_tuples,
    Eigen::VectorXd* lagrangian_gram_var_vals,
    Eigen::VectorXd* verified_gram_var_vals,
    Eigen::VectorXd* separating_plane_var_vals,
    CspaceFreeRegionSolution* cspace_free_region_solution) {
  bool ret_val{true};
  if (num_threads.has_value()) {
    ret_val = FindLagrangianAndSeparatingPlanesMultiThread(
        cspace_free_region, alternation_tuples, C, d, lagrangian_gram_vars,
        verified_gram_vars, separating_plane_vars,
        separating_plane_to_lorentz_cone_constraints, t_lower, t_upper,
        verification_option, redundant_tighten, solver_options, verbose,
        separating_plane_to_tuples, num_threads.value(),
        lagrangian_gram_var_vals, verified_gram_var_vals,
        separating_plane_var_vals,
        &(cspace_free_region_solution->separating_planes));
    if (ret_val) {
      // ensure that the planes and polytope solutions match
      cspace_free_region_solution->C = C;
      cspace_free_region_solution->d = d;
    }
    return ret_val;
  } else {
    std::vector<bool> is_plane_active(
        cspace_free_region.separating_planes().size(), false);
    for (int i = 0; i < static_cast<int>(is_plane_active.size()); ++i) {
      is_plane_active[i] = !separating_plane_to_tuples[i].empty();
    }
    ret_val = FindLagrangianAndSeparatingPlanesSingleThread(
        cspace_free_region, alternation_tuples, C, d, lagrangian_gram_vars,
        verified_gram_vars, separating_plane_vars,
        separating_plane_to_lorentz_cone_constraints, t_lower, t_upper,
        verification_option, redundant_tighten, solver_options, verbose,
        is_plane_active, lagrangian_gram_var_vals, verified_gram_var_vals,
        separating_plane_var_vals,
        &(cspace_free_region_solution->separating_planes));
    if (ret_val) {
      cspace_free_region_solution->C = C;
      cspace_free_region_solution->d = d;
    }
    return ret_val;
  }
}

/**
 * @param is_plane_active: sometimes the collision between a pair of geometries
 * is ignored. is_plane_active[i] means whether the plane
 * cspace_free_region.separating_planes()[i] is active or not.
 */
std::vector<SeparatingPlane<double>> GetSeparatingPlanesSolution(
    const CspaceFreeRegion& cspace_free_region,
    const std::vector<bool>& is_plane_active,
    const solvers::MathematicalProgramResult& result) {
  std::vector<SeparatingPlane<double>> planes_sol;
  DRAKE_DEMAND(is_plane_active.size() ==
               cspace_free_region.separating_planes().size());
  int num_active_planes = 0;
  for (bool flag : is_plane_active) {
    if (flag) {
      num_active_planes++;
    }
  }
  planes_sol.reserve(num_active_planes);

  for (int plane_index = 0;
       plane_index <
       static_cast<int>(cspace_free_region.separating_planes().size());
       ++plane_index) {
    if (is_plane_active[plane_index]) {
      planes_sol.push_back(GetSeparatingPlaneSolution(
          cspace_free_region.separating_planes()[plane_index], result));
    }
  }
  return planes_sol;
}
}  // namespace internal

void CspaceFreeRegion::CspacePolytopeBilinearAlternation(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs,
    const Eigen::Ref<const Eigen::MatrixXd>& C_init,
    const Eigen::Ref<const Eigen::VectorXd>& d_init,
    const CspaceFreeRegion::BilinearAlternationOption&
        bilinear_alternation_option,
    const solvers::SolverOptions& solver_options,
    const std::optional<Eigen::MatrixXd>& t_inner_pts,
    const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
        inner_polytope,
    CspaceFreeRegionSolution* cspace_free_region_solution,
    std::vector<double>* polytope_volumes,
    std::vector<double>* ellipsoid_determinants) const {
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
  DRAKE_DEMAND(polytope_volumes != nullptr);
  polytope_volumes->clear();
  DRAKE_DEMAND(ellipsoid_determinants != nullptr);
  ellipsoid_determinants->clear();

  // First normalize each row of C and d, such that each row of C has a unit
  // norm. This is important as later when we search for polytope, we impose the
  // constraint |C.row(i)|<=1, hence we need to first start with C and d
  // satisfying this constraint.
  //  Eigen::MatrixXd C_val = C_init;
  //  Eigen::VectorXd d_val = d_init;
  (cspace_free_region_solution->C) = C_init;
  (cspace_free_region_solution->d) = d_init;
  for (int i = 0; i < C_rows; ++i) {
    const double C_row_norm = (cspace_free_region_solution->C).row(i).norm();
    (cspace_free_region_solution->C).row(i) /= C_row_norm;
    (cspace_free_region_solution->d)(i) /= C_row_norm;
  }
  std::vector<CspaceFreeRegion::CspacePolytopeTuple> alternation_tuples;
  VectorX<symbolic::Polynomial> d_minus_Ct;
  Eigen::VectorXd t_lower, t_upper;
  VectorX<symbolic::Polynomial> t_minus_t_lower, t_upper_minus_t;
  MatrixX<symbolic::Variable> C_var;
  VectorX<symbolic::Variable> d_var, lagrangian_gram_vars, verified_gram_vars,
      separating_plane_vars;
  std::vector<std::vector<int>> separating_plane_to_tuples;
  std::vector<std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>
      separating_plane_to_lorentz_cone_constraints;
  GenerateTuplesForBilinearAlternation(
      q_star, filtered_collision_pairs, C_rows, &alternation_tuples,
      &d_minus_Ct, &t_lower, &t_upper, &t_minus_t_lower, &t_upper_minus_t,
      &C_var, &d_var, &lagrangian_gram_vars, &verified_gram_vars,
      &separating_plane_vars, &separating_plane_to_tuples,
      &separating_plane_to_lorentz_cone_constraints);
  if (bilinear_alternation_option.compute_polytope_volume) {
    drake::log()->info(
        fmt::format("Polytope volume {}",
                    CalcCspacePolytopeVolume((cspace_free_region_solution->C),
                                             (cspace_free_region_solution->d),
                                             t_lower, t_upper)));
  }

  const std::vector<bool> is_plane_active =
      internal::IsPlaneActive(separating_planes_, filtered_collision_pairs);

  VectorX<symbolic::Variable> margin;
  int iter_count = 0;
  double cost_improvement = kInf;
  double previous_cost = -kInf;
  VerificationOption verification_option{};
  while (iter_count < bilinear_alternation_option.max_iters &&
         cost_improvement > bilinear_alternation_option.convergence_tol) {
    Eigen::VectorXd lagrangian_gram_var_vals, verified_gram_var_vals,
        separating_plane_var_vals;
    auto clock_start = std::chrono::system_clock::now();
    const bool find_lagrangian = internal::FindLagrangianAndSeparatingPlanes(
        *this, alternation_tuples, (cspace_free_region_solution->C),
        (cspace_free_region_solution->d), lagrangian_gram_vars,
        verified_gram_vars, separating_plane_vars,
        separating_plane_to_lorentz_cone_constraints, t_lower, t_upper,
        verification_option, bilinear_alternation_option.redundant_tighten,
        solver_options, bilinear_alternation_option.verbose,
        bilinear_alternation_option.num_threads, separating_plane_to_tuples,
        &lagrangian_gram_var_vals, &verified_gram_var_vals,
        &separating_plane_var_vals, cspace_free_region_solution);
    auto clock_now = std::chrono::system_clock::now();
    drake::log()->info(fmt::format(
        "Lagrangian step time {} s",
        static_cast<float>(
            std::chrono::duration_cast<std::chrono::milliseconds>(clock_now -
                                                                  clock_start)
                .count()) /
            1000));
    if (!find_lagrangian) {
      return;
    }
    // Now construct a program that finds the maximal inner ellipsoid in C*t<=d,
    // t_lower<=t<=t_upper. This program can be solved independently from
    // prog_lagrangian.
    double ellipsoid_cost_val;
    FindLargestInscribedEllipsoid(
        (cspace_free_region_solution->C), (cspace_free_region_solution->d),
        t_lower, t_upper, bilinear_alternation_option.lagrangian_backoff_scale,
        bilinear_alternation_option.ellipsoid_volume, solver_options,
        bilinear_alternation_option.verbose, &(cspace_free_region_solution->P),
        &(cspace_free_region_solution->q), &ellipsoid_cost_val);
    // Update the cost.
    cost_improvement = ellipsoid_cost_val - previous_cost;
    drake::log()->info(fmt::format("cost improvement {}", cost_improvement));
    previous_cost = ellipsoid_cost_val;

    // Now solve the polytope problem (fix Lagrangian).
    auto prog_polytope = ConstructPolytopeProgram(
        alternation_tuples, C_var, d_var, d_minus_Ct, lagrangian_gram_var_vals,
        verified_gram_vars, separating_plane_vars,
        separating_plane_to_lorentz_cone_constraints, t_minus_t_lower,
        t_upper_minus_t, verification_option);
    // Add the constraint that the polytope contains the ellipsoid
    margin = prog_polytope->NewContinuousVariables(C_var.rows(), "margin");
    AddOuterPolytope(prog_polytope.get(), (cspace_free_region_solution->P),
                     (cspace_free_region_solution->q), C_var, d_var, margin);

    // We know that the verified polytope has to be contained in the box t_lower
    // <= t <= t_upper. Hence there is no point to grow the polytope such that
    // any of its halfspace C.row(i) * t <= d(i) contains the entire box t_lower
    // <= t <= t_upper. Hence an upper bound of the margin δ is the maximal
    // distance from any vertices of the box t_lower <= t <= t_upper to the
    // ellipsoid.
    // Computing the distance from a point to a hyperellipsoid is non-trivial.
    // Here we use an upper bound of this distance, which is the maximal
    // distance between any two points within the box.
    const double margin_upper_bound = (t_upper - t_lower).norm();
    prog_polytope->AddBoundingBoxConstraint(0, margin_upper_bound, margin);
    // Add the constraint that the polytope contains the t_inner_pts.
    if (t_inner_pts.has_value()) {
      for (int i = 0; i < t_inner_pts->cols(); ++i) {
        DRAKE_DEMAND((t_inner_pts->col(i).array() <= t_upper.array()).all());
        DRAKE_DEMAND((t_inner_pts->col(i).array() >= t_lower.array()).all());
      }
      AddCspacePolytopeContainment(prog_polytope.get(), C_var, d_var,
                                   t_inner_pts.value());
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
      drake::log()->info(
          fmt::format("mosek info {}, {}",
                      result_polytope.get_solver_details<solvers::MosekSolver>()
                          .solution_status,
                      result_polytope.get_solution_result()));

      drake::log()->warn(fmt::format(
          "Failed to find the polytope at iteration {}", iter_count));
      return;
    }

    if (bilinear_alternation_option.verbose) {
      drake::log()->info(
          fmt::format("Iter: {}, polytope step cost {}, solver time {}",
                      iter_count, result_polytope.get_optimal_cost(),
                      result_polytope.get_solver_details<solvers::MosekSolver>()
                          .optimizer_time));
    }
    if (bilinear_alternation_option.polytope_backoff_scale > 0) {
      result_polytope = internal::BackoffProgram(
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
    (cspace_free_region_solution->C) = result_polytope.GetSolution(C_var);
    (cspace_free_region_solution->d) = result_polytope.GetSolution(d_var);
    if (bilinear_alternation_option.compute_polytope_volume) {
      polytope_volumes->push_back(CalcCspacePolytopeVolume(
          (cspace_free_region_solution->C), (cspace_free_region_solution->d),
          t_lower, t_upper));
      drake::log()->info(
          fmt::format("Polytope volume {}", polytope_volumes->back()));
    }
    ellipsoid_determinants->push_back(
        cspace_free_region_solution->P.determinant());

    (cspace_free_region_solution->separating_planes) =
        internal::GetSeparatingPlanesSolution(*this, is_plane_active,
                                              result_polytope);

    iter_count += 1;
  }
  double ellipsoid_cost_val;

  FindLargestInscribedEllipsoid(
      (cspace_free_region_solution->C), (cspace_free_region_solution->d),
      t_lower, t_upper, bilinear_alternation_option.lagrangian_backoff_scale,
      bilinear_alternation_option.ellipsoid_volume, solver_options,
      bilinear_alternation_option.verbose, &(cspace_free_region_solution->P),
      &(cspace_free_region_solution->q), &ellipsoid_cost_val);
}

void CspaceFreeRegion::CspacePolytopeBinarySearch(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d_init,
    const BinarySearchOption& binary_search_option,
    const solvers::SolverOptions& solver_options,
    const std::optional<Eigen::MatrixXd>& t_inner_pts,
    const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
        inner_polytope,
    CspaceFreeRegionSolution* cspace_free_region_solution) const {
  // The polytope region is C * t <= d_without_epsilon + epsilon. We might
  // change d_without_epsilon during the binary search process.
  (*cspace_free_region_solution).C = C;
  (*cspace_free_region_solution).d = Eigen::VectorXd(C.rows());

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
  std::vector<std::vector<int>> separating_plane_to_tuples;
  std::vector<std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>
      separating_plane_to_lorentz_cone_constraints;
  GenerateTuplesForBilinearAlternation(
      q_star, filtered_collision_pairs, C_rows, &alternation_tuples,
      &d_minus_Ct, &t_lower, &t_upper, &t_minus_t_lower, &t_upper_minus_t,
      &C_var, &d_var, &lagrangian_gram_vars, &verified_gram_vars,
      &separating_plane_vars, &separating_plane_to_tuples,
      &separating_plane_to_lorentz_cone_constraints);
  DRAKE_DEMAND(binary_search_option.epsilon_min >=
               FindEpsilonLower(C, d_init, t_lower, t_upper, t_inner_pts,
                                inner_polytope));
  const Eigen::VectorXd d_max = ComputeMaxD(C, t_lower, t_upper);

  VerificationOption verification_option{};
  const auto is_plane_active =
      internal::IsPlaneActive(separating_planes_, filtered_collision_pairs);
  // Checks if C*t<=d, t_lower<=t<=t_upper is collision free.
  // This function will update d_sol when the polytope is collision free.
  // TODO(hongkai.dai): refactor this lambda function to its own method.
  auto is_polytope_collision_free =
      [this, &alternation_tuples, &C, &lagrangian_gram_vars,
       &verified_gram_vars, &separating_plane_vars,
       &separating_plane_to_lorentz_cone_constraints, &t_lower, &t_upper,
       &binary_search_option, &verification_option, &solver_options, &C_var,
       &d_var, &d_minus_Ct, &t_minus_t_lower, &t_upper_minus_t, &t_inner_pts,
       &inner_polytope, &d_max, &is_plane_active, &separating_plane_to_tuples,
       cspace_free_region_solution](const Eigen::VectorXd& d) {
        const double redundant_tighten = 0.;
        double ellipsoid_cost_val;

        Eigen::VectorXd lagrangian_gram_var_vals, verified_gram_var_vals,
            separating_plane_var_vals;
        const bool is_success = internal::FindLagrangianAndSeparatingPlanes(
            *this, alternation_tuples, C, d, lagrangian_gram_vars,
            verified_gram_vars, separating_plane_vars,
            separating_plane_to_lorentz_cone_constraints, t_lower, t_upper,
            verification_option, redundant_tighten, solver_options,
            binary_search_option.verbose, binary_search_option.num_threads,
            separating_plane_to_tuples, &lagrangian_gram_var_vals,
            &verified_gram_var_vals, &separating_plane_var_vals,
            cspace_free_region_solution);
        if (is_success) {
          (cspace_free_region_solution->d) = d;

          FindLargestInscribedEllipsoid(
              (cspace_free_region_solution->C),
              (cspace_free_region_solution->d), t_lower, t_upper,
              binary_search_option.lagrangian_backoff_scale,
              binary_search_option.ellipsoid_volume, solver_options,
              binary_search_option.verbose, &(cspace_free_region_solution->P),
              &(cspace_free_region_solution->q), &ellipsoid_cost_val);
          if (binary_search_option.search_d) {
            // Now fix the Lagrangian and C, and search for d.
            auto prog_polytope = this->ConstructPolytopeProgram(
                alternation_tuples, C_var, d_var, d_minus_Ct,
                lagrangian_gram_var_vals, verified_gram_vars,
                separating_plane_vars,
                separating_plane_to_lorentz_cone_constraints, t_minus_t_lower,
                t_upper_minus_t, verification_option);
            // Calling AddBoundingBoxConstraint(C, C, C_var) for matrix C and
            // C_var might have problem, see Drake issue #16421
            prog_polytope->AddBoundingBoxConstraint(
                Eigen::Map<const Eigen::VectorXd>(C.data(),
                                                  C.rows() * C.cols()),
                Eigen::Map<const Eigen::VectorXd>(C.data(),
                                                  C.rows() * C.cols()),
                Eigen::Map<const VectorX<symbolic::Variable>>(
                    C_var.data(), C.rows() * C.cols()));
            // d <= d_var <= d_max
            prog_polytope->AddBoundingBoxConstraint(d, d_max, d_var);
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
            prog_polytope->AddLinearCost(-Eigen::VectorXd::Ones(d_var.rows()),
                                         0, d_var);
            const auto result_polytope =
                solvers::Solve(*prog_polytope, std::nullopt, solver_options);
            drake::log()->info(fmt::format("search d is successful = {}",
                                           result_polytope.is_success()));
            if (result_polytope.is_success()) {
              (cspace_free_region_solution->d) =
                  result_polytope.GetSolution(d_var);
              (cspace_free_region_solution->separating_planes) =
                  internal::GetSeparatingPlanesSolution(*this, is_plane_active,
                                                        result_polytope);

              FindLargestInscribedEllipsoid(
                  (cspace_free_region_solution->C),
                  (cspace_free_region_solution->d), t_lower, t_upper,
                  binary_search_option.lagrangian_backoff_scale,
                  binary_search_option.ellipsoid_volume, solver_options,
                  binary_search_option.verbose,
                  &(cspace_free_region_solution->P),
                  &(cspace_free_region_solution->q), &ellipsoid_cost_val);
            }
          }
        }
        if (binary_search_option.compute_polytope_volume) {
          drake::log()->info(
              fmt::format("C-space polytope volume {}",
                          CalcCspacePolytopeVolume(C, d, t_lower, t_upper)));
        }
        return is_success;
      };

  if (is_polytope_collision_free(
          d_without_epsilon +
          binary_search_option.epsilon_max *
              Eigen::VectorXd::Ones(d_without_epsilon.rows()))) {
    return;
  }
  if (binary_search_option.check_epsilon_min) {
    if (!is_polytope_collision_free(
            d_without_epsilon +
            binary_search_option.epsilon_min *
                Eigen::VectorXd::Ones(d_without_epsilon.rows()))) {
      throw std::runtime_error(
          fmt::format("binary search: the initial epsilon {} is infeasible",
                      binary_search_option.epsilon_min));
    }
  }
  double eps_max = binary_search_option.epsilon_max;
  double eps_min = binary_search_option.epsilon_min;
  int iter_count = 0;
  while (iter_count < binary_search_option.max_iters) {
    const double eps = (eps_max + eps_min) / 2;
    const Eigen::VectorXd d =
        d_without_epsilon +
        eps * Eigen::VectorXd::Ones(d_without_epsilon.rows());
    const bool is_feasible = is_polytope_collision_free(d);
    if (is_feasible) {
      drake::log()->info(fmt::format("epsilon={} is feasible", eps));
      // Now we need to reset d_without_epsilon. The invariance we want is that
      // C*t<= d_without_epsilon + eps_min * 𝟏 is collision free, while C*t <=
      // d_without_epsilon + eps_max * 𝟏 is not feasible with SOS. We know that
      // d_final + 0 * 𝟏 is collision free. Also (d_without_epsilon + eps_max *
      // 𝟏) is not. So we set d_without_epsilon to d_final, and update eps_min
      // and eps_max accordingly.
      eps_max =
          (d_without_epsilon + eps_max * Eigen::VectorXd::Ones(d.rows(), 1) -
           (cspace_free_region_solution->d))
              .maxCoeff();
      d_without_epsilon = (cspace_free_region_solution->d);
      eps_min = 0;
      drake::log()->info(
          fmt::format("reset eps_min={}, eps_max={}", eps_min, eps_max));
    } else {
      drake::log()->info(fmt::format("epsilon={} is infeasible", eps));
      eps_max = eps;
    }
    iter_count++;
  }
  double ellipsoid_cost_val;

  FindLargestInscribedEllipsoid(
      cspace_free_region_solution->C, cspace_free_region_solution->d, t_lower,
      t_upper, binary_search_option.lagrangian_backoff_scale,
      binary_search_option.ellipsoid_volume, solver_options,
      binary_search_option.verbose, &(cspace_free_region_solution->P),
      &(cspace_free_region_solution->q), &ellipsoid_cost_val);
}

// Get the Lorentz cone constraint for |a|<=1
std::vector<solvers::Binding<solvers::LorentzConeConstraint>>
GetNormLessThan1Constraint(const Vector3<symbolic::Expression>& a_A) {
  Eigen::Matrix<double, 4, 3> A_lorentz = Eigen::Matrix<double, 4, 3>::Zero();
  A_lorentz.bottomRows<3>() = Eigen::Matrix3d::Identity();
  const Eigen::Vector4d b_lorentz(1, 0, 0, 0);
  Vector3<symbolic::Variable> a_var;
  for (int i = 0; i < 3; ++i) {
    DRAKE_DEMAND(symbolic::is_variable(a_A(i)));
    a_var(i) = *a_A(i).GetVariables().begin();
  }
  std::vector<solvers::Binding<solvers::LorentzConeConstraint>>
      lorentz_cone_constraints;
  lorentz_cone_constraints.emplace_back(
      std::make_shared<solvers::LorentzConeConstraint>(A_lorentz, b_lorentz),
      a_var);
  return lorentz_cone_constraints;
}

std::vector<LinkOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const CollisionGeometry* link_geometry,
    const CollisionGeometry* other_side_geometry,
    const RationalForwardKinematics::Pose<symbolic::Polynomial>&
        X_AB_multilinear,
    const drake::Vector3<symbolic::Expression>& a_A,
    const symbolic::Expression& b, PlaneSide plane_side,
    SeparatingPlaneOrder plane_order, double separating_polytope_delta) {
  std::vector<LinkOnPlaneSideRational> rational_fun;
  double separating_delta = 0;
  if (link_geometry->type() == CollisionGeometryType::kPolytope &&
      other_side_geometry->type() == CollisionGeometryType::kPolytope) {
    separating_delta = separating_polytope_delta;
  }

  const symbolic::Monomial monomial_one{};
  // a_A and b are not polynomial of sinθ or cosθ.
  Vector3<symbolic::Polynomial> a_A_poly;
  for (int i = 0; i < 3; ++i) {
    a_A_poly(i) = symbolic::Polynomial({{monomial_one, a_A(i)}});
  }
  const symbolic::Polynomial b_poly({{monomial_one, b}});

  // Compute the rational for a.dot(p_AQ)+b-offset on the positive side, and
  // -offset - a.dot(p_AQ)-b on the negative side, add this rational to
  // rational_fun.
  auto compute_rational =
      [&rational_fun, &X_AB_multilinear, &a_A_poly, &b_poly,
       &rational_forward_kinematics, plane_side, &a_A, &b, link_geometry,
       other_side_geometry, plane_order](
          const Eigen::Vector3d& p_BQ, double offset,
          const std::vector<solvers::Binding<solvers::LorentzConeConstraint>>&
              lorentz_cone_constraints) {
        // Step 1: Compute p_AQ.
        const Vector3<drake::symbolic::Polynomial> p_AQ =
            X_AB_multilinear.p_AB + X_AB_multilinear.R_AB * p_BQ;

        // Step 2: Compute a_A.dot(p_AQ) + b
        const drake::symbolic::Polynomial point_on_hyperplane_side =
            a_A_poly.dot(p_AQ) + b_poly;

        // Step 3: Convert the multilinear polynomial to rational function.
        rational_fun.emplace_back(
            rational_forward_kinematics
                .ConvertMultilinearPolynomialToRationalFunction(
                    plane_side == PlaneSide::kPositive
                        ? point_on_hyperplane_side - offset
                        : -offset - point_on_hyperplane_side),
            link_geometry, X_AB_multilinear.frame_A_index, other_side_geometry,
            a_A, b, plane_side, plane_order, lorentz_cone_constraints);
      };

  switch (link_geometry->type()) {
    case CollisionGeometryType::kPolytope: {
      // TODO(hongkai.dai): cache the map from geometry_id to vertices since
      // getting the vertices might be expensive.

      // We compute the rational for every polytope vertex.
      const Eigen::Matrix3Xd p_BV =
          link_geometry->X_BG() * GetVertices(link_geometry->geometry());
      rational_fun.reserve(p_BV.cols());
      for (int i = 0; i < p_BV.cols(); ++i) {
        compute_rational(p_BV.col(i), separating_delta, {});
      }
      break;
    }
    case CollisionGeometryType::kSphere: {
      // We will generate the rational for
      // aᵀc + b − r (positive side) or -r − b − aᵀc(negative side)
      // where c is the center of the sphere, r is the radius of the sphere.
      // Additionally we will need to add the constraint |a|≤1 as a
      // second-order cone constraint.
      const auto link_sphere =
          dynamic_cast<const geometry::Sphere*>(&link_geometry->geometry());
      rational_fun.reserve(1);
      const double radius = link_sphere->radius();
      DRAKE_DEMAND(plane_order == SeparatingPlaneOrder::kConstant);
      compute_rational(link_geometry->X_BG().translation(), radius,
                       GetNormLessThan1Constraint(a_A));
      break;
    }
    case CollisionGeometryType::kCapsule: {
      // We will generate the rational for
      // aᵀc₁ + b − r, aᵀc₂ + b − r (positive side) or -r − b − aᵀc₁, -r − b −
      // aᵀc₂ (negative side) where c₁, c₂ are the centers of the capsule
      // spheres, r is the radius of the capsule. Additionally we will need to
      // add the constraint |a|≤1 as a second-order cone constraint.
      const auto link_capsule =
          dynamic_cast<const geometry::Capsule*>(&link_geometry->geometry());
      rational_fun.reserve(2);
      Eigen::Matrix<double, 3, 2> p_BC;
      p_BC.col(0) = link_geometry->X_BG() *
                    Eigen::Vector3d(0, 0, -link_capsule->length() / 2);
      p_BC.col(1) = link_geometry->X_BG() *
                    Eigen::Vector3d(0, 0, link_capsule->length() / 2);
      for (int i = 0; i < 2; ++i) {
        DRAKE_DEMAND(plane_order == SeparatingPlaneOrder::kConstant);
        std::vector<solvers::Binding<solvers::LorentzConeConstraint>>
            lorentz_cone_constraints;
        if (i == 0) {
          // We only need to impose the Lorentz cone constraint |a|<=1 for
          // once, no need to do that for both spheres on the capsule.
          lorentz_cone_constraints = GetNormLessThan1Constraint(a_A);
        }
        compute_rational(p_BC.col(i), link_capsule->radius(),
                         lorentz_cone_constraints);
      }
      break;
    }
    default: {
      throw std::runtime_error("Not implemented yet");
    }
  }
  return rational_fun;
}

bool IsGeometryPairCollisionIgnored(
    const SortedPair<geometry::GeometryId>& geometry_pair,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  return filtered_collision_pairs.count(geometry_pair) > 0;
}

bool IsGeometryPairCollisionIgnored(
    geometry::GeometryId id1, geometry::GeometryId id2,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  return IsGeometryPairCollisionIgnored(
      drake::SortedPair<geometry::GeometryId>(id1, id2),
      filtered_collision_pairs);
}

void ComputeBoundsOnT(const Eigen::Ref<const Eigen::VectorXd>& q_star,
                      const Eigen::Ref<const Eigen::VectorXd>& q_lower,
                      const Eigen::Ref<const Eigen::VectorXd>& q_upper,
                      Eigen::VectorXd* t_lower, Eigen::VectorXd* t_upper) {
  DRAKE_DEMAND((q_upper.array() >= q_lower.array()).all());
  // Currently I require that q_upper - q_star < pi and q_star - q_lower >
  // -pi.
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
  // Add the constraint |cᵢᵀP|₂ ≤ dᵢ − cᵢᵀq − δᵢ as a Lorentz cone
  // constraint, namely [dᵢ − cᵢᵀq − δᵢ, cᵢᵀP] is in the Lorentz cone. [dᵢ
  // − cᵢᵀq − δᵢ, cᵢᵀP] = A_lorentz1 * [cᵢᵀ, dᵢ, δᵢ] + b_lorentz1
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
  // Add the constraint |cᵢᵀ|₂ ≤ 1 as a Lorentz cone constraint that [1,
  // cᵢᵀ] is in the Lorentz cone. [1, cᵢᵀ] = A_lorentz2 * cᵢᵀ + b_lorentz2
  Eigen::MatrixXd A_lorentz2 = Eigen::MatrixXd::Zero(1 + C.cols(), C.cols());
  A_lorentz2.bottomRows(C.cols()) =
      Eigen::MatrixXd::Identity(C.cols(), C.cols());
  Eigen::VectorXd b_lorentz2 = Eigen::VectorXd::Zero(1 + C.cols());
  b_lorentz2(0) = 1;
  for (int i = 0; i < C.rows(); ++i) {
    prog->AddLorentzConeConstraint(A_lorentz2, b_lorentz2, C.row(i));
  }
}

std::map<BodyIndex, std::vector<std::unique_ptr<CollisionGeometry>>>
GetCollisionGeometries(const systems::Diagram<double>& diagram,
                       const MultibodyPlant<double>* plant,
                       const geometry::SceneGraph<double>* scene_graph) {
  std::map<BodyIndex, std::vector<std::unique_ptr<CollisionGeometry>>> ret;
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
        const auto& shape = inspector.GetShape(geometry_id);
        // Get the pose X_BG;
        const math::RigidTransformd X_WB =
            query_object.GetPoseInWorld(*frame_id);
        const math::RigidTransformd& X_WG =
            query_object.GetPoseInWorld(geometry_id);
        const math::RigidTransformd X_BG = X_WB.InvertAndCompose(X_WG);
        std::unique_ptr<CollisionGeometry> collision_geometry{nullptr};
        if (dynamic_cast<const geometry::Convex*>(&shape) ||
            dynamic_cast<const geometry::Box*>(&shape)) {
          // For a convex mesh or a box, construct a VPolytope.
          geometry::optimization::VPolytope v_polytope(
              query_object, geometry_id, frame_id.value());
          collision_geometry = std::make_unique<CollisionGeometry>(
              CollisionGeometryType::kPolytope, &shape, body_index, geometry_id,
              X_BG);
        } else if (dynamic_cast<const geometry::Sphere*>(&shape)) {
          collision_geometry = std::make_unique<CollisionGeometry>(
              CollisionGeometryType::kSphere, &shape, body_index, geometry_id,
              X_BG);
        } else if (dynamic_cast<const geometry::Capsule*>(&shape)) {
          collision_geometry = std::make_unique<CollisionGeometry>(
              CollisionGeometryType::kCapsule, &shape, body_index, geometry_id,
              X_BG);
        } else {
          throw std::runtime_error(
              "GetCollisionGeometries: unsupported shape type.");
        }
        DRAKE_DEMAND(collision_geometry.get() != nullptr);

        auto it = ret.find(body_index);
        if (it == ret.end()) {
          std::vector<std::unique_ptr<CollisionGeometry>> body_geometries;
          body_geometries.push_back(std::move(collision_geometry));
          ret.emplace_hint(it, body_index, std::move(body_geometries));
        } else {
          it->second.push_back(std::move(collision_geometry));
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
  // We aggregate the constraint {C*t<=d, t_lower <= t <= t_upper} as C̅t ≤
  // d̅
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
  // According to duality theory, the polytope C*t<=d contains the
  // polytope C_bar*t <= d_bar if and only if there exists variable λᵢ≥ 0,
  // dᵢ−λᵢᵀd̅≥0, C̅ᵀλᵢ=cᵢ for every row of C*t<=d.
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
  // TODO(hongkai.dai) call v_poly.CalcVolume() when Drake PR 16409 is
  // merged.
  orgQhull::Qhull qhull;
  try {
    qhull.runQhull("", nt, v_poly.vertices().cols(), v_poly.vertices().data(),
                   "");
    if (qhull.qhullStatus() != 0) {
      throw std::runtime_error(
          fmt::format("Qhull terminated with status {} and message:\n{}",
                      qhull.qhullStatus(), qhull.qhullMessage()));
    }
  } catch (const orgQhull::QhullError& e) {
    drake::log()->warn("Qhull::runQhull fails.");
    return NAN;
  }
  double volume{NAN};
  try {
    volume = qhull.volume();
  } catch (const orgQhull::QhullError& e) {
    drake::log()->warn("Qhull fails to compute the volume");
  }
  return volume;
}

void WriteCspacePolytopeToFile(
    const CspaceFreeRegionSolution& solution,
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraphInspector<double>& inspector,

    const std::string& file_name, int precision) {
  std::ofstream myfile;
  myfile.open(file_name, std::ios::out);
  if (myfile.is_open()) {
    myfile << fmt::format("{} {}\n", solution.C.rows(), solution.C.cols());
    for (int i = 0; i < solution.C.rows(); ++i) {
      for (int j = 0; j < solution.C.cols(); ++j) {
        myfile << std::setprecision(precision) << solution.C(i, j) << " ";
      }
      myfile << "\n";
    }
    myfile << "\n";
    for (int i = 0; i < solution.d.rows(); ++i) {
      myfile << std::setprecision(precision) << solution.d(i) << " ";
    }
    myfile << "\n";
    // Write separating planes.
    myfile << solution.separating_planes.size() << "\n";
    for (const auto& plane : solution.separating_planes) {
      // positive side.
      // model instance name
      myfile << plant.GetModelInstanceName(
                    plant.get_body(plane.positive_side_geometry->body_index())
                        .model_instance())
             << "\n";
      // body name
      myfile
          << plant.get_body(plane.positive_side_geometry->body_index()).name()
          << "\n";
      // geometry name
      myfile << inspector.GetName(plane.positive_side_geometry->id()) << "\n";
      // negative side
      // model instance name
      myfile << plant.GetModelInstanceName(
                    plant.get_body(plane.negative_side_geometry->body_index())
                        .model_instance())
             << "\n";
      // body name
      myfile
          << plant.get_body(plane.negative_side_geometry->body_index()).name()
          << "\n";
      // geometry name
      myfile << inspector.GetName(plane.negative_side_geometry->id()) << "\n";

      // Experessed
      // model instance
      myfile << plant.GetModelInstanceName(
                    plant.get_body(plane.expressed_link).model_instance())
             << "\n";
      // body name
      myfile << plant.get_body(plane.expressed_link).name() << "\n";
      myfile << plane.decision_variables.rows() << "\n";
      for (int i = 0; i < plane.decision_variables.rows(); ++i) {
        myfile << std::setprecision(precision) << plane.decision_variables(i)
               << " ";
      }
      myfile << "\n";
    }
    myfile.close();
  }
}

void ReadCspacePolytopeFromFile(
    const std::string& filename, const MultibodyPlant<double>& plant,
    const geometry::SceneGraphInspector<double>& inspector, Eigen::MatrixXd* C,
    Eigen::VectorXd* d,
    std::unordered_map<SortedPair<geometry::GeometryId>,
                       std::pair<BodyIndex, Eigen::VectorXd>>*
        separating_planes) {
  std::ifstream infile;
  infile.open(filename, std::ios::in);
  std::string line;
  if (infile.is_open()) {
    // Read the size of C, d
    std::getline(infile, line);
    std::istringstream ss(line);
    std::string word;
    ss >> word;
    const int C_rows = std::stoi(word);
    ss >> word;
    const int C_cols = std::stoi(word);
    C->resize(C_rows, C_cols);
    for (int i = 0; i < C_rows; ++i) {
      std::getline(infile, line);
      ss = std::istringstream(line);
      for (int j = 0; j < C_cols; ++j) {
        ss >> word;
        (*C)(i, j) = std::stod(word);
      }
    }
    std::getline(infile, line);
    // get d.
    std::getline(infile, line);
    d->resize(C_rows);
    ss = std::istringstream(line);
    for (int i = 0; i < C_rows; ++i) {
      ss >> word;
      (*d)(i) = std::stod(word);
    }
    // get separating plane.
    std::getline(infile, line);
    ss = std::istringstream(line);
    ss >> word;
    const int num_separating_planes = std::stoi(word);
    for (int i = 0; i < num_separating_planes; ++i) {
      // positive side.
      // model instance
      std::getline(infile, line);
      ss = std::istringstream(line);
      ss >> word;
      const ModelInstanceIndex positive_model =
          plant.GetModelInstanceByName(word);
      // body
      std::getline(infile, line);
      ss = std::istringstream(line);
      ss >> word;
      const geometry::FrameId positive_frame = plant.GetBodyFrameIdOrThrow(
          plant.GetBodyByName(word, positive_model).index());
      // geometry
      std::getline(infile, line);
      ss = std::istringstream(line);
      ss >> word;
      const geometry::GeometryId positive_geo_id =
          inspector.GetGeometryIdByName(positive_frame,
                                        geometry::Role::kProximity, word);

      // negative side
      // model instance
      std::getline(infile, line);
      ss = std::istringstream(line);
      ss >> word;
      const ModelInstanceIndex negative_model =
          plant.GetModelInstanceByName(word);
      // body
      std::getline(infile, line);
      ss = std::istringstream(line);
      ss >> word;
      const geometry::FrameId negative_frame = plant.GetBodyFrameIdOrThrow(
          plant.GetBodyByName(word, negative_model).index());
      // geometry
      std::getline(infile, line);
      ss = std::istringstream(line);
      ss >> word;
      const geometry::GeometryId negative_geo_id =
          inspector.GetGeometryIdByName(negative_frame,
                                        geometry::Role::kProximity, word);

      // expressed
      // expressed model instance
      std::getline(infile, line);
      ss = std::istringstream(line);
      ss >> word;
      const ModelInstanceIndex expressed_model =
          plant.GetModelInstanceByName(word);
      // expressed link
      std::getline(infile, line);
      ss = std::istringstream(line);
      ss >> word;
      const BodyIndex expressed_link =
          plant.GetBodyByName(word, expressed_model).index();

      // plane decision variable values.
      std::getline(infile, line);
      ss = std::istringstream(line);
      ss >> word;
      const int num_var = std::stoi(word);
      Eigen::VectorXd plane_vars(num_var);
      std::getline(infile, line);
      ss = std::istringstream(line);
      for (int j = 0; j < num_var; ++j) {
        ss >> word;
        plane_vars(j) = std::stod(word);
      }
      separating_planes->emplace(SortedPair(positive_geo_id, negative_geo_id),
                                 std::make_pair(expressed_link, plane_vars));
    }
  } else {
    throw std::runtime_error(
        fmt::format("Cannot open file {} for c-space polytope.", filename));
  }
  infile.close();
}

void WriteCspacePolytopeToFile(const Eigen::Ref<const Eigen::MatrixXd>& C,
                               const Eigen::Ref<const Eigen::VectorXd>& d,
                               const Eigen::Ref<const Eigen::VectorXd>& t_lower,
                               const Eigen::Ref<const Eigen::VectorXd>& t_upper,
                               const std::string& file_name, int precision) {
  std::ofstream myfile;
  myfile.open(file_name, std::ios::out);
  if (myfile.is_open()) {
    myfile << fmt::format("{} {}\n", C.rows(), C.cols());
    for (int i = 0; i < C.rows(); ++i) {
      for (int j = 0; j < C.cols(); ++j) {
        myfile << std::setprecision(precision) << C(i, j) << " ";
      }
      myfile << "\n";
    }
    myfile << "\n";
    for (int i = 0; i < d.rows(); ++i) {
      myfile << std::setprecision(precision) << d(i) << " ";
    }
    myfile << "\n";
    for (int i = 0; i < t_lower.rows(); ++i) {
      myfile << std::setprecision(precision) << t_lower(i) << " ";
    }
    myfile << "\n";
    for (int i = 0; i < t_upper.rows(); ++i) {
      myfile << std::setprecision(precision) << t_upper(i) << " ";
    }
    myfile.close();
  }
}

void ReadCspacePolytopeFromFile(const std::string& filename, Eigen::MatrixXd* C,
                                Eigen::VectorXd* d, Eigen::VectorXd* t_lower,
                                Eigen::VectorXd* t_upper) {
  std::ifstream infile;
  infile.open(filename, std::ios::in);
  std::string line;
  if (infile.is_open()) {
    // Read the size of C, d
    std::getline(infile, line);
    std::istringstream ss(line);
    std::string word;
    ss >> word;
    const int C_rows = std::stoi(word);
    ss >> word;
    const int C_cols = std::stoi(word);
    C->resize(C_rows, C_cols);
    for (int i = 0; i < C_rows; ++i) {
      std::getline(infile, line);
      ss = std::istringstream(line);
      for (int j = 0; j < C_cols; ++j) {
        ss >> word;
        (*C)(i, j) = std::stod(word);
      }
    }
    std::getline(infile, line);
    // get d.
    std::getline(infile, line);
    d->resize(C_rows);
    ss = std::istringstream(line);
    for (int i = 0; i < C_rows; ++i) {
      ss >> word;
      (*d)(i) = std::stod(word);
    }
    // get t_lower;
    std::getline(infile, line);
    t_lower->resize(C_cols);
    ss = std::istringstream(line);
    for (int i = 0; i < C_cols; ++i) {
      ss >> word;
      (*t_lower)(i) = std::stod(word);
    }
    // get t_upper;
    std::getline(infile, line);
    t_upper->resize(C_cols);
    ss = std::istringstream(line);
    for (int i = 0; i < C_cols; ++i) {
      ss >> word;
      (*t_upper)(i) = std::stod(word);
    }
  } else {
    throw std::runtime_error(
        fmt::format("Cannot open file {} for c-space polytope.", filename));
  }
  infile.close();
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
