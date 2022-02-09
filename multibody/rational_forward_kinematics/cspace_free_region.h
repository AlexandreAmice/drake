#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/multibody/rational_forward_kinematics/convex_geometry.h"
#include "drake/multibody/rational_forward_kinematics/plane_side.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"

/**
 * This file is largely the same as configuration_space_collision_free_region.h.
 * The major differences are
 * 1. The separating hyperplane is parameterized as aᵀx + b ≥ 1 and aᵀx+b ≤ −1
 * 2. We first focus on the generic polytopic region C*t<=d in the configuration
 * space (we will add the special case for axis-aligned bounding box region
 * t_lower <= t <= t_upper later).
 */

namespace drake {
namespace multibody {
/* The separating plane aᵀx + b ≥ 1, aᵀx+b ≤ −1 has parameters a and b. These
 * parameters can be a constant of affine function of t.
 */
enum class SeparatingPlaneOrder {
  kConstant,
  kAffine,
};

/**
 * One polytope is on the "positive" side of the separating plane, namely {x|
 * aᵀx + b ≥ 1}, and the other polytope is on the "negative" side of the
 * separating plane, namely {x|aᵀx+b ≤ −1}.
 */
struct SeparatingPlane {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SeparatingPlane)

  SeparatingPlane() = default;

  SeparatingPlane(
      drake::Vector3<symbolic::Expression> m_a, symbolic::Expression m_b,
      const ConvexPolytope* m_positive_side_polytope,
      const ConvexPolytope* m_negative_side_polytope,
      multibody::BodyIndex m_expressed_link, SeparatingPlaneOrder m_order,
      const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>>&
          m_decision_variables)
      : a{std::move(m_a)},
        b{std::move(m_b)},
        positive_side_polytope{m_positive_side_polytope},
        negative_side_polytope{m_negative_side_polytope},
        expressed_link{std::move(m_expressed_link)},
        order{m_order},
        decision_variables{m_decision_variables} {}

  Vector3<symbolic::Expression> a;
  symbolic::Expression b;
  const ConvexPolytope* positive_side_polytope;
  const ConvexPolytope* negative_side_polytope;
  multibody::BodyIndex expressed_link;
  SeparatingPlaneOrder order;
  VectorX<symbolic::Variable> decision_variables;
};


/**
 * We need to verify that C * t <= d implies p(t) >= 0, where p(t) is the
 * numerator of the rational function aᵀx + b - 1 or -1 - aᵀx-b. Namely we need
 * to verify the non-negativity of the lagrangian polynomial l(t), together with
 * p(t) - l(t)ᵀ(d - C * t). We can choose the type of the non-negative
 * polynomials (sos, dsos, sdsos).
 */
struct VerificationOption {
  solvers::MathematicalProgram::NonnegativePolynomial link_polynomial_type;
  solvers::MathematicalProgram::NonnegativePolynomial lagrangian_type;
};

/**
 * A verified region is a polytope C*t <= d along with the multiplier polynomials acting as the certificate,
 * the planes for all the collision pairs, and the inscribed ellipsoid approximating the volume of the polytope
 */
struct CspaceFreeRegionSolution{
   public:
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(CspaceFreeRegionSolution)
    CspaceFreeRegionSolution() = default;

   CspaceFreeRegionSolution(
     Eigen::MatrixXd m_C,
     Eigen::MatrixXd m_d,
     Eigen::MatrixXd m_P,
     Eigen::MatrixXd m_q,
//     const std::vector<VectorX<symbolic::Polynomial>> m_polytope_lagrangians,
//     const std::vector<VectorX<symbolic::Polynomial>> m_t_lower_lagrangians,
//     const std::vector<VectorX<symbolic::Polynomial>> m_t_upper_lagrangians,
//     const std::vector<symbolic::Polynomial> m_verified_polynomials,
     std::vector<SeparatingPlane> m_separating_planes)
      : C{std::move(m_C)},
        d{std::move(m_d)},
        P{std::move(m_P)},
        q{std::move(m_q)},
//        polytope_lagrangians{m_polytope_lagrangians},
//        t_lower_lagrangians{m_t_lower_lagrangians},
//        t_upper_lagrangians{m_t_upper_lagrangians},
//        verified_polynomials{m_verified_polynomials},
        separating_planes{std::move(m_separating_planes)} {}
//
   CspaceFreeRegionSolution(
     Eigen::MatrixXd m_C,
     Eigen::MatrixXd m_d)
      : C{m_C},
        d{m_d}
//        polytope_lagrangians{num_pairs},
//        t_lower_lagrangians{num_pairs},
//        t_upper_lagrangians{num_pairs},
//        verified_polynomials{num_pairs},
//        separating_planes{num_pairs}
        { }

   // values defining Hpolyhedron Ct <= d
   Eigen::MatrixXd C;
   Eigen::VectorXd d;

   // values defining Inscribed ellipsoid {t | t = Ps + q , norm(s) <= 1}
   Eigen::MatrixXd P;
   Eigen::VectorXd q;


   // TODO (Alex.Amice) add these polynomials back in one I figure out how to extract them
//    // multipliers for C*t <= d
//    std::vector<VectorX<symbolic::Polynomial>> polytope_lagrangians;
//    // multiplier for t >= t_lower
//    std::vector<VectorX<symbolic::Polynomial>> t_lower_lagrangians;
//    // multiplier for t <= t_upper
//    std::vector<VectorX<symbolic::Polynomial>> t_upper_lagrangians;
//    // verified_polynomial[i] is p(t) - l_polytope(t)ᵀ(d - C*t) -
//    // l_lower(t)ᵀ(t-t_lower) - l_upper(t)ᵀ(t_upper-t)
//    std::vector<symbolic::Polynomial> verified_polynomials;

    // Separating hyperplanes that are the certificate
    std::vector<SeparatingPlane> separating_planes;
 };

/**
 * The rational function representing that a link vertex V is on the desired
 * side of the plane. If the link is on the positive side of the plane, then the
 * rational is aᵀx + b - 1, otherwise it is -1 - aᵀx - b
 */
struct LinkVertexOnPlaneSideRational {
  LinkVertexOnPlaneSideRational(
      symbolic::RationalFunction m_rational,
      const ConvexPolytope* m_link_polytope,
      multibody::BodyIndex m_expressed_body_index,
      const ConvexPolytope* m_other_side_link_polytope,
      Vector3<symbolic::Expression> m_a_A, symbolic::Expression m_b,
      PlaneSide m_plane_side, SeparatingPlaneOrder m_plane_order)
      : rational{std::move(m_rational)},
        link_polytope{m_link_polytope},
        expressed_body_index{m_expressed_body_index},
        other_side_link_polytope{m_other_side_link_polytope},
        a_A{std::move(m_a_A)},
        b{std::move(m_b)},
        plane_side{m_plane_side},
        plane_order{m_plane_order} {}
  const symbolic::RationalFunction rational;
  const ConvexPolytope* const link_polytope;
  const multibody::BodyIndex expressed_body_index;
  const ConvexPolytope* const other_side_link_polytope;
  const Vector3<symbolic::Expression> a_A;
  const symbolic::Expression b;
  const PlaneSide plane_side;
  const SeparatingPlaneOrder plane_order;
};

enum class CspaceRegionType { kGenericPolytope, kAxisAlignedBoundingBox };

/**
 * When we maximize the inscribed ellipsoid, we can measure the size of the
 * ellipsoid by either the logarithm of its volume, or the n'th root of its
 * volume. The logarithm volume would introduce exponential cone constraints,
 * while the n'th root of the volume would introduce second order cone
 * constraints.
 */
enum class EllipsoidVolume { kLog, kNthRoot };



/**
 * This class tries to find a large convex set in the configuration space, such
 * that this whole convex set is collision free. We assume that the obstacles
 * are unions of polytopes in the workspace, and the robot link poses
 * (position/orientation) can be written as rational functions of some
 * variables. Such robot can have only revolute (or prismatic joint). We also
 * suppose that the each link of the robot is represented as a union of
 * polytopes. We will find the convex collision free set in the configuration
 * space through convex optimization.
 */
class CspaceFreeRegion {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CspaceFreeRegion)

  using FilteredCollisionPairs =
      std::unordered_set<drake::SortedPair<ConvexGeometry::Id>>;

  CspaceFreeRegion(const systems::Diagram<double>& diagram,
                   const multibody::MultibodyPlant<double>* plant,
                   const geometry::SceneGraph<double>* scene_graph,
                   SeparatingPlaneOrder plane_order,
                   CspaceRegionType cspace_region_type);

  CspaceFreeRegion(const multibody::MultibodyPlant<double>& plant,
                   const std::vector<const ConvexPolytope*>& link_polytopes,
                   const std::vector<const ConvexPolytope*>& obstacles,
                   SeparatingPlaneOrder plane_order,
                   CspaceRegionType cspace_region_type);

  /** separating_planes()[map_polytopes_to_separating_planes.at(geometry1_id,
   * geometry2_id)] is the separating plane that separates geometry1 and
   * geometry 2.
   */
  const std::unordered_map<SortedPair<ConvexGeometry::Id>, int>&
  map_polytopes_to_separating_planes() const {
    return map_polytopes_to_separating_planes_;
  }

  /**
   * Generate all the rational functions in the form aᵀx + b -1 or -1-aᵀx-b
   * whose non-negativity implies that the separating plane aᵀx + b =0 separates
   * a pair of polytopes.
   * This function loops over all pair of polytopes between a link and an
   * obstacle that are not in filtered_collision_pair.
   */
  // TODO(hongkai.dai): also consider the self-collision pairs.
  std::vector<LinkVertexOnPlaneSideRational>
  GenerateLinkOnOneSideOfPlaneRationals(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs)
      const;

  /**
   * This struct is the return type of ConstructProgramForCspacePolytope, to
   * verify the C-space region C * t <= d is collision free.
   */
  struct CspacePolytopeProgramReturn {
    explicit CspacePolytopeProgramReturn(size_t rationals_size)
        : prog{new solvers::MathematicalProgram()},
          polytope_lagrangians{rationals_size},
          t_lower_lagrangians{rationals_size},
          t_upper_lagrangians{rationals_size},
          verified_polynomials{rationals_size} {}

    std::unique_ptr<solvers::MathematicalProgram> prog;
    // polytope_lagrangians has size rationals.size(), namely it is the number
    // of (link_polytope, obstacle_polytope) pairs. lagrangians[i] has size
    // C.rows()
    std::vector<VectorX<symbolic::Polynomial>> polytope_lagrangians;
    // t_lower_lagrangians[i][j] is the lagrangian for t(j) >= t_lower(j) to
    // verify rationals[i]>= 0.
    std::vector<VectorX<symbolic::Polynomial>> t_lower_lagrangians;
    // t_upper_lagrangians[i][j] is the lagrangian for t(j) <= t_lower(j) to
    // verify rationals[i]>= 0.
    std::vector<VectorX<symbolic::Polynomial>> t_upper_lagrangians;
    // verified_polynomial[i] is p(t) - l_polytope(t)ᵀ(d - C*t) -
    // l_lower(t)ᵀ(t-t_lower) - l_upper(t)ᵀ(t_upper-t)
    std::vector<symbolic::Polynomial> verified_polynomials;
  };

  /* @note I strongly recommend NOT to use this function. I created this
   * function for fast prototyping. It is very slow when constructing the
   * program (as it incurs a lot of dynamic memory allocation.
   */
  CspacePolytopeProgramReturn ConstructProgramForCspacePolytope(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const std::vector<LinkVertexOnPlaneSideRational>& rationals,
      const Eigen::Ref<const Eigen::MatrixXd>& C,
      const Eigen::Ref<const Eigen::VectorXd>& d,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const VerificationOption& verification_option = {}) const;

  bool IsPostureInCollision(const systems::Context<double>& context) const;

  /** Each tuple corresponds to one rational aᵀx + b - 1 or -1 - aᵀx - b.
   * This tuple should be used with GenerateTuplesForBilinearAlternation. To
   * save computation time, this class minimizes using dynamic memory
   * allocation.
   */
  struct CspacePolytopeTuple {
    CspacePolytopeTuple(symbolic::Polynomial m_rational_numerator,
                        std::vector<int> m_polytope_lagrangian_gram_lower_start,
                        std::vector<int> m_t_lower_lagrangian_gram_lower_start,
                        std::vector<int> m_t_upper_lagrangian_gram_lower_start,
                        int m_verified_polynomial_gram_lower_start,
                        VectorX<symbolic::Monomial> m_monomial_basis)
        : rational_numerator{std::move(m_rational_numerator)},
          polytope_lagrangian_gram_lower_start{
              std::move(m_polytope_lagrangian_gram_lower_start)},
          t_lower_lagrangian_gram_lower_start{
              std::move(m_t_lower_lagrangian_gram_lower_start)},
          t_upper_lagrangian_gram_lower_start{
              std::move(m_t_upper_lagrangian_gram_lower_start)},
          verified_polynomial_gram_lower_start{
              m_verified_polynomial_gram_lower_start},
          monomial_basis{std::move(m_monomial_basis)} {}

    // This is the numerator of the rational
    // aᵀx+b-1 or -1-aᵀx-b
    symbolic::Polynomial rational_numerator;
    // lagrangian_gram_var.segment(polytope_lagrangian_gram_lower_start,
    // n(n+1)/2) is the lower diagonal entries of the gram matrix in
    // l_polytope(t)(i). l_polytope(t) will multiply with (d-C*t).
    std::vector<int> polytope_lagrangian_gram_lower_start;
    // lagrangian_gram_vars.segment(t_lower_lagrangian_gram_start_index[i],
    // n(n+1)/2) is the lower diagonal entries of the gram matrix in
    // l_lower(t)(i). l_lower(t) will multiply with (t - t_lower).
    std::vector<int> t_lower_lagrangian_gram_lower_start;
    // lagrangian_gram_vars.segment(t_upper_lagrangian_gram_start_index[i],
    // n(n+1)/2) is the lower diagonal entries of the gram matrix in
    // l_upper(t)(i). l_upper(t) will multiply with (t_upper - t).
    std::vector<int> t_upper_lagrangian_gram_lower_start;
    // Verified polynomial is p(t) - l_polytope(t)ᵀ(d-C*t) -
    // l_lower(t)ᵀ(t-t_lower) - l_upper(t)ᵀ(t_upper-t)
    // verified_gram_vars.segment(verified_polynomial_gram_lower_start[i],
    // n(n+1)/2) is the lower part of the gram matrix of the verified
    // polynomial.
    int verified_polynomial_gram_lower_start;
    VectorX<symbolic::Monomial> monomial_basis;
  };

  /** Generate the tuples for bilinear alternation.
   * @param[out] d_minus_Ct Both C and d are decision variables in d_minus_Ct
   * (instead of fixed double values).
   * @param[out] t_lower Lower bounds on t computed from joint limits.
   * @param[out] t_upper Upper bounds on t computed from joint limits.
   * @param[out] t_minus_t_lower t - t_lower
   * @param[out] t_upper_minus_t t_upper - t
   * @param[out] lagrangian_gram_vars All of the variables in the Gram matrices
   * for all Lagrangian polynomials.
   * @param[out] verified_gram_vars All of the variables in the verified
   * polynomial p(t) - l_polytope(t)ᵀ(d-C*t) - l_lower(t)ᵀ(t-t_lower) -
   * l_upper(t)ᵀ(t_upper-t) for all of the rationals.
   * @param[out] separating_plane_vars All of the variables in the separating
   * plane aᵀx + b = 0.
   * @param[out] separating_plane_to_tuples alternation_tuples can be grouped
   * based on the separating planes. separating_plane_to_tuples[i] are the
   * indices in alternation_tuples such that these tuples are all for
   * this->separating_planes()[i].
   */
  void GenerateTuplesForBilinearAlternation(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs, int C_rows,
      std::vector<CspacePolytopeTuple>* alternation_tuples,
      VectorX<symbolic::Polynomial>* d_minus_Ct, Eigen::VectorXd* t_lower,
      Eigen::VectorXd* t_upper, VectorX<symbolic::Polynomial>* t_minus_t_lower,
      VectorX<symbolic::Polynomial>* t_upper_minus_t,
      MatrixX<symbolic::Variable>* C, VectorX<symbolic::Variable>* d,
      VectorX<symbolic::Variable>* lagrangian_gram_vars,
      VectorX<symbolic::Variable>* verified_gram_vars,
      VectorX<symbolic::Variable>* separating_plane_vars,
      std::vector<std::vector<int>>* separating_plane_to_tuples) const;

  /**
   * Given the C-space free region candidate C*t<=d,
   * construct an optimization program with the constraint
   * p(t) - l_polytope(t).dot(d - C*t) - l_lower(t).dot(t - t_lower) -
   * l_upper(t).dot(t_upper-t) >= 0
   * l_polytope(t)>=0, l_lower(t)>=0, l_upper(t)>=0
   * |cᵢᵀP|₂ ≤ dᵢ−cᵢᵀq The unknowns are the separating
   * plane parameters (a, b), the lagrangian multipliers l_polytope(t),
   * l_lower(t), l_upper(t), the inscribed ellipsoid parameter P, q
   * @param alternation_tuples computed from
   * GenerateTuplesForBilinearAlternation.
   * @param lagrangian_gram_vars computed from
   * GenerateTuplesForBilinearAlternation.
   * @param verified_gram_vars computed from
   * GenerateTuplesForBilinearAlternation.
   * @param separating_plane_vars computed from
   * GenerateTuplesForBilinearAlternation.
   * @param t_lower The lower bounds of t computed from joint limits.
   * @param t_upper The upper bounds of t computed from joint limits.
   * @param redundant_tighten. We aggregate the constraint {C*t<=d, t_lower <= t
   * <= t_upper} as C̅t ≤ d̅. A row of C̅t ≤ d̅is regarded as redundant, if the C̅ᵢt
   * ≤ d̅ᵢ − δ is implied by the rest of the constraint, where
   * δ=redundant_tighten. If redundant_tighten=std::nullopt, then we don't try
   * to identify the redundant constraints.
   * @param[out] P The inscribed ellipsoid is parameterized as {Py+q | |y|₂ ≤
   * 1}. Set P=nullptr if you don't want the inscribed ellipsoid.
   * @param[out] q The inscribed ellipsoid is parameterized as {Py+q | |y|₂ ≤
   * 1}. Set q=nullptr if you don't want the inscribed ellipsoid.
   * @note The constructed program doesn't have a cost yet.
   */
  std::unique_ptr<solvers::MathematicalProgram> ConstructLagrangianProgram(
      const std::vector<CspacePolytopeTuple>& alternation_tuples,
      const Eigen::Ref<const Eigen::MatrixXd>& C,
      const Eigen::Ref<const Eigen::VectorXd>& d,
      const VectorX<symbolic::Variable>& lagrangian_gram_vars,
      const VectorX<symbolic::Variable>& verified_gram_vars,
      const VectorX<symbolic::Variable>& separating_plane_vars,
      const Eigen::Ref<const Eigen::VectorXd>& t_lower,
      const Eigen::Ref<const Eigen::VectorXd>& t_upper,
      const VerificationOption& option, std::optional<double> redundant_tighten,
      MatrixX<symbolic::Variable>* P, VectorX<symbolic::Variable>* q) const;

  /**
   * Given lagrangian polynomials, construct an optimization program to search
   * for the separating plane, the C-space polytope C*t<=d.
   * Mathematically the optimization program is
   * p(t) - l_polytope(t).dot(d - C*t) - l_lower(t).dot(t - t_lower) -
   * l_upper(t).dot(t_upper-t) >= 0
   * @note The constructed program doesn't have a cost yet.
   * @param alternation_tuples Returned from
   * GenerateTuplesForBilinearAlternation.
   * @parm C Returned from GenerateTuplesForBilinearAlternation.
   * @parm d Returned from GenerateTuplesForBilinearAlternation.
   * @parm d_minus_Ct Returned from GenerateTuplesForBilinearAlternation. d - C
   * * t.
   * @param lagrangian_gram_var_vals The value for all the lagrangian gram
   * variables.
   * @param verified_gram_vars Returned from
   * GenerateTuplesForBilinearAlternation.
   * @param separating_plane_vars Returned from
   * GenerateTuplesForBilinearAlternation.
   * @param t_minus_t_lower t - t_lower
   * @param t_upper_minus_t t_upper - t
   */
  std::unique_ptr<solvers::MathematicalProgram> ConstructPolytopeProgram(
      const std::vector<CspacePolytopeTuple>& alternation_tuples,
      const MatrixX<symbolic::Variable>& C,
      const VectorX<symbolic::Variable>& d,
      const VectorX<symbolic::Polynomial>& d_minus_Ct,
      const Eigen::VectorXd& lagrangian_gram_var_vals,
      const VectorX<symbolic::Variable>& verified_gram_vars,
      const VectorX<symbolic::Variable>& separating_plane_vars,
      const VectorX<symbolic::Polynomial>& t_minus_t_lower,
      const VectorX<symbolic::Polynomial>& t_upper_minus_t,
      const VerificationOption& option) const;

  struct BilinearAlternationOption {
    /** Number of iterations. One lagrangian step + one polytope step is counted
     * as one iteration (namely we solve two SOS programs in one iteration).
     */
    int max_iters{10};
    /** When the increase of the lagrangian step (log(det(P)) which measures the
     * inscribed ellipsoid volume) is smaller than this tolerance, stop the
     * alternation.
     */
    double convergence_tol{0.001};
    /** Backoff the optimization program to search for a strictly feasible
     * solution not on the boundary of the feasible region. backoff_scale = 0
     * means no backoff, backoff_scale should be a non-negative number.
     * lagrangian_backoff_scale is for the "Lagrangian step", when we search for
     * the Lagrangian and the inscribed ellipsoid given the polytopic region
     * C*t<=d. polytope_backoff_scale is for the "polytope step", when we search
     * for the polytopic region C*t<=d given the Lagrangian and the inscribed
     * ellipsoid.
     */
    double lagrangian_backoff_scale{0.};
    double polytope_backoff_scale{0.};
    int verbose{true};
    // How much tighten we use to determine if a row in {C*t<=d, t_lower <= t <=
    // t_upper} is redundant.
    std::optional<double> redundant_tighten{std::nullopt};
    // Whether to compute and print the volume of the polytope {C*t<=d,
    // t_lower<= t <= t_upper} each time we search for the polytope.
    bool compute_polytope_volume{false};
    // The objective function used in maximizing the volume of the inscribed
    // ellipsoid.
    EllipsoidVolume ellipsoid_volume{EllipsoidVolume::kNthRoot};
    // If set to an integer, then solve the Lagrangian program through many
    // small SOS with the specified number of threads (if num_thread <= 0 then
    // we launch as many threads as possible).. Each program for one pair of
    // collision geometries. std::nullopt means to solve a single big SOS with
    // all the separating planes simultaneously.
    std::optional<int> num_threads{std::nullopt};
  };

  /**
   * Search the C-space polytopic free region
   * C * t <= d
   * t_lower <= t <= t_upper
   * through bilinear alternation.
   * t_lower/t_upper ar the lower and upper bounds of t computed from joint
   * limits.
   * @param q_star t = tan((q - q_star)/2)
   * @param q_inner_pts If this input is not empty, then the searched polytope
   * {C*t<=d, t_lower<=t<=t_upper} needs to contain the t computed from each
   * column of q_inner_pts.
   * @param inner_polytope (C_inner, d_inner) If this input is not empty, then
   * the searched polytope {C*t<=d} needs to contain {C_inner * t <= d_inner,
   * t_lower <= t <= t_upper}.
   * @param[out] C_final At termination, the free polytope is C_final * t <=
   * d_final, t_lower <= t <= t_upper.
   * @param[out] d_final At termination, the free polytope is C_final * t <=
   * d_final, t_lower <= t <= t_upper.
   * @param[out] P_final The inscribed ellipsoid if {P_final*y+q_final | |y|₂≤1}
   * @param[out] q_final The inscribed ellipsoid if {P_final*y+q_final | |y|₂≤1}
   */
  void CspacePolytopeBilinearAlternation(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const Eigen::Ref<const Eigen::MatrixXd>& C_init,
      const Eigen::Ref<const Eigen::VectorXd>& d_init,
      const BilinearAlternationOption& bilinear_alternation_option,
      const solvers::SolverOptions& solver_options,
      const std::optional<Eigen::MatrixXd>& q_inner_pts,
      const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
          inner_polytope,
      CspaceFreeRegionSolution* cspace_free_region_solution) const;

  struct BinarySearchOption {
    double epsilon_max{10};
    double epsilon_min{0};
    int max_iters{5};
    double lagrangian_backoff_scale{0.};
    // If set to true, then after we verify that C*t<=d is collision free, we
    // then fix the Lagrangian multiplier and search the right-hand side vector
    // d through another SOS program.
    bool search_d{true};
    // Whether to compute and print the volume of the C-space polytope.
    bool compute_polytope_volume{false};
    bool verbose{true};
    // If set to an integer, then solve the Lagrangian program through many
    // small SOS with the specified number of threads (if num_threads <= 0 then
    // we launch as many threads as possible). Each program for one pair of
    // collision geometries. std::nullopt means to solve a single big SOS with
    // all the separating planes simultaneously.
    std::optional<int> num_threads{std::nullopt};

    // The objective function used in maximizing the volume of the inscribed
    // ellipsoid.
    EllipsoidVolume ellipsoid_volume{EllipsoidVolume::kNthRoot};
  };

  struct VectorBisectionSearchOption {
    Eigen::VectorXd epsilon_max;
    Eigen::VectorXd epsilon_min;
    // maximum total number of iterations
    int max_iters{20};

    // maximum number of successful bisection steps
    int max_feasible_iters{5};

    // If set to true, then after we verify that C*t<=d is collision free, we
    // then fix the Lagrangian multiplier and search the right-hand side vector
    // d through another SOS program.
    bool search_d{true};
    // Whether to compute and print the volume of the C-space polytope.
    bool compute_polytope_volume{false};

    bool verbose{true};
    // If set to an integer, then solve the Lagrangian program through many
    // small SOS with the specified number of threads (if num_threads <= 0 then
    // we launch as many threads as possible). Each program for one pair of
    // collision geometries. std::nullopt means to solve a single big SOS with
    // all the separating planes simultaneously.
    std::optional<int> num_threads{std::nullopt};

    double lagrangian_backoff_scale{0.};
    // The objective function used in maximizing the volume of the inscribed
    // ellipsoid.
    EllipsoidVolume ellipsoid_volume{EllipsoidVolume::kNthRoot};
  };

  struct InterleavedRegionSearchOptions {
    // options used for VectorBisectionSearch (if used)
    VectorBisectionSearchOption vector_bisection_search_options;

    // options used for scalar Binary search (if used)
    BinarySearchOption scalar_binary_search_options;

    // options used for bilinear alternation
    BilinearAlternationOption bilinear_alternation_options;

    // number of times bisection search and bilinear alternation search are themselves alternated
    int max_method_switch{2};


  };

  /**
   * Find the C-space free polytope C*t<=d through binary search.
   * In each iteration we check if this polytope
   * C*t <= d + ε
   * t_lower <= t <= t_upper
   * is collision free, and do a binary search on ε.
   * where t = tan((q - q_star)/2), t_lower and t_upper are computed from robot
   * joint limits.
   * @note that if binary_search_option.search_d is true, then after we find a
   * feasible scalar ε with C*t<= d+ε being collision free, we then fix the
   * Lagrangian multiplier and search d, and denote the newly found d as
   * d_reset. We then reset ε to zero and find the collision free region C*t <=
   * d_reset + ε through binary search.
   * @param q_inner_pts If this input is not empty, then the searched polytope
   * {C*t<=d, t_lower<=t<=t_upper} needs to contain the t computed from each
   * column of q_inner_pts.
   * @param inner_polytope (C_inner, d_inner) If this input is not empty, then
   * the searched polytope {C*t<=d} needs to contain {C_inner * t <= d_inner,
   * t_lower <= t <= t_upper}.
   */
  void CspacePolytopeBinarySearch(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const Eigen::Ref<const Eigen::MatrixXd>& C,
      const Eigen::Ref<const Eigen::VectorXd>& d_init,
      const BinarySearchOption& binary_search_option,
      const solvers::SolverOptions& solver_options,
      const std::optional<Eigen::MatrixXd>& q_inner_pts,
      const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
          inner_polytope,
      CspaceFreeRegionSolution* cspace_free_region_solution) const;


  /**
   * Run bisection search on one inequality at a time. The order of the inequalities is chosen by bisecting
   * on the vector_bisection_search_option.epsilon_max-vector_bisection_search_option.epsilon min from greatest to smallest
   * @param q_star
   * @param filtered_collision_pairs
   * @param C
   * @param d_init
   * @param vector_bisection_search_option
   * @param num_rounds
   * @param solver_options
   * @param q_inner_pts
   * @param inner_polytope
   * @param cspace_free_region_solution
   */
  void CspacePolytopeRoundRobinBisectionSearch(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const Eigen::Ref<const Eigen::MatrixXd>& C,
      const Eigen::Ref<const Eigen::VectorXd>& d_init,
      const int num_rounds,
      const VectorBisectionSearchOption& vector_bisection_search_option,
      const solvers::SolverOptions& solver_options,
      const std::optional<Eigen::MatrixXd>& q_inner_pts,
      const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
          inner_polytope,
      const std::optional<const systems::Context<double>*> context,
      CspaceFreeRegionSolution* cspace_free_region_solution) const;


  /**
   * Run round robin search for a collection of initial regions.
   * @param q_star
   * @param filtered_collision_pairs
   * @param C
   * @param d_init
   * @param vector_bisection_search_option
   * @param num_rounds
   * @param solver_options
   * @param q_inner_pts
   * @param inner_polytope
   * @param cspace_free_region_solution
   * @param context: diagram context if we want to use snopt max eps
   */
  void CspacePolytopeRoundRobinBisectionSearchForSeedPoints(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const std::vector<Eigen::Ref<Eigen::MatrixXd>>& C_mat_vect,
      const std::vector<Eigen::Ref<Eigen::VectorXd>>& d_init_vect,
      const int num_rounds,
      const std::vector<VectorBisectionSearchOption>& vector_bisection_search_option_vect,
      const solvers::SolverOptions& solver_options,
      const std::vector<Eigen::MatrixXd>& seed_points,
      const std::optional<std::vector<std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>>>&
          inner_polytope_vect,
      const std::optional<const systems::Context<double>*> context,
      std::vector<CspaceFreeRegionSolution*> cspace_free_region_solution_vect) const;

  /**
   * Find the C-space free polytope C*t<=d through bisection search.
   * In each iteration we check if this polytope
   * C*t <= d + ε
   * t_lower <= t <= t_upper
   * is collision free, and do a binary search on ε.
   * where t = tan((q - q_star)/2), t_lower and t_upper are computed from robot
   * joint limits.
   * @note that if vector_bisection_search_option.search_d is true, then after we find a
   * feasible vector ε with C*t<= d+ε being collision free, we then fix the
   * Lagrangian multiplier and search d, and denote the newly found d as
   * d_reset. We then reset ε to zero and find the collision free region C*t <=
   * d_reset + ε through bisection search.
   * @param q_inner_pts If this input is not empty, then the searched polytope
   * {C*t<=d, t_lower<=t<=t_upper} needs to contain the t computed from each
   * column of q_inner_pts.
   * @param inner_polytope (C_inner, d_inner) If this input is not empty, then
   * the searched polytope {C*t<=d} needs to contain {C_inner * t <= d_inner,
   * t_lower <= t <= t_upper}.
   */
  void CspacePolytopeBisectionSearchVector(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const Eigen::Ref<const Eigen::MatrixXd>& C,
      const Eigen::Ref<const Eigen::VectorXd>& d_init,
      const VectorBisectionSearchOption& vector_bisection_search_option,
      const solvers::SolverOptions& solver_options,
      const std::optional<Eigen::MatrixXd>& q_inner_pts,
      const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
          inner_polytope,
      CspaceFreeRegionSolution* cspace_free_region_solution) const;

  /**
   * Search for a cspace polytope from an initial guess polytope by alternating between
   * bisection search and bilinear alternation
   */
   void InterleavedCSpacePolytopeSearch(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const Eigen::Ref<const Eigen::MatrixXd>& C_init,
    const Eigen::Ref<const Eigen::VectorXd>& d_init,
    const int num_round_robin_rounds,
    const InterleavedRegionSearchOptions& interleaved_region_search_option,
    const solvers::SolverOptions& solver_options,
    const Eigen::MatrixXd& seed_point_t,
    const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
        inner_polytope,
    const std::optional<const systems::Context<double>*> context,
    CspaceFreeRegionSolution* cspace_free_region_solution) const;
   /**
    * same as above with a list of seed points
    * @param q_star
    * @param filtered_collision_pairs
    * @param C_mat_vect
    * @param d_init_vect
    * @param num_round_robin_rounds
    * @param interleaved_region_search_option
    * @param solver_options
    * @param seed_points
    * @param inner_polytope_vect
    * @param cspace_free_region_solution_vect
    */
   void InterleavedCSpacePolytopeSearchForSeedPoints(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const std::vector<Eigen::Ref<Eigen::MatrixXd>>& C_mat_vect,
      const std::vector<Eigen::Ref<Eigen::VectorXd>>& d_init_vect,
      const int num_round_robin_rounds,
      const std::vector<InterleavedRegionSearchOptions>& interleaved_region_search_option_arg,
      const solvers::SolverOptions& solver_options,
      const std::vector<Eigen::MatrixXd>& seed_points,
      const std::optional<std::vector<std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>>>&
          inner_polytope_vect,
      const std::optional<const systems::Context<double>*> context,
      std::vector<CspaceFreeRegionSolution*> cspace_free_region_solution_vect) const ;

  const RationalForwardKinematics& rational_forward_kinematics() const {
    return rational_forward_kinematics_;
  }

  const geometry::SceneGraph<double>& scene_graph() const {
    return *scene_graph_;
  }

  const std::vector<SeparatingPlane>& separating_planes() const {
    return separating_planes_;
  }

  const std::map<multibody::BodyIndex, std::vector<ConvexPolytope>>&
  polytope_geometries() const {
    return polytope_geometries_;
  }

  const SeparatingPlaneOrder& plane_order() const {
    return plane_order_;
  }

  const CspaceRegionType& cspace_region_type() const {
    return cspace_region_type_;
  }

 private:
  RationalForwardKinematics rational_forward_kinematics_;
  const geometry::SceneGraph<double>* scene_graph_;
  std::map<multibody::BodyIndex, std::vector<ConvexPolytope>>
      polytope_geometries_;

  SeparatingPlaneOrder plane_order_;
  CspaceRegionType cspace_region_type_;
  std::vector<SeparatingPlane> separating_planes_;
//  std::shared_ptr<systems::Context<double>> diagram_context_;

  // separating_planes_[(geometry1_id, geometry2_id)] is the separating plane
  // that separates geometry1 and geometry 2.
  std::unordered_map<SortedPair<ConvexGeometry::Id>, int>
      map_polytopes_to_separating_planes_;
};

/**
 * Generate the rational functions a_A.dot(p_AVi(t)) + b(i) - 1 or -1 -
 * a_A.dot(p_AVi(t)) - b(i). Which represents that the link (whose vertex Vi has
 * position p_AVi in the frame A) is on the positive (or negative) side of the
 * plane a_A * x + b = 0
 * @param X_AB_multilinear The pose of the link frame B in the expressed body
 * frame A. Note that this pose is a multilinear function of sinθ and cosθ.
 */
std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const ConvexPolytope* polytope, const ConvexPolytope* other_side_polytope,
    const RationalForwardKinematics::Pose<symbolic::Polynomial>&
        X_AB_multilinear,
    const drake::Vector3<symbolic::Expression>& a_A,
    const symbolic::Expression& b, PlaneSide plane_side,
    SeparatingPlaneOrder plane_order);

bool IsGeometryPairCollisionIgnored(
    ConvexGeometry::Id id1, ConvexGeometry::Id id2,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs);

bool IsGeometryPairCollisionIgnored(
    const SortedPair<ConvexGeometry::Id>& geometry_pair,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs);

void ComputeBoundsOnT(const Eigen::Ref<const Eigen::VectorXd>& q_star,
                      const Eigen::Ref<const Eigen::VectorXd>& q_upper,
                      const Eigen::Ref<const Eigen::VectorXd>& q_lower,
                      Eigen::VectorXd* t_lower, Eigen::VectorXd* t_upper);

/**
 * Construct the polynomial mᵀQm from monomial basis m and the lower diagonal
 * part of Q.
 * @param monomial_basis m in the documentation above.
 * @param gram Q in the documentation above.
 */
template <typename T>
symbolic::Polynomial CalcPolynomialFromGram(
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Eigen::Ref<const MatrixX<T>>& gram);

/**
 * Overloads CalcPolynomialFromGram. It first evaluates each entry of gram
 * from `result`. This function avoids dynamic memory allocation of the gram
 * matrix result.
 */
symbolic::Polynomial CalcPolynomialFromGram(
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& gram,
    const solvers::MathematicalProgramResult& result);

/**
 * Construct the polynomial mᵀQm from monomial basis m and the lower diagonal
 * part of Q.
 * @param monomial_basis m in the documentation above.
 * @param gram_lower stacking the columns of Q's lower part.
 */
template <typename T>
symbolic::Polynomial CalcPolynomialFromGramLower(
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Eigen::Ref<const VectorX<T>>& gram_lower);

/**
 * Overload CalcPolynomialFromGramLower, but the value of gram matrix Q is
 * obtained from result. This function avoids dynamically allocate the matrix
 * for Q's value.
 */
symbolic::Polynomial CalcPolynomialFromGramLower(
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& gram_lower,
    const solvers::MathematicalProgramResult& result);

/**
 * Given the lower part of a symmetric matrix, fill the whole matrix.
 * @param lower Stacking the column of the matrix lower part.
 */
template <typename T>
void SymmetricMatrixFromLower(int mat_rows,
                              const Eigen::Ref<const VectorX<T>>& lower,
                              MatrixX<T>* mat);

/**
 * Add the constraint that an ellipsoid {Py+q | |y|₂ <= 1} is contained in the
 * polytope {t | C*t <= d, t_lower <= t <= t_upper}.
 *
 * Mathematically the constraint is
 * |cᵢᵀP|₂ ≤ dᵢ−cᵢᵀq
 * |P.row(i)|₂ + qᵢ ≤ t_upper(i)
 * −|P.row(i)|₂ + qᵢ ≥ t_lower(i)
 * P is p.s.d,
 * @param P A symmetric matrix already registered as decision variable in
 * `prog`.
 * @param q Already registered as decision variable in `prog`.
 * @param constrain_P_psd Whether we add the constraint P is psd (If you
 * maximize log(det(P)) later, then calling
 * prog.AddMaximizeLogDeterminantSymmetricMatrixCost(P) will automatically
 * constraint P being psd).
 */
void AddInscribedEllipsoid(
    solvers::MathematicalProgram* prog,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& P,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& q,
    bool constrain_P_psd = true);

/**
 * Add the constraint such that the ellipsoid {Py+q | |y|₂≤ 1} is contained
 * within the polytope C * t <=d with a margin no smaller than δ.
 * Mathematically the constraint is
 * |cᵢᵀP|₂ ≤ dᵢ − cᵢᵀq − δᵢ
 * |cᵢᵀ|₂ ≤ 1
 * where cᵢᵀ is the i'th row of C.
 * @param C This should have registered as decision variable in prog.
 * @param d This should have registered as decision variable in prog.
 * @param margin δ in the documentation above. This should have registered as
 * decision variable in prog.
 */
void AddOuterPolytope(
    solvers::MathematicalProgram* prog,
    const Eigen::Ref<const Eigen::MatrixXd>& P,
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& C,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& d,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& margin);

/**
 * Given a diagram (which contains the plant and the scene_graph), returns all
 * the convex polytopes.
 */
std::map<BodyIndex, std::vector<ConvexPolytope>> GetConvexPolytopes(
    const systems::Diagram<double>& diagram,
    const MultibodyPlant<double>* plant,
    const geometry::SceneGraph<double>* scene_graph);

/**
 * Find the redundant constraint in {C*t<=d, t_lower<=t<=t_upper}. We regard a
 * constraint aᵀt ≤ b being redundant if its tightened version aᵀt≤ b-tighten is
 * implied by the other (untightened) constraints.
 */
void FindRedundantInequalities(
    const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
    const Eigen::VectorXd& t_lower, const Eigen::VectorXd& t_upper,
    double tighten, std::unordered_set<int>* C_redundant_indices,
    std::unordered_set<int>* t_lower_redundant_indices,
    std::unordered_set<int>* t_upper_redundant_indices);

/**
 * When we do binary search to find epsilon such that C*t<= d + epsilon is
 * collision free, if epsilon is too small, then this polytope {t| C*t<=
 * d+epsilon * 𝟏, t_lower<=t<=t_upper} can be empty. To avoid this case, we find
 * the minimal epsilon such that this polytope is non-empty.
 * @param t_lower The lower bound of t computed from joint lower limits.
 * @param t_upper The upper bound of t computed from joint upper limits.
 * @param t_inner_pts. If non-empty, then the polytope C*t+epsilon also have to
 * contain each column of t_inner_pts.
 * @param inner_polytope. A pair (C_bar, d_bar). If non-empty, then the polytope
 * C*t<=d+epsilon also has to contain the polytope {C_bar*t<=d_bar,
 * t_lower<=t<=t_upper}.
 */
// TODO(Alex.Amice): add the version with epsilon being a vector, and search for
// each epsilon independently.
double FindEpsilonLower(
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    const std::optional<Eigen::MatrixXd>& t_inner_pts,
    const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
        inner_polytope);

Eigen::VectorXd FindEpsilonLowerVector(
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    const Eigen::MatrixXd& t_inner_pts,
    const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
        inner_polytope);

Eigen::VectorXd FindEpsilonUpperVector(
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper);


/**
 * When we do binary search to find epsilon such that C*t<= d + epsilon is
 * collision free, if epsilon is too small, then this polytope {t| C*t<=
 * d+epsilon * 𝟏, t_lower<=t<=t_upper} can be empty. To avoid this case, we find
 * the minimal epsilon such that this polytope is non-empty.
 * @param t_lower The lower bound of t computed from joint lower limits.
 * @param t_upper The upper bound of t computed from joint upper limits.
 * @param t_inner_pts. If non-empty, then the polytope C*t+epsilon also have to
 * contain each column of t_inner_pts.
 * @param inner_polytope. A pair (C_bar, d_bar). If non-empty, then the polytope
 * C*t<=d+epsilon also has to contain the polytope {C_bar*t<=d_bar,
 * t_lower<=t<=t_upper}.
 */
// TODO(Alex.Amice): add the version with epsilon being a vector, and search for
// each epsilon independently.
double FindEpsilonVectorLower(
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    const std::optional<Eigen::MatrixXd>& t_inner_pts,
    const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
        inner_polytope);

/**
 * Concatenate the polytope C*t<=d, t_lower <= t <= t_upper as C_bar * t <=
 * d_bar, where C_bar = [C; I; -I], d_bar = [d;t_upper;-t_lower].
 */
void GetCspacePolytope(const Eigen::Ref<const Eigen::MatrixXd>& C,
                       const Eigen::Ref<const Eigen::VectorXd>& d,
                       const Eigen::Ref<const Eigen::VectorXd>& t_lower,
                       const Eigen::Ref<const Eigen::VectorXd>& t_upper,
                       Eigen::MatrixXd* C_bar, Eigen::VectorXd* d_bar);

/**
 * Add the constraint that the polytope C*t<=d contains the polytope
 * C_inner*t<=d_inner, t_lower <= t <= t_upper.
 */
void AddCspacePolytopeContainment(
    solvers::MathematicalProgram* prog, const MatrixX<symbolic::Variable>& C,
    const VectorX<symbolic::Variable>& d,
    const Eigen::Ref<const Eigen::MatrixXd>& C_inner,
    const Eigen::Ref<const Eigen::VectorXd>& d_inner,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper);

/**
 * Add the constraints that C*t<=d contains each column of inner_pts.
 */
void AddCspacePolytopeContainment(
    solvers::MathematicalProgram* prog, const MatrixX<symbolic::Variable>& C,
    const VectorX<symbolic::Variable>& d,
    const Eigen::Ref<const Eigen::MatrixXd>& inner_pts);

/**
 * Compute the volume of the Cspace region C*t<=d, t_lower <= t <= t_upper.
 */
[[nodiscard]] double CalcCspacePolytopeVolume(const Eigen::MatrixXd& C,
                                              const Eigen::VectorXd& d,
                                              const Eigen::VectorXd& t_lower,
                                              const Eigen::VectorXd& t_upper);

namespace internal {
// Some of the separating planes will be ignored by filtered_collision_pairs.
// Returns std::vector<bool> to indicate if each plane is active or not.
std::vector<bool> IsPlaneActive(
    const std::vector<SeparatingPlane>& separating_planes,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs);

/** For a given polytopic C-space region C * t <= d, t_lower <= t <= t_upper,
 * verify if this region is collision-free by solving the SOS program to find
 * the separating planes and the Lagrangian multipliers. Return true if the SOS
 * is successful, false otherwise.
 * The inputs to this function is the output of
 * GenerateTuplesForBilinearAlternation()
 * @param num_threads If set to std::nullopt, then solve a large SOS that
 * searches for the separating planes and Lagrangian multipliers for all pairs
 * of collision geometries. Otherwise, we solve many small SOS in parallel with
 * this specified number of threads (if num_threads <= 0 then we launch as many
 * threads as possible), each SOS for one pair of collision geometries.
 * @param If a solution is found, it is written into cspace_free_region
 */
bool FindLagrangianAndSeparatingPlanes(
    const CspaceFreeRegion& cspace_free_region,
    const std::vector<CspaceFreeRegion::CspacePolytopeTuple>&
        alternation_tuples,
    const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
    const VectorX<symbolic::Variable>& lagrangian_gram_vars,
    const VectorX<symbolic::Variable>& verified_gram_vars,
    const VectorX<symbolic::Variable>& separating_plane_vars,
    const Eigen::VectorXd& t_lower, const Eigen::VectorXd& t_upper,
    const VerificationOption& verification_option,
    std::optional<double> redundant_tighten,
    const solvers::SolverOptions& solver_options, bool verbose,
    std::optional<int> num_threads,
    const std::vector<std::vector<int>>& separating_plane_to_tuples,
    Eigen::VectorXd* lagrangian_gram_var_vals,
    Eigen::VectorXd* verified_gram_var_vals,
    Eigen::VectorXd* separating_plane_var_vals,
    CspaceFreeRegionSolution* cspace_free_region_solution);

struct LagrangianStepDetail {
  public:

  LagrangianStepDetail(double time):optimizer_time{time} {};

  const double optimizer_time;

};

struct PolytopeStepDetail {
  public:

  PolytopeStepDetail(double time,
                     double pre_volume,
                     double post_volume):
                     optimizer_time{time}, volume_before{pre_volume},
                     volume_after{post_volume}, growth{post_volume-pre_volume} {};
  const double optimizer_time;
  const double volume_before;
  const double volume_after;
  const double growth;
};

struct AlternationSolutionDetails {
 public:
  std::deque<LagrangianStepDetail> lagrangian_step_details;

  std::deque<PolytopeStepDetail> poltope_step_details;

  void log_lagrangian_step(LagrangianStepDetail step) {
    lagrangian_step_details.push_back(step);
  }

  void log_polytope_step(PolytopeStepDetail step) {
    poltope_step_details.push_back(step);
  }
};



}  // namespace internal
}  // namespace multibody
}  // namespace drake
