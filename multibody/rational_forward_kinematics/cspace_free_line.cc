#include "drake/multibody/rational_forward_kinematics/cspace_free_line.h"

#include <execution>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>

#include "drake/common/symbolic/decompose.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace multibody {
using solvers::MathematicalProgram;

namespace {
// TODO(Alex.Amice) THIS is COPIED FROM cspace_free_region.cc. Resolve this with
// Hongkai
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

CspaceLineTuple::CspaceLineTuple(
    const symbolic::Variable& mu, const drake::VectorX<symbolic::Variable>& s0,
    const drake::VectorX<symbolic::Variable>& s1,
    const symbolic::Polynomial& m_rational_numerator,
    const VerificationOption& option)
    : p_{m_rational_numerator},
      psatz_variables_and_psd_constraints_{solvers::MathematicalProgram()},
      s0_{s0},
      s1_{s1} {
  psatz_variables_and_psd_constraints_.AddIndeterminates(
      solvers::VectorIndeterminate<1>(mu));
  p_.SetIndeterminates({mu});

  // Constructs the multiplier polynomials and their associated Gram matrices as
  // well as the polynomial p_.
  if (p_.TotalDegree() > 0) {
    int d = p_.TotalDegree() / 2;
    auto [lambda, Q_lambda] =
        psatz_variables_and_psd_constraints_.NewSosPolynomial(
            {mu}, 2 * d, option.lagrangian_type, "Sl");
    if (p_.TotalDegree() % 2 == 0) {
      auto [nu, Q_nu] = psatz_variables_and_psd_constraints_.NewSosPolynomial(
          {mu}, 2 * d - 2, option.lagrangian_type, "Sv");
      p_ -= lambda + nu * mu * (symbolic::Polynomial(1, {mu}) - mu);
    } else {
      auto [nu, Q_nu] = psatz_variables_and_psd_constraints_.NewSosPolynomial(
          {mu}, 2 * d, option.lagrangian_type, "Sv");
      p_ -= lambda * mu + nu * (symbolic::Polynomial(1, {mu}) - mu);
    }
  }
  p_ = p_.Expand();
}

std::vector<solvers::Binding<solvers::LinearEqualityConstraint>>
CspaceLineTuple::AddPsatzConstraintToProg(
    solvers::MathematicalProgram* prog,
    const Eigen::Ref<const Eigen::VectorXd>& s0,
    const Eigen::Ref<const Eigen::VectorXd>& s1) const {
  symbolic::Environment env;
  for (int i = 0; i < s0.size(); ++i) {
    env.insert(s0_[i], s0[i]);
    env.insert(s1_[i], s1[i]);
  }
  // Imposing the constraint that p(μ) = 0 by substituting s0 and s1 with their
  // corresponding values.
  return prog->AddEqualityConstraintBetweenPolynomials(p_.EvaluatePartial(env),
                                                       symbolic::Polynomial());
}

void CspaceLineTuple::AddTupleOnSideOfPlaneConstraint(
    solvers::MathematicalProgram* prog,
    const Eigen::Ref<const Eigen::VectorXd>& s0,
    const Eigen::Ref<const Eigen::VectorXd>& s1) const {
  // p_ is a constant and therefore must be non-negative to be a non-negative
  // function
  if (p_.TotalDegree() == 0) {
    for (const auto& [monomial, coeff] : p_.monomial_to_coefficient_map()) {
      prog->AddLinearConstraint(coeff >= 0);
    }
  }
  // p_ is a polynomial function and therefore requires a psatz condition
  else {
    prog->AddDecisionVariables(
        psatz_variables_and_psd_constraints_.decision_variables());
    for (const auto& binding :
         psatz_variables_and_psd_constraints_.GetAllConstraints()) {
      prog->AddConstraint(binding);
    }
    AddPsatzConstraintToProg(prog, s0, s1);
  }
}

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
      s0_{symbolic::MakeVectorContinuousVariable(plant->num_positions(), "s0")},
      s1_{symbolic::MakeVectorContinuousVariable(plant->num_positions(), "s1")},
      filtered_collision_pairs_{filtered_collision_pairs},
      option_{option} {
  if (q_star.has_value()) {
    DRAKE_DEMAND(q_star.value().size() == plant->num_positions());
    q_star_ = q_star.value();
  } else {
    q_star_ = Eigen::VectorXd::Zero(plant->num_positions());
  }

  // allocate all the tuples
  GenerateTuplesForCertification(
      q_star_, filtered_collision_pairs_, &tuples_, &separating_plane_vars_,
      &separating_plane_to_tuples_,
      &separating_plane_to_lorentz_cone_constraints_);
}

bool CspaceFreeLine::CertifyTangentConfigurationSpaceLine(
    const Eigen::Ref<const Eigen::VectorXd>& s0,
    const Eigen::Ref<const Eigen::VectorXd>& s1,
    const solvers::SolverOptions& solver_options,
    std::vector<SeparatingPlane<double>>* separating_planes_sol) const {
  if (!PointInJointLimits(s0) || !PointInJointLimits(s1)) {
    return false;
  }

  std::vector<bool> is_success(separating_planes().size());
  separating_planes_sol->resize(separating_planes().size());

  // TODO(Alex.Amice) parallelize the certification of each plane.
  auto clock_start = std::chrono::system_clock::now();
  //  std::for_each(std::execution::par_unseq, separating_planes().begin(),
  //                separating_planes().end(),
  //                [&is_success, &separating_planes_sol, &solver_options,
  //                 this](SeparatingPlane<symbolic::Variable>& plane) {
  //                  int plane_index =
  //                      static_cast<int>(&plane - &(separating_planes()[0]));
  //                  is_success[plane_index] = this->CertifyPlane(
  //                      plane_index,
  //                      &(separating_planes_sol->at(plane_index)),
  //                      solver_options);
  //                });

  //    for (const auto& plane : separating_planes()) {
  //      int plane_index = static_cast<int>(&plane -
  //      &(separating_planes()[0])); is_success[plane_index] =
  //      this->CertifyPlane(
  //          plane_index, &(separating_planes_sol->at(plane_index)),
  //          solver_options);
  //    }
  //  bool ret = std::all_of(is_success.begin(), is_success.end(),
  //                         [](bool val) { return val; });

  // make one big program
  solvers::MathematicalProgram prog = solvers::MathematicalProgram();
  prog.AddDecisionVariables(separating_plane_vars_);
  prog.AddIndeterminates(solvers::VectorIndeterminate<1>(mu_));
  for (int i = 0; i < static_cast<int>(separating_planes().size()); ++i) {
    AddCertifySeparatingPlaneConstraintToProg(&prog, i, s0, s1);
  }
  auto result = solvers::Solve(prog, std::nullopt, solver_options);
  auto clock_now = std::chrono::system_clock::now();
  drake::log()->debug(fmt::format(
      "Line\n s0: {}\n s1: {}\n certified in {} s", s0, s1,
      static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(
                             clock_now - clock_start)
                             .count()) /
          1000));
  if (result.is_success()) {
    for (int i = 0; i < static_cast<int>(separating_planes().size()); ++i) {
      separating_planes_sol->at(i) =
          GetSeparatingPlaneSolution(this->separating_planes()[i], result);
    }
  }
  bool ret = result.is_success();
  return ret;
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
std::vector<bool> CspaceFreeLine::CertifyTangentConfigurationSpaceLines(
    const Eigen::Ref<const Eigen::MatrixXd>& s0,
    const Eigen::Ref<const Eigen::MatrixXd>& s1,
    const solvers::SolverOptions& solver_options,
    std::vector<std::vector<SeparatingPlane<double>>>*
        separating_planes_sol_per_row) const {
  DRAKE_DEMAND(s0.rows() == s1.rows());
  DRAKE_DEMAND(s0.cols() == s1.cols());

  // cannot use vector of bools as they aren't thread safe like other types.
  std::vector<uint8_t> pair_is_safe(s0.rows(), 0);
  separating_planes_sol_per_row->resize(s0.rows());
  const auto certify_line = [this, &pair_is_safe, &s0, &s1, &solver_options,
                             &separating_planes_sol_per_row](int i) {
    solvers::MathematicalProgram prog = solvers::MathematicalProgram();
    pair_is_safe.at(i) = this->CertifyTangentConfigurationSpaceLine(
                             s0.row(i), s1.row(i), solver_options,
                             &(separating_planes_sol_per_row->at(i)))
                             ? 1
                             : 0;
    return i;
  };

  // We implement the "thread pool" idea here, by following
  // MonteCarloSimulationParallel class. This implementation doesn't use
  // openMP library. Storage for active parallel SOS operations.

  // use as many threads as optimal for hardware.
  unsigned int num_threads = std::thread::hardware_concurrency();
  drake::log()->debug("Certifying with {} threads", num_threads);
  std::list<std::future<int>> active_operations;
  // Keep track of how many certifications have been dispatched already.
  int certs_dispatched = 0;
  while ((active_operations.size() > 0 || certs_dispatched < s0.rows())) {
    // Check for completed operations.
    for (auto operation = active_operations.begin();
         operation != active_operations.end();) {
      if (IsFutureReady(*operation)) {
        // This call to future.get() is necessary to propagate any exception
        // thrown during certification setup/solve.
        const int certification_count = operation->get();
        drake::log()->debug(
            "Pair s0: {}, s1: {} completed, is_collision free = {}",
            s0.row(certification_count), s1.row(certification_count),
            pair_is_safe.at(certification_count));
        // Erase returns iterator to the next node in the list.
        operation = active_operations.erase(operation);
      } else {
        // Advance to next node in the list.
        ++operation;
      }
    }

    // Dispatch new certification.
    while (active_operations.size() < num_threads &&
           certs_dispatched < s0.rows()) {
      active_operations.emplace_back(std::async(
          std::launch::async, std::move(certify_line), certs_dispatched));
      drake::log()->debug("Certification of {}/{} dispatched",
                          certs_dispatched + 1, s0.rows());
      ++certs_dispatched;
    }
    // Wait a bit before checking for completion.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::vector<bool> pair_is_safe_bool(pair_is_safe.size());
  for (int i = 0; i < static_cast<int>(pair_is_safe.size()); ++i) {
    pair_is_safe_bool.at(i) = pair_is_safe.at(i) > 0;
  }
  return pair_is_safe_bool;
}

namespace {
// Performing the substitution t = μ*s₀ + (1−μ)*s₁ can require expanding very
// high large products of μ which requires a lot of time using factory methods.
// Here we construct the substitution manually to avoid these long substitution
// times.
// @param t_to_line_subs: a dictionary containing the substitution t[i] =
// μ*s₀[i] + (1−μ)*s₁[i]
// @param t_monomial_to_mu_polynomial_map: a dictionary mapping monomials of the
// form ∏ᵢt[i] to expanded polynomial products of ∏ᵢ(μ*s₀[i] + (1−μ)*s₁[i]).
// This dictionary will be updated with new expanded products as this method is
// called to avoid needing to recompute the expansion several times.
symbolic::Polynomial PerformTtoMuSubstitution(
    const symbolic::Polynomial& t_polynomial, const symbolic::Variable& mu,
    const std::unordered_map<symbolic::Variable, symbolic::Expression>&
        t_to_line_subs,
    std::unordered_map<symbolic::Monomial, symbolic::Polynomial>*
        t_monomial_to_mu_polynomial_map) {
  // This is the final monomial to coefficient map for the returned polynomial.
  symbolic::Polynomial::MapType mu_monomial_to_coeff_map;

  for (const auto& [t_monomial, t_coeff] :
       t_polynomial.monomial_to_coefficient_map()) {
    // If we haven't already computed the substitution for the current monomial
    // ∏ᵢt[i] then do so now. Each t_monomial maps to a polynomial with
    // indeterminates μ and coefficients in s₀ and s₁
    if (t_monomial_to_mu_polynomial_map->find(t_monomial) ==
        t_monomial_to_mu_polynomial_map->end()) {
      symbolic::Expression t_monomial_expression =
          t_monomial.ToExpression().Substitute(t_to_line_subs).Expand();
      symbolic::Polynomial t_monomial_to_mu_substitution =
          symbolic::Polynomial(t_monomial_expression, {mu});
      t_monomial_to_mu_polynomial_map->insert(
          {t_monomial, t_monomial_to_mu_substitution});
    }

    // Now add the value of t_coeff * t_monomial maps to the monomials in μ.
    for (const auto& [mu_monomial, mu_coeff] :
         t_monomial_to_mu_polynomial_map->at(t_monomial)
             .monomial_to_coefficient_map()) {
      // We don't have this basis element in μ yet so add it in.
      if (mu_monomial_to_coeff_map.find(mu_monomial) ==
          mu_monomial_to_coeff_map.end()) {
        mu_monomial_to_coeff_map.insert({mu_monomial, symbolic::Expression()});
      }

      mu_monomial_to_coeff_map.at(mu_monomial) += mu_coeff * t_coeff;
    }
  }
  return symbolic::Polynomial(mu_monomial_to_coeff_map);
}

}  // namespace

std::vector<LinkOnPlaneSideRational>
CspaceFreeLine::GenerateRationalsForLinkOnOneSideOfPlane(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs)
    const {
  // First we reuse the code from CspaceFreeRegion to obtain the rational
  // forward kinematics expressions for all the objects in the scene
  std::vector<LinkOnPlaneSideRational> generic_rationals =
      CspaceFreeRegion::GenerateRationalsForLinkOnOneSideOfPlane(
          q_star, filtered_collision_pairs);

  // Now we will perform the substitution t = μ*s₀ + (1−μ)*s₁ to take the
  // forward kinematics from an n-variate function in t to a univariate function
  // in μ. First we precompute the map for performing substitution from t to
  // μ*s₀ + (1−μ)*s₁
  std::unordered_map<symbolic::Variable, symbolic::Expression>
      t_to_line_subs_map;
  const drake::VectorX<drake::symbolic::Variable>& t =
      (this->rational_forward_kinematics()).t();
  for (int i = 0;
       i < ((this->rational_forward_kinematics()).plant().num_positions());
       ++i) {
    // equivalent to μ*s₀ + (1−μ)*s₁ but requires less traversal in
    // sustitutions
    t_to_line_subs_map[t[i]] = (s0_[i] - s1_[i]) * mu_ + s1_[i];
  }
  std::unordered_map<symbolic::Monomial, symbolic::Polynomial>
      t_monomial_to_mu_polynomial_map;

  std::vector<LinkOnPlaneSideRational> rationals;
  int num_rats = generic_rationals.size();
  rationals.reserve(num_rats);

  symbolic::Expression numerator_expr;
  symbolic::Polynomial numerator_poly;
  int ctr = 0;

  // TODO(Alex.Amice) We could parallelize this too to avoid long construction
  // times but for now I think it is okay.
  for (const auto& rational : generic_rationals) {
    auto clock_start = std::chrono::system_clock::now();
    numerator_poly = PerformTtoMuSubstitution(rational.rational.numerator(),
                                              mu_, t_to_line_subs_map,
                                              &t_monomial_to_mu_polynomial_map);
    auto clock_end = std::chrono::system_clock::now();
    drake::log()->debug(fmt::format(
        "numerator poly constructed in {} s. Has degree: {}",
        static_cast<float>(
            std::chrono::duration_cast<std::chrono::milliseconds>(clock_end -
                                                                  clock_start)
                .count()) /
            1000,
        numerator_poly.TotalDegree()));

    rationals.emplace_back(
        symbolic::RationalFunction(numerator_poly, symbolic::Polynomial(1)),
        rational.link_geometry, rational.expressed_body_index,
        rational.other_side_link_geometry, rational.a_A, rational.b,
        rational.plane_side, rational.plane_order,
        rational.lorentz_cone_constraints);
    drake::log()->debug(fmt::format("Done rational {}/{}", ctr, num_rats));
    ctr++;
  }
  return rationals;
}

void CspaceFreeLine::GenerateTuplesForCertification(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs,
    std::list<CspaceLineTuple>* tuples,
    VectorX<symbolic::Variable>* separating_plane_vars,
    std::vector<std::vector<int>>* separating_plane_to_tuples,
    std::vector<std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>*
        separating_plane_to_lorentz_cone_constraints) const {
  // Build tuples.
  const auto rationals = GenerateRationalsForLinkOnOneSideOfPlane(
      q_star, filtered_collision_pairs);

  separating_plane_to_tuples->resize(
      static_cast<int>(this->separating_planes().size()));
  separating_plane_to_tuples->resize(3);

  for (const auto& rational : rationals) {
    tuples->emplace_back(mu_, s0_, s1_, rational.rational.numerator(), option_);
    (*separating_plane_to_tuples)
        [this->map_geometries_to_separating_planes().at(
             SortedPair<geometry::GeometryId>(
                 rational.link_geometry->id(),
                 rational.other_side_link_geometry->id()))]
            .push_back(tuples->size() - 1);
  }

  // Set separating_plane_vars.
  int separating_plane_vars_count = 0;
  for (const auto& separating_plane : this->separating_planes()) {
    separating_plane_vars_count += separating_plane.decision_variables.rows();
  }
  separating_plane_vars->resize(separating_plane_vars_count);
  separating_plane_vars_count = 0;
  for (const auto& separating_plane : this->separating_planes()) {
    separating_plane_vars->segment(separating_plane_vars_count,
                                   separating_plane.decision_variables.rows()) =
        separating_plane.decision_variables;
    separating_plane_vars_count += separating_plane.decision_variables.rows();
  }
  // Set the separating plane lorentz cone constraints.
  separating_plane_to_lorentz_cone_constraints->clear();
  separating_plane_to_lorentz_cone_constraints->resize(
      this->separating_planes().size());
  for (const auto& rational : rationals) {
    if (!rational.lorentz_cone_constraints.empty()) {
      const int plane_index = this->map_geometries_to_separating_planes().at(
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

void CspaceFreeLine::AddCertifySeparatingPlaneConstraintToProg(
    solvers::MathematicalProgram* prog, int i,
    const Eigen::Ref<const Eigen::VectorXd>& s0,
    const Eigen::Ref<const Eigen::VectorXd>& s1) const {
  // tuples_ is a list and there are typically more of these than indices in
  // separating_plane_to_tuples_[i] so it is faster to step through the list in
  // this outer loop.
  int ctr = 0;
  const std::vector<int> tuple_indices = separating_plane_to_tuples_[i];
  for (const auto& tuple : tuples_) {
    // if this index is one we need
    if (std::find(tuple_indices.begin(), tuple_indices.end(), ctr) !=
        tuple_indices.end()) {
      tuple.AddTupleOnSideOfPlaneConstraint(prog, s0, s1);
    }
    ++ctr;
  }
}

bool CspaceFreeLine::PointInJointLimits(
    const Eigen::Ref<const Eigen::VectorXd>& s) const {
  const Eigen::VectorXd q_min =
      this->rational_forward_kinematics().plant().GetPositionLowerLimits();
  const Eigen::VectorXd q_max =
      this->rational_forward_kinematics().plant().GetPositionUpperLimits();
  const Eigen::VectorXd s_min =
      this->rational_forward_kinematics().ComputeTValue(q_min, this->q_star_);
  const Eigen::VectorXd s_max =
      this->rational_forward_kinematics().ComputeTValue(q_max, this->q_star_);
  DRAKE_DEMAND(s.size() == s_min.size());
  for (int i = 0; i < s.size(); ++i) {
    if (s(i) < s_min(i) || s_max(i) < s(i)) {
      return false;
    }
  }
  return true;
}

}  // namespace multibody
}  // namespace drake
