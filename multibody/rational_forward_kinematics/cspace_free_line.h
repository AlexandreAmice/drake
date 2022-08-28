#pragma once
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"

namespace drake {
namespace multibody {

class CspaceLineTuple {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CspaceLineTuple)

  CspaceLineTuple(const symbolic::Variable& mu,
                  const drake::VectorX<symbolic::Variable>& s0,
                  const drake::VectorX<symbolic::Variable>& s1,
                  const symbolic::Polynomial& m_rational_numerator,
                  const VerificationOption& option);

  void AddTupleOnSideOfPlaneConstraint(
      solvers::MathematicalProgram* prog,
      const Eigen::Ref<const Eigen::VectorXd>& s0,
      const Eigen::Ref<const Eigen::VectorXd>& s1) const;

  const solvers::MathematicalProgram* get_psatz_variables_and_psd_constraints()
      const {
    return &psatz_variables_and_psd_constraints_;
  }

  const symbolic::Polynomial& get_p() const { return p_; }

 private:
  // a univariate polynomial q(μ) is nonnegative on [0, 1] if and
  // only if q(μ) = λ(μ) + ν(μ)*μ*(1-μ) if deg(q) = 2d with deg(λ) ≤ 2d and
  // deg(ν) ≤ 2d - 2 or q(μ) = λ(μ)*μ + ν(μ)*(1-μ) if deg(q) = 2d + 1 with
  // deg(λ) ≤ 2d and deg(ν) ≤ 2d and λ, ν are SOS. We construct the polynomial
  // p_(μ) = m_rational_numerator-q(μ) which we will later constrain to be equal
  // to 0.
  symbolic::Polynomial p_;

  // A program which stores the psd variables and constraints associated to λ
  // and ν.
  solvers::MathematicalProgram psatz_variables_and_psd_constraints_;

  // the symbolic start of the free line
  drake::VectorX<drake::symbolic::Variable> s0_;
  // the symbolic end of the free line
  drake::VectorX<drake::symbolic::Variable> s1_;

 private:  // Methods
  /**
   * Adds the constraints that p_(s0,s1, mu) = 0 to the program prog as well as
   * the decision variables associated to λ and ν
   */
  std::vector<solvers::Binding<solvers::LinearEqualityConstraint>>
  AddPsatzConstraintToProg(solvers::MathematicalProgram* prog,
                           const Eigen::Ref<const Eigen::VectorXd>& s0,
                           const Eigen::Ref<const Eigen::VectorXd>& s1) const;
};

/**
 * This class is designed to certify lines in the configuration space
 * parametrized as μ*s₀ + (1−μ)*s₁ where μ is an indeterminate and s₀ and s₁ are
 * parameters specifying the end points of the line. This is a special case of
 * certifying a 1-dimensional polytope but we provide this implementation to
 * enable certain optimizations.
 */
class CspaceFreeLine : public CspaceFreeRegion {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CspaceFreeLine)

  CspaceFreeLine(const systems::Diagram<double>& diagram,
                 const multibody::MultibodyPlant<double>* plant,
                 const geometry::SceneGraph<double>* scene_graph,
                 SeparatingPlaneOrder plane_order,
                 std::optional<Eigen::VectorXd> q_star,
                 const FilteredCollisionPairs& filtered_collision_pairs = {},
                 const VerificationOption& option = {});
  /**
   * Generate all the tuples for certification of lines
   * @param q_star: the point around which the stereographic projection will be
   * taken.
   * @param filtered_collision_pairs: pairs of geometries which are filtered out
   * from the collision checking.
   * @param tuples: a vector to fill with CspaceLineTuples.
   * @param separating_plane_vars: All of the variables in the separating
   * plane aᵀx + b = 0.
   * @param separating_plane_to_tuples: alternation_tuples can be grouped
   * based on the separating planes. separating_plane_to_tuples[i] are the
   * indices in alternation_tuples such that these tuples are all for
   * this->separating_planes()[i].
   * @param separating_plane_to_lorentz_cone_constraints: If the collision
   * geometry is not a polyhedron, but instead ellipsoid, capsules or cylinders,
   * we will also impose Lorentz cone constraints.
   */
  void GenerateTuplesForCertification(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      std::list<CspaceLineTuple>* tuples,
      VectorX<symbolic::Variable>* separating_plane_vars,
      std::vector<std::vector<int>>* separating_plane_to_tuples,
      std::vector<
          std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>*
          separating_plane_to_lorentz_cone_constraints) const;

  const symbolic::Variable& get_mu() const { return mu_; }
  const drake::VectorX<drake::symbolic::Variable>& get_s0() const {
    return s0_;
  }
  const drake::VectorX<drake::symbolic::Variable>& get_s1() const {
    return s1_;
  }
  const std::list<CspaceLineTuple>* get_tuples() const { return &tuples_; }
  const VectorX<symbolic::Variable>& get_separating_plane_vars() const {
    return separating_plane_vars_;
  }

  std::vector<LinkOnPlaneSideRational> GenerateRationalsForLinkOnOneSideOfPlane(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs) const override;

  /**
   * Certifies whether the line μ*s₀ + (1−μ)*s₁ is collision free. If the return
   * is true, a formal proof of non-collision is generated and populates
   * separating_planes_sol. If the result is false there may be no collisions
   * but this fact cannot be certified with the constructed SOS program.
   * @param s0
   * @param s1
   * @param solver_options
   * @return
   */
  bool CertifyTangentConfigurationSpaceLine(
      const Eigen::Ref<const Eigen::VectorXd>& s0,
      const Eigen::Ref<const Eigen::VectorXd>& s1,
      const solvers::SolverOptions& solver_options,
      std::vector<SeparatingPlane<double>>* separating_planes_sol) const;
  /**
   * Certifies whether a set of lines is collision free in parallel.
   */
  std::vector<bool> CertifyTangentConfigurationSpaceLines(
      const Eigen::Ref<const Eigen::MatrixXd>& s0,
      const Eigen::Ref<const Eigen::MatrixXd>& s1,
      const solvers::SolverOptions& solver_options,
      std::vector<std::vector<SeparatingPlane<double>>>*
          separating_planes_sol_per_row) const;

 protected:
  /**
   * Adds the constraint that all of the tuples in the @param i separating plane
   * are on the appropriate side of the plane to @param prog.
   * Assumes that @param prog contains the separating plane variables already
   * and has μ set as the indeterminate already.
   */
  void AddCertifySeparatingPlaneConstraintToProg(
      solvers::MathematicalProgram* prog, int i,
      const Eigen::Ref<const Eigen::VectorXd>& s0,
      const Eigen::Ref<const Eigen::VectorXd>& s1) const;

  bool PointInJointLimits(const Eigen::Ref<const Eigen::VectorXd>& s) const;

 private:
  // the variable of the line going from 0 to 1
  const symbolic::Variable mu_;
  // the symbolic start of the free line
  const drake::VectorX<drake::symbolic::Variable> s0_;
  // the symbolic end of the free line
  const drake::VectorX<drake::symbolic::Variable> s1_;

  // q_star_ for stereographic projection
  Eigen::VectorXd q_star_;

  FilteredCollisionPairs filtered_collision_pairs_;

  VerificationOption option_;

  std::list<CspaceLineTuple> tuples_;
  std::mutex update_bindings_mutex_;

  VectorX<symbolic::Variable> separating_plane_vars_;
  /*
   * tuples can be grouped
   * based on the separating planes. separating_plane_to_tuples[i] are the
   * indices in alternation_tuples such that these tuples are all for
   * this->separating_planes()[i].
   */
  std::vector<std::vector<int>> separating_plane_to_tuples_;
  std::vector<std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>
      separating_plane_to_lorentz_cone_constraints_;
};

}  // namespace multibody
}  // namespace drake
