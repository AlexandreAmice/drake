#pragma once
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"

namespace drake {
namespace multibody {

namespace internal {
// Checks whether two polynomials are equal. Needed to build maps with
// Polynomials as keys in the AllocateProgram method
struct ComparePolynomials {
  bool operator()(const symbolic::Polynomial& p1,
                  const symbolic::Polynomial& p2) const {
    return p1.CoefficientsAlmostEqual(p2, 1E-12);
  }
};

// a class for storing the allocated certification program and constructing the
// final program
class AllocatedCertificationProgram {
 public:
  AllocatedCertificationProgram() = default;

  AllocatedCertificationProgram(
      std::unique_ptr<solvers::MathematicalProgram> prog,
      std::unordered_map<
          symbolic::Polynomial,
          std::unordered_map<
              symbolic::Monomial,
              solvers::Binding<solvers::LinearEqualityConstraint>>,
          std::hash<symbolic::Polynomial>, internal::ComparePolynomials>
          polynomial_to_monomial_to_bindings_map);

  void EvaluatePolynomialsAndUpdateProgram(symbolic::Environment env);

  solvers::MathematicalProgramResult solve(
      const solvers::SolverOptions& solver_options);

  const solvers::MathematicalProgram* get_prog() const { return prog_.get(); }
  const std::unordered_map<
      symbolic::Polynomial,
      std::unordered_map<symbolic::Monomial,
                         solvers::Binding<solvers::LinearEqualityConstraint>>,
      std::hash<symbolic::Polynomial>, internal::ComparePolynomials>
  get_polynomial_to_monomial_to_bindings_map() const {
    return polynomial_to_monomial_to_bindings_map_;
  }

 private:
  std::unique_ptr<solvers::MathematicalProgram> prog_;
  std::unordered_map<
      symbolic::Polynomial,
      std::unordered_map<symbolic::Monomial,
                         solvers::Binding<solvers::LinearEqualityConstraint>>,
      std::hash<symbolic::Polynomial>, internal::ComparePolynomials>
      polynomial_to_monomial_to_bindings_map_;
};
}  // namespace internal

/**
 * This class is designed to certify lines in the configuration space
 * parametrized as μ*s₀ + (1−μ)*s₁. This is a special case of certifying a
 * 1-dimensional polytope but we provide this implementation to enable certain
 * optimizations.
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

  void GenerateTuplesForCertification(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      std::vector<CspaceFreeRegion::CspacePolytopeTuple>* alternation_tuples,
      VectorX<symbolic::Variable>* lagrangian_gram_vars,
      VectorX<symbolic::Variable>* verified_gram_vars,
      VectorX<symbolic::Variable>* separating_plane_vars,
      std::vector<std::vector<int>>* separating_plane_to_tuples,
      std::vector<
          std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>*
          separating_plane_to_lorentz_cone_constraints) const;

  const symbolic::Variable get_mu() const { return mu_; }
  const drake::VectorX<drake::symbolic::Variable> get_s0() const { return s0_; }
  const drake::VectorX<drake::symbolic::Variable> get_s1() const { return s1_; }

  std::vector<LinkOnPlaneSideRational> GenerateRationalsForLinkOnOneSideOfPlane(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs) const override;

  /**
   * Certifies whether the line μ*s₀ + (1−μ)*s₁. If the return is true, a formal
   * proof of non-collision is generated. If the result is false there may be no
   * collisions but this fact cannot be certified with the constructed SOS
   * program.
   * @param s0
   * @param s1
   * @param solver_options
   * @return
   */
  bool CertifyTangentConfigurationSpaceLine(
      const Eigen::Ref<const Eigen::VectorXd>& s0,
      const Eigen::Ref<const Eigen::VectorXd>& s1,
      const solvers::SolverOptions& solver_options = solvers::SolverOptions());

 private:
  // the variable of the line going from 0 to 1
  const symbolic::Variable mu_;
  // the symbolic start of the free line
  drake::VectorX<drake::symbolic::Variable> s0_;
  // the symbolic end of the free line
  drake::VectorX<drake::symbolic::Variable> s1_;
  // map for performing substitution from t to μ*s₀ + (1−μ)*s₁
  std::unordered_map<symbolic::Variable, symbolic::Expression>
      t_to_line_subs_map_;
  // q_star_ for stereographic projection
  Eigen::VectorXd q_star_;

  FilteredCollisionPairs filtered_collision_pairs_;

  VerificationOption option_;

  // preallocated certification program
  internal::AllocatedCertificationProgram allocated_certification_program_;

  void InitializeLineVariables(int num_positions);

  internal::AllocatedCertificationProgram AllocateCertificationProgram() const;
};

}  // namespace multibody
}  // namespace drake
