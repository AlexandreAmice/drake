#pragma once

#include <unordered_map>
#include <vector>

#include "drake/common/symbolic/expression.h"
#include "drake/common/symbolic/polynomial.h"
#include "drake/common/symbolic/rational_function.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/multibody_tree.h"
#include "drake/multibody/tree/prismatic_mobilizer.h"
#include "drake/multibody/tree/quaternion_floating_mobilizer.h"
#include "drake/multibody/tree/revolute_mobilizer.h"
#include "drake/multibody/tree/space_xyz_mobilizer.h"
#include "drake/multibody/tree/weld_mobilizer.h"

namespace drake {
namespace multibody {
namespace rational_old {
/**
 * We can represent the pose (position, orientation) of each link, as rational
 * functions, namely n(t) / d(t) where both the numerator n(t) and denominator
 * d(t) are polynomials of t, and t is some variable related to the generalized
 * position.
 *
 * One example is that for a rotation matrix with angle θ and axis a, the
 * rotation matrix can be written as I + sinθ A + (1-cosθ) A², where A is the
 * skew-symmetric matrix from axis a. We can use the half-angle formulat to
 * substitute the trigonometric function sinθ and cosθ as
 * cosθ = cos(θ*+Δθ) = cosθ*cosΔθ - sinθ*sinΔθ
 *      = (1-t²)/(1+t²) cosθ*- 2t/(1+t²) sinθ*     (1)
 * sinθ = sin(θ*+Δθ) = sinθ*cosΔθ - cosθ*sinΔθ
 *      = (1-t²)/(1+t²) sinθ*- 2t/(1+t²) cosθ*     (2)
 * where θ = θ*+Δθ, and t = tan(Δθ/2). θ* is some given angle.
 * With (1) and (2), both sinθ and cosθ are written as a rational function of t.
 * Thus the rotation matrix can be written as rational functions of t.
 */
class RationalForwardKinematicsOld {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RationalForwardKinematicsOld)

  template <typename T>
  struct Pose {
    //    drake::Vector3<T> p_AB;
    Eigen::Matrix<T, 3, 1> p_AB;
    //    drake::Matrix3<T> R_AB;
    Eigen::Matrix<T, 3, 3> R_AB;

    drake::multibody::BodyIndex frame_A_index;
    drake::math::RigidTransform<drake::symbolic::Expression>
    asRigidTransformExpression() const {
      Eigen::Matrix<drake::symbolic::Expression, 3, 3> R_AB_expr;
      R_AB_expr << R_AB.coeff(0, 0).ToExpression(),
          R_AB.coeff(0, 1).ToExpression(), R_AB.coeff(0, 2).ToExpression(),
          R_AB.coeff(1, 0).ToExpression(), R_AB.coeff(1, 1).ToExpression(),
          R_AB.coeff(1, 2).ToExpression(), R_AB.coeff(2, 0).ToExpression(),
          R_AB.coeff(2, 1).ToExpression(), R_AB.coeff(2, 2).ToExpression();
      math::RotationMatrix<drake::symbolic::Expression> R(R_AB_expr);

      Vector3<drake::symbolic::Expression> p{p_AB.coeff(0).ToExpression(),
                                             p_AB.coeff(1).ToExpression(),
                                             p_AB.coeff(2).ToExpression()};
      return drake::math::RigidTransform(R, p);
    }
  };

  struct LinkPoints {
    LinkPoints(int m_link_index,
               const Eigen::Ref<const Eigen::Matrix3Xd>& m_p_BQ)
        : link_index{m_link_index}, p_BQ{m_p_BQ} {}
    int link_index;
    // The position of the points Q in the link frame B.
    Eigen::Matrix3Xd p_BQ;
  };

  explicit RationalForwardKinematicsOld(
      const drake::multibody::MultibodyPlant<double>& plant);

  /** Compute the pose of each link as fractional functions of t.
   * We will set up the indeterminates t also.
   * A revolute joint requires a single t, where t = tan(Δθ/2).
   * A prismatic joint requires a single t, where t = Δd, d being the
   * translational motion of the prismatic joint.
   * A free-floating joint requires 12 t, 3 for position, and 9 for the rotation
   * matrix.
   * A gimbal joint requires 9 t, for the rotation matrix.
   * @param q_star The nominal posture around which we compute the link poses.
   * @param expressed_body_index The pose of each link is expressed in this
   * body.
   */
  std::vector<Pose<drake::symbolic::RationalFunction>> CalcLinkPoses(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      drake::multibody::BodyIndex expressed_body_index) const;

  std::vector<Pose<drake::symbolic::Polynomial>>
  CalcLinkPosesAsMultilinearPolynomials(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      drake::multibody::BodyIndex expressed_body_index) const;

  /**
   * Compute the pose X_AB as a multilinear polynomial function.
   * @param q_star The nominal posture
   * @param link_index Frame B, the link whose pose is computed.
   * @param expressed_body_index Frame A, the link in whose frame the pose is
   * expressed.
   */
  Pose<drake::symbolic::Polynomial> CalcLinkPoseAsMultilinearPolynomial(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      drake::multibody::BodyIndex link_index,
      drake::multibody::BodyIndex expressed_body_index) const;

  drake::symbolic::RationalFunction
  ConvertMultilinearPolynomialToRationalFunction(
      const drake::symbolic::Polynomial& e) const;

  const drake::multibody::MultibodyPlant<double>& plant() const {
    return plant_;
  }

  const drake::VectorX<drake::symbolic::Variable>& t() const { return t_; }

  /** Each t(i) is associated with a mobilizer.
   */
  const std::unordered_map<drake::symbolic::Variable::Id,
                           drake::multibody::internal::MobilizerIndex>&
  map_t_to_mobilizer() const {
    return map_t_to_mobilizer_;
  }

  /**
   * compute t = tan((q_val - q_star_val / 2)), while handling the index
   * matching between q and t. If @p clamp_angle = true, then t = infinity if
   * q_val >= q_star_val + pi, and t = -infinity if q_val <= q_star_val - pi.
   */
  Eigen::VectorXd ComputeTValue(
      const Eigen::Ref<const Eigen::VectorXd>& q_val,
      const Eigen::Ref<const Eigen::VectorXd>& q_star_val,
      bool clamp_angle = false) const;

  /**
   * compute t = tan((q_val - q_star_val / 2)), while handling the index
   * matching between q and t. If @p clamp_angle = true, then t = infinity if
   * q_val >= q_star_val + pi, and t = -infinity if q_val <= q_star_val - pi.
   */
  VectorX<symbolic::Expression> ComputeTValue(
      const Eigen::Ref<const VectorX<symbolic::Expression>>& q_val,
      const Eigen::Ref<const Eigen::VectorXd>& q_star_val,
      bool clamp_angle = false) const;

  /**
   * compute q = arctan2(2*t/(1+t**2), (1-t**2)/(1+t**2)) + q_star, while
   * handling the index matching between q and t.
   */
  Eigen::VectorXd ComputeQValue(
      const Eigen::Ref<const Eigen::VectorXd>& t_val,
      const Eigen::Ref<const Eigen::VectorXd>& q_star_val) const;

  /**
   * compute q = arctan2(2*t/(1+t**2), (1-t**2)/(1+t**2)) + q_star, while
   * handling the index matching between q and t.
   */
  VectorX<symbolic::Expression> ComputeQValue(
      const Eigen::Ref<const VectorX<symbolic::Expression>>& t_val,
      const Eigen::Ref<const Eigen::VectorXd>& q_star_val) const;

  /**
   * compute q = arctan2(2*t/(1+t**2), (1-t**2)/(1+t**2)) + q_star, while
   * handling the index matching between q and t.
   */
  VectorX<AutoDiffXd> ComputeQValue(
      const Eigen::Ref<const VectorX<AutoDiffXd>>& t_val,
      const Eigen::Ref<const Eigen::VectorXd>& q_star_val) const;

  // Return the indeterminates t on the path from start to the end.
  drake::VectorX<drake::symbolic::Variable> FindTOnPath(
      drake::multibody::BodyIndex start, drake::multibody::BodyIndex end) const;

  const std::unordered_map<drake::symbolic::Variable::Id, int>& t_id_to_index()
      const {
    return t_id_to_index_;
  }

 private:
  // Compute the pose of the link, connected to its parent link through a
  // revolute joint.
  // We will first compute the link pose as multilinear polynomials, with
  // indeterminates cos_delta and sin_delta, representing cos(Δθ) and sin(Δθ)
  // respectively. We will then replace cos_delta and sin_delta in the link
  // pose with rational functions (1-t^2)/(1+t^2) and 2t/(1+t^2) respectively.
  // The reason why we don't use RationalFunction directly, is that currently
  // our rational function can't find the common factor in the denominator,
  // namely the sum between rational functions p1(x) / (q1(x) * r(x)) + p2(x) /
  // r(x) is computed as (p1(x) * r(x) + p2(x) * q1(x) * r(x)) / (q1(x) * r(x) *
  // r(x)), without handling the common factor r(x) in the denominator.
  template <typename T>
  void CalcLinkPoseAsMultilinearPolynomialWithRevoluteJoint(
      const Eigen::Ref<const Eigen::Vector3d>& axis_F,
      const drake::math::RigidTransformd& X_PF,
      const drake::math::RigidTransformd& X_MC, const Pose<T>& X_AP,
      double theta_star, const drake::symbolic::Variable& cos_delta,
      const drake::symbolic::Variable& sin_delta, Pose<T>* X_AC) const;

  // Compute the pose of the link, connected to its parent link through a
  // weld joint.
  template <typename T>
  void CalcLinkPoseWithWeldJoint(const drake::math::RigidTransformd& X_FM,
                                 const drake::math::RigidTransformd& X_PF,
                                 const drake::math::RigidTransformd& X_MC,
                                 const Pose<T>& X_AP, Pose<T>* X_AC) const;

  void CalcReshuffledChildLinkPoseAsMultilinearPolynomial(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      drake::multibody::BodyIndex reshuffled_parent,
      drake::multibody::BodyIndex reshuffled_child,
      const Pose<drake::symbolic::Polynomial>& X_AP,
      Pose<drake::symbolic::Polynomial>* X_AC) const;

  const drake::multibody::MultibodyPlant<double>& plant_;
  // The variables used in computing the pose as rational functions. t_ are the
  // indeterminates in the rational functions.
  drake::VectorX<drake::symbolic::Variable> t_;
  // Each t(i) is associated with a mobilizer.
  std::unordered_map<drake::symbolic::Variable::Id,
                     drake::multibody::internal::MobilizerIndex>
      map_t_to_mobilizer_;
  // Given a mobilizer, returns the index of the mobilizer's slack variable in
  // t_.
  std::unordered_map<drake::multibody::internal::MobilizerIndex, int>
      map_mobilizer_to_t_index_;

  // The variables used to represent tan(θ / 2).
  drake::VectorX<drake::symbolic::Variable> t_angles_;
  drake::VectorX<drake::symbolic::Polynomial> one_plus_t_angles_squared_;
  drake::VectorX<drake::symbolic::Polynomial> two_t_angles_;
  drake::VectorX<drake::symbolic::Polynomial> one_minus_t_angles_squared_;

  drake::VectorX<drake::symbolic::Variable> cos_delta_;
  drake::VectorX<drake::symbolic::Variable> sin_delta_;
  // t could contain both prismatic t and angle t.
  // t_angles_[map_t_index_to_angle_index_[i]] = t_[i]
  std::unordered_map<int, int> map_t_index_to_angle_index_;
  std::unordered_map<int, int> map_angle_index_to_t_index_;
  // t_id_to_index_(t_(i).get_id()) = i
  std::unordered_map<drake::symbolic::Variable::Id, int> t_id_to_index_;
  drake::symbolic::Variables t_variables_;
};

/** If e is a multilinear polynomial of cos_delta and sin_delta, and no
 * cos_delta(i) and sin_delta(i) appear in the same monomial, then we replace
 * cos_delta(i) with (1-t_angles(i)^2)/(1+t_angles(i)^2), and sin_delta(i)
 * with 2t_angles(i)/(1+t_angles(i)^2), and get a rational polynomial of t.
 */
void ReplaceCosAndSinWithRationalFunction(
    const drake::symbolic::Polynomial& e,
    const drake::VectorX<drake::symbolic::Variable>& cos_delta,
    const drake::VectorX<drake::symbolic::Variable>& sin_delta,
    const drake::VectorX<drake::symbolic::Variable>& t_angles,
    const drake::symbolic::Variables& t,
    drake::symbolic::RationalFunction* e_rational);

}  // namespace rational_old
}  // namespace multibody
}  // namespace drake
