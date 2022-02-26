#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"

#include <limits>
#include <queue>
#include <set>

#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"

namespace drake {
namespace multibody {
using drake::Isometry3;
using drake::Matrix3;
using drake::Vector3;
using drake::VectorX;
using drake::multibody::BodyIndex;
using drake::multibody::Frame;
using drake::multibody::MultibodyPlant;
using drake::multibody::internal::BodyTopology;
using drake::multibody::internal::GetInternalTree;
using drake::multibody::internal::Mobilizer;
using drake::multibody::internal::MobilizerIndex;
using drake::multibody::internal::MultibodyTree;
using drake::multibody::internal::PrismaticMobilizer;
using drake::multibody::internal::QuaternionFloatingMobilizer;
using drake::multibody::internal::RevoluteMobilizer;
using drake::multibody::internal::SpaceXYZMobilizer;
using drake::multibody::internal::WeldMobilizer;
using drake::symbolic::Expression;
using drake::symbolic::Polynomial;
using drake::symbolic::RationalFunction;

bool CheckPolynomialIndeterminatesAreCosSinDelta(
    const Polynomial& e_poly,
    const VectorX<drake::symbolic::Variable>& cos_delta,
    const VectorX<drake::symbolic::Variable>& sin_delta) {
  VectorX<drake::symbolic::Variable> cos_sin_delta(cos_delta.rows() +
                                                   sin_delta.rows());
  cos_sin_delta << cos_delta, sin_delta;
  const drake::symbolic::Variables cos_sin_delta_variables(cos_sin_delta);
  return e_poly.indeterminates().IsSubsetOf(cos_sin_delta_variables);
}

void ReplaceCosAndSinWithRationalFunction(
    const drake::symbolic::Polynomial& e_poly,
    const VectorX<drake::symbolic::Variable>& cos_delta,
    const VectorX<drake::symbolic::Variable>& sin_delta,
    const VectorX<drake::symbolic::Variable>& t_angle,
    const drake::symbolic::Variables& t,
    const VectorX<drake::symbolic::Polynomial>& one_plus_t_angles_squared,
    const VectorX<drake::symbolic::Polynomial>& two_t_angles,
    const VectorX<drake::symbolic::Polynomial>& one_minus_t_angles_squared,
    drake::symbolic::RationalFunction* e_rational) {
  DRAKE_DEMAND(cos_delta.rows() == sin_delta.rows());
  DRAKE_DEMAND(cos_delta.rows() == t_angle.rows());
  DRAKE_DEMAND(CheckPolynomialIndeterminatesAreCosSinDelta(e_poly, cos_delta,
                                                           sin_delta));
  // First find the angles whose cos or sin appear in the polynomial. This
  // will determine the denominator of the rational function.
  std::set<int> angle_indices;
  for (const auto& pair : e_poly.monomial_to_coefficient_map()) {
    // Also check that this monomial can't contain both cos_delta(i) and
    // sin_delta(i).
    for (int i = 0; i < cos_delta.rows(); ++i) {
      const int angle_degree =
          pair.first.degree(cos_delta(i)) + pair.first.degree(sin_delta(i));
      DRAKE_DEMAND(angle_degree <= 1);
      if (angle_degree == 1) {
        angle_indices.insert(i);
      }
    }
  }
  if (angle_indices.empty()) {
    *e_rational = RationalFunction(Polynomial(e_poly.ToExpression(), t));
    return;
  }
  const drake::symbolic::Monomial monomial_one{};
  drake::symbolic::Polynomial denominator{1};
  for (int angle_index : angle_indices) {
    // denominator *= (1 + t_angle(angle_index)^2)
    denominator *= one_plus_t_angles_squared[angle_index];
  }
  drake::symbolic::Polynomial numerator{};

  for (const auto& [monomial, coeff] : e_poly.monomial_to_coefficient_map()) {
    // If the monomial contains cos_delta(i), then replace cos_delta(i) with
    // 1 - t_angle(i) * t_angle(i).
    // If the monomial contains sin_delta(i), then replace sin_delta(i) with
    // 2 * t_angle(i).
    // Otherwise, multiplies with 1 + t_angle(i) * t_angle(i)

    // The coefficient could contain "t", (the indeterminates for e are
    // cos_delta and sin_delta). Hence we first need to write the coefficient
    // as a polynomial of indeterminates interset(t, coeff.variables()).
    drake::symbolic::Polynomial numerator_term(
        coeff, symbolic::intersect(t, coeff.GetVariables()));
    for (int angle_index : angle_indices) {
      if (monomial.degree(cos_delta(angle_index)) > 0) {
        numerator_term *= one_minus_t_angles_squared[angle_index];
      } else if (monomial.degree(sin_delta(angle_index)) > 0) {
        numerator_term *= two_t_angles[angle_index];
      } else {
        numerator_term *= one_plus_t_angles_squared[angle_index];
      }
    }
    numerator += numerator_term;
  }

  *e_rational = RationalFunction(numerator, denominator);
}

void ReplaceCosAndSinWithRationalFunction(
    const drake::symbolic::Polynomial& e_poly,
    const VectorX<drake::symbolic::Variable>& cos_delta,
    const VectorX<drake::symbolic::Variable>& sin_delta,
    const VectorX<drake::symbolic::Variable>& t_angle,
    const drake::symbolic::Variables& t,
    drake::symbolic::RationalFunction* e_rational) {
  const drake::symbolic::Monomial monomial_one{};
  VectorX<Polynomial> one_minus_t_square(t_angle.rows());
  VectorX<Polynomial> two_t(t_angle.rows());
  VectorX<Polynomial> one_plus_t_square(t_angle.rows());
  for (int i = 0; i < t_angle.rows(); ++i) {
    one_minus_t_square[i] = Polynomial(
        {{monomial_one, 1}, {drake::symbolic::Monomial(t_angle(i), 2), -1}});
    two_t[i] = Polynomial({{drake::symbolic::Monomial(t_angle(i), 1), 2}});
    one_plus_t_square[i] = Polynomial(
        {{monomial_one, 1}, {drake::symbolic::Monomial(t_angle(i), 2), 1}});
  }
  ReplaceCosAndSinWithRationalFunction(e_poly, cos_delta, sin_delta, t_angle, t,
                                       one_plus_t_square, two_t,
                                       one_minus_t_square, e_rational);
}

RationalForwardKinematics::RationalForwardKinematics(
    const MultibodyPlant<double>& plant)
    : plant_(plant) {
  int num_t = 0;
  const auto& tree = GetInternalTree(plant_);
  for (BodyIndex body_index(1); body_index < plant_.num_bodies();
       ++body_index) {
    const auto& body_topology = tree.get_topology().get_body(body_index);
    const auto mobilizer =
        &(tree.get_mobilizer(body_topology.inboard_mobilizer));
    if (dynamic_cast<const RevoluteMobilizer<double>*>(mobilizer) != nullptr) {
      const drake::symbolic::Variable t_angle("t[" + std::to_string(num_t) +
                                              "]");
      t_.conservativeResize(t_.rows() + 1);
      t_angles_.conservativeResize(t_angles_.rows() + 1);
      cos_delta_.conservativeResize(cos_delta_.rows() + 1);
      sin_delta_.conservativeResize(sin_delta_.rows() + 1);
      t_(t_.rows() - 1) = t_angle;
      t_angles_(t_angles_.rows() - 1) = t_angle;
      cos_delta_(cos_delta_.rows() - 1) = drake::symbolic::Variable(
          "cos_delta[" + std::to_string(cos_delta_.rows() - 1) + "]");
      sin_delta_(sin_delta_.rows() - 1) = drake::symbolic::Variable(
          "sin_delta[" + std::to_string(cos_delta_.rows() - 1) + "]");
      num_t += 1;
      map_t_index_to_angle_index_.emplace(t_.rows() - 1, t_angles_.rows() - 1);
      map_angle_index_to_t_index_.emplace(t_angles_.rows() - 1, t_.rows() - 1);
      map_t_to_mobilizer_.emplace(t_(t_.rows() - 1).get_id(),
                                  mobilizer->index());
      map_mobilizer_to_t_index_.emplace(mobilizer->index(), t_.rows() - 1);
      t_id_to_index_.emplace(t_angle.get_id(), t_.rows() - 1);
    } else if (dynamic_cast<const WeldMobilizer<double>*>(mobilizer) !=
               nullptr) {
    } else if (dynamic_cast<const SpaceXYZMobilizer<double>*>(mobilizer) !=
               nullptr) {
      throw std::runtime_error("Gimbal joint has not been handled yet.");
    } else if (dynamic_cast<const PrismaticMobilizer<double>*>(mobilizer) !=
               nullptr) {
      throw std::runtime_error("Prismatic joint has not been handled yet.");
    }
  }
  const drake::symbolic::Monomial monomial_one{};
  one_plus_t_angles_squared_.resize(t_angles_.rows());
  two_t_angles_.resize(t_angles_.rows());
  one_minus_t_angles_squared_.resize(t_angles_.rows());
  for (int i = 0; i < t_angles_.rows(); ++i) {
    one_minus_t_angles_squared_(i) = Polynomial(
        {{monomial_one, 1}, {drake::symbolic::Monomial(t_angles_(i), 2), -1}});
    two_t_angles_(i) =
        Polynomial({{drake::symbolic::Monomial(t_angles_(i), 1), 2}});
    one_plus_t_angles_squared_(i) = Polynomial(
        {{monomial_one, 1}, {drake::symbolic::Monomial(t_angles_(i), 2), 1}});
  }
  t_variables_ = drake::symbolic::Variables(t_);
}

template <typename Scalar1, typename Scalar2>
void CalcChildPose(const Matrix3<Scalar2>& R_WP, const Vector3<Scalar2>& p_WP,
                   const drake::math::RigidTransform<double>& X_PF,
                   const drake::math::RigidTransform<double>& X_MC,
                   const Matrix3<Scalar1>& R_FM, const Vector3<Scalar1>& p_FM,
                   Matrix3<Scalar2>* R_WC, Vector3<Scalar2>* p_WC) {
  // Frame F is the inboard frame (attached to the parent link), and frame
  // M is the outboard frame (attached to the child link).
  const Matrix3<Scalar2> R_WF = R_WP * X_PF.rotation().matrix();
  const Vector3<Scalar2> p_WF = R_WP * X_PF.translation() + p_WP;
  const Matrix3<Scalar2> R_WM = R_WF * R_FM;
  const Vector3<Scalar2> p_WM = R_WF * p_FM + p_WF;
  const Matrix3<double> R_MC = X_MC.rotation().matrix();
  const Vector3<double> p_MC = X_MC.translation();
  *R_WC = R_WM * R_MC;
  *p_WC = R_WM * p_MC + p_WM;
}

template <typename T>
void RationalForwardKinematics::
    CalcLinkPoseAsMultilinearPolynomialWithRevoluteJoint(
        const Eigen::Ref<const Eigen::Vector3d>& axis_F,
        const drake::math::RigidTransformd& X_PF,
        const drake::math::RigidTransformd& X_MC, const Pose<T>& X_AP,
        double theta_star, const drake::symbolic::Variable& cos_delta_theta,
        const drake::symbolic::Variable& sin_delta_theta, Pose<T>* X_AC) const {
  // clang-format off
      const Eigen::Matrix3d A_F =
          (Eigen::Matrix3d() << 0, -axis_F(2), axis_F(1),
                                axis_F(2), 0, -axis_F(0),
                                -axis_F(1), axis_F(0), 0).finished();
  // clang-format on
  const drake::symbolic::Variables cos_sin_delta(
      {cos_delta_theta, sin_delta_theta});
  const double cos_theta_star = cos(theta_star);
  const double sin_theta_star = sin(theta_star);
  const Polynomial cos_angle(
      {{drake::symbolic::Monomial(cos_delta_theta, 1), cos_theta_star},
       {drake::symbolic::Monomial(sin_delta_theta, 1), -sin_theta_star}});
  const Polynomial sin_angle(
      {{drake::symbolic::Monomial(cos_delta_theta, 1), sin_theta_star},
       {drake::symbolic::Monomial(sin_delta_theta, 1), cos_theta_star}});
  // Frame F is the inboard frame (attached to the parent link), and frame
  // M is the outboard frame (attached to the child link).
  const Matrix3<drake::symbolic::Polynomial> R_FM =
      Eigen::Matrix3d::Identity() + sin_angle * A_F +
      (1 - cos_angle) * A_F * A_F;
  const drake::symbolic::Polynomial poly_zero{};
  const Vector3<drake::symbolic::Polynomial> p_FM(poly_zero, poly_zero,
                                                  poly_zero);
  CalcChildPose(X_AP.R_AB, X_AP.p_AB, X_PF, X_MC, R_FM, p_FM, &(X_AC->R_AB),
                &(X_AC->p_AB));
  X_AC->frame_A_index = X_AP.frame_A_index;
}

template <typename T>
void RationalForwardKinematics::CalcLinkPoseWithWeldJoint(
    const drake::math::RigidTransformd& X_FM,
    const drake::math::RigidTransformd& X_PF,
    const drake::math::RigidTransformd& X_MC, const Pose<T>& X_AP,
    Pose<T>* X_AC) const {
  const Matrix3<double> R_FM = X_FM.rotation().matrix();
  const Vector3<double> p_FM = X_FM.translation();
  CalcChildPose(X_AP.R_AB, X_AP.p_AB, X_PF, X_MC, R_FM, p_FM, &(X_AC->R_AB),
                &(X_AC->p_AB));
  X_AC->frame_A_index = X_AP.frame_A_index;
}

void SetPoseToIdentity(
    RationalForwardKinematics::Pose<drake::symbolic::Polynomial>* pose) {
  const Polynomial poly_zero{};
  const Polynomial poly_one{1};
  // clang-format off
  pose->R_AB <<
    poly_one, poly_zero, poly_zero,
    poly_zero, poly_one, poly_zero,
    poly_zero, poly_zero, poly_one;
  pose->p_AB << poly_zero, poly_zero, poly_zero;
  // clang-format on
}

void RationalForwardKinematics::
    CalcReshuffledChildLinkPoseAsMultilinearPolynomial(
        const Eigen::Ref<const Eigen::VectorXd>& q_star,
        BodyIndex reshuffled_parent, BodyIndex reshuffled_child,
        const RationalForwardKinematics::Pose<drake::symbolic::Polynomial>&
            X_AP,
        RationalForwardKinematics::Pose<drake::symbolic::Polynomial>* X_AC)
        const {
  // if reshuffled_child was a child of reshuffled_parent in the
  // original tree before reshuffling, then is_order_reversed = false;
  // otherwise it is true.
  // If we denote the frames related to the two adjacent bodies connected
  // by a mobilizer in the original tree as P->F->M->C, then after reversing
  // the order, the new frames should reverse the order, namely
  // P' = C, F' = M, M' = F, C' = P, and hence we know that
  // X_P'F' = X_MC.inverse()
  // X_F'M' = X_FM.inverse()
  // X_M'C' = X_PF.inverse()
  const MultibodyTree<double>& tree = GetInternalTree(plant_);
  const BodyTopology& reshuffled_parent_topology =
      tree.get_topology().get_body(reshuffled_parent);
  const BodyTopology& reshuffled_child_topology =
      tree.get_topology().get_body(reshuffled_child);
  MobilizerIndex mobilizer_index;
  bool is_order_reversed;
  if (reshuffled_parent_topology.parent_body.is_valid() &&
      reshuffled_parent_topology.parent_body == reshuffled_child) {
    is_order_reversed = true;
    mobilizer_index = reshuffled_parent_topology.inboard_mobilizer;
  } else if (reshuffled_child_topology.parent_body.is_valid() &&
             reshuffled_child_topology.parent_body == reshuffled_parent) {
    is_order_reversed = false;
    mobilizer_index = reshuffled_child_topology.inboard_mobilizer;
  } else {
    throw std::invalid_argument(
        "CalcReshuffledChildLinkPoseAsMultilinearPolynomial: reshuffled_parent "
        "is not a parent nor a child of reshuffled_child.");
  }
  const Mobilizer<double>* mobilizer = &(tree.get_mobilizer(mobilizer_index));
  if (dynamic_cast<const RevoluteMobilizer<double>*>(mobilizer) != nullptr) {
    // A revolute joint.
    const RevoluteMobilizer<double>* revolute_mobilizer =
        dynamic_cast<const RevoluteMobilizer<double>*>(mobilizer);
    const int t_index = map_mobilizer_to_t_index_.at(mobilizer->index());
    const int q_index = revolute_mobilizer->position_start_in_q();
    const int t_angle_index = map_t_index_to_angle_index_.at(t_index);
    Eigen::Vector3d axis_F;
    drake::math::RigidTransformd X_PF, X_MC;
    if (!is_order_reversed) {
      axis_F = revolute_mobilizer->revolute_axis();
      const Frame<double>& frame_F = mobilizer->inboard_frame();
      const Frame<double>& frame_M = mobilizer->outboard_frame();
      X_PF = frame_F.GetFixedPoseInBodyFrame();
      X_MC = frame_M.GetFixedPoseInBodyFrame();
    } else {
      // By negating the revolute axis, we know that R(a, θ)⁻¹ = R(-a, θ)
      axis_F = -revolute_mobilizer->revolute_axis();
      X_PF = mobilizer->outboard_frame().GetFixedPoseInBodyFrame().inverse();
      X_MC = mobilizer->inboard_frame().GetFixedPoseInBodyFrame().inverse();
    }
    CalcLinkPoseAsMultilinearPolynomialWithRevoluteJoint(
        axis_F, X_PF, X_MC, X_AP, q_star(q_index), cos_delta_(t_angle_index),
        sin_delta_(t_angle_index), X_AC);
  } else if (dynamic_cast<const PrismaticMobilizer<double>*>(mobilizer) !=
             nullptr) {
    throw std::runtime_error(
        "RationalForwardKinematics: prismatic joint is not supported yet.");
  } else if (dynamic_cast<const WeldMobilizer<double>*>(mobilizer) != nullptr) {
    const WeldMobilizer<double>* weld_mobilizer =
        dynamic_cast<const WeldMobilizer<double>*>(mobilizer);
    drake::math::RigidTransformd X_FM, X_PF, X_MC;
    if (!is_order_reversed) {
      X_FM = weld_mobilizer->get_X_FM();
      X_PF = mobilizer->inboard_frame().GetFixedPoseInBodyFrame();
      X_MC = mobilizer->outboard_frame().GetFixedPoseInBodyFrame();
    } else {
      X_FM = weld_mobilizer->get_X_FM().inverse();
      X_PF = mobilizer->outboard_frame().GetFixedPoseInBodyFrame().inverse();
      X_MC = mobilizer->inboard_frame().GetFixedPoseInBodyFrame().inverse();
    }
    CalcLinkPoseWithWeldJoint(X_FM, X_PF, X_MC, X_AP, X_AC);
  } else if (dynamic_cast<const SpaceXYZMobilizer<double>*>(mobilizer) !=
             nullptr) {
    throw std::runtime_error("Gimbal joint has not been handled yet.");
  } else if (dynamic_cast<const QuaternionFloatingMobilizer<double>*>(
                 mobilizer) != nullptr) {
    throw std::runtime_error("Free floating joint has not been handled yet.");
  } else {
    throw std::runtime_error(
        "RationalForwardKinematics: Can't handle this mobilizer.");
  }
}

std::vector<RationalForwardKinematics::Pose<Polynomial>>
RationalForwardKinematics::CalcLinkPosesAsMultilinearPolynomials(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    BodyIndex expressed_body_index) const {
  // We need to change the frame in which the link pose is expressed. To do so,
  // we will reshuffle the tree structure in the MultibodyPlant, namely if we
  // express the pose in link A's frame, we will treat A as the root of the
  // reshuffled tree. From link A, we propogate to its child links and so on, so
  // as to compute the pose of all links in A's frame.
  std::vector<RationalForwardKinematics::Pose<Polynomial>> poses_poly(
      plant_.num_bodies());
  SetPoseToIdentity(&(poses_poly[expressed_body_index]));
  poses_poly[expressed_body_index].frame_A_index = expressed_body_index;
  // In the reshuffled tree, the expressed body is the root. We will compute the
  // pose of each link w.r.t this root link.
  internal::ReshuffledBody reshuffled_expressed_body(expressed_body_index,
                                                     nullptr, nullptr);
  internal::ReshuffleKinematicsTree(plant_, &reshuffled_expressed_body);
  // Now do a breadth-first-search on this reshuffled tree, to compute the pose
  // of each link w.r.t the root.
  std::queue<internal::ReshuffledBody*> bfs_queue;
  bfs_queue.push(&reshuffled_expressed_body);
  while (!bfs_queue.empty()) {
    const internal::ReshuffledBody* reshuffled_body = bfs_queue.front();
    if (reshuffled_body->parent != nullptr) {
      CalcReshuffledChildLinkPoseAsMultilinearPolynomial(
          q_star, reshuffled_body->parent->body_index,
          reshuffled_body->body_index,
          poses_poly[reshuffled_body->parent->body_index],
          &(poses_poly[reshuffled_body->body_index]));
      poses_poly[reshuffled_body->body_index].frame_A_index =
          expressed_body_index;
    }
    bfs_queue.pop();
    for (const auto& reshuffled_child : reshuffled_body->children) {
      bfs_queue.push(reshuffled_child.get());
    }
  }
  return poses_poly;
}

RationalForwardKinematics::Pose<drake::symbolic::Polynomial>
RationalForwardKinematics::CalcLinkPoseAsMultilinearPolynomial(
    const Eigen::Ref<const Eigen::VectorXd>& q_star, BodyIndex link_index,
    BodyIndex expressed_body_index) const {
  // First find the path from expressed_body_index to link_index.
  const std::vector<BodyIndex> path =
      internal::FindShortestPath(plant_, expressed_body_index, link_index);
  std::vector<RationalForwardKinematics::Pose<drake::symbolic::Polynomial>>
      poses(path.size());
  SetPoseToIdentity(&(poses[0]));
  poses[0].frame_A_index = expressed_body_index;
  for (int i = 1; i < static_cast<int>(path.size()); ++i) {
    CalcReshuffledChildLinkPoseAsMultilinearPolynomial(
        q_star, path[i - 1], path[i], poses[i - 1], &(poses[i]));
    poses[i].frame_A_index = expressed_body_index;
  }
  return poses[poses.size() - 1];
}

RationalFunction
RationalForwardKinematics::ConvertMultilinearPolynomialToRationalFunction(
    const drake::symbolic::Polynomial& e) const {
  RationalFunction e_rational;
  ReplaceCosAndSinWithRationalFunction(
      e, cos_delta_, sin_delta_, t_angles_, t_variables_,
      one_plus_t_angles_squared_, two_t_angles_, one_minus_t_angles_squared_,
      &e_rational);
  return e_rational;
}

std::vector<RationalForwardKinematics::Pose<RationalFunction>>
RationalForwardKinematics::CalcLinkPoses(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    BodyIndex expressed_body_index) const {
  // We will first compute the link pose as multilinear polynomials, with
  // indeterminates cos_delta and sin_delta, representing cos(Δθ) and
  // sin(Δθ)
  // respectively. We will then replace cos_delta and sin_delta in the link
  // pose with rational functions (1-t^2)/(1+t^2) and 2t/(1+t^2)
  // respectively.
  // The reason why we don't use RationalFunction directly, is that
  // currently
  // our rational function can't find the common factor in the denominator,
  // namely the sum between rational functions p1(x) / (q1(x) * r(x)) +
  // p2(x) /
  // r(x) is computed as (p1(x) * r(x) + p2(x) * q1(x) * r(x)) / (q1(x) *
  // r(x) *
  // r(x)), without handling the common factor r(x) in the denominator.
  const RationalFunction rational_zero(0);
  const RationalFunction rational_one(1);
  std::vector<Pose<RationalFunction>> poses(plant_.num_bodies());
  // We denote the expressed body frame as A.
  poses[expressed_body_index].p_AB << rational_zero, rational_zero,
      rational_zero;
  // clang-format off
  poses[expressed_body_index].R_AB <<
    rational_one, rational_zero, rational_zero,
    rational_zero, rational_one, rational_zero,
    rational_zero, rational_zero, rational_one;
  // clang-format on
  poses[expressed_body_index].frame_A_index = expressed_body_index;
  std::vector<Pose<Polynomial>> poses_poly =
      CalcLinkPosesAsMultilinearPolynomials(q_star, expressed_body_index);
  for (BodyIndex body_index{0}; body_index < plant_.num_bodies();
       ++body_index) {
    // Now convert the multilinear polynomial of cos and sin to rational
    // function of t.
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        poses[body_index].R_AB(i, j) =
            ConvertMultilinearPolynomialToRationalFunction(
                poses_poly[body_index].R_AB(i, j));
      }
      poses[body_index].p_AB(i) =
          ConvertMultilinearPolynomialToRationalFunction(
              poses_poly[body_index].p_AB(i));
      poses[body_index].frame_A_index = expressed_body_index;
    }
  }
  return poses;
}

Eigen::VectorXd RationalForwardKinematics::ComputeTValue(
    const Eigen::Ref<const Eigen::VectorXd>& q_val,
    const Eigen::Ref<const Eigen::VectorXd>& q_star_val,
    bool clamp_angle) const {
  Eigen::VectorXd t_val(t_.size());
  const double kInf = std::numeric_limits<double>::infinity();
  for (int i = 0; i < t_val.size(); ++i) {
    const Mobilizer<double>& mobilizer = GetInternalTree(plant_).get_mobilizer(
        map_t_to_mobilizer_.at(t_(i).get_id()));
    if (dynamic_cast<const RevoluteMobilizer<double>*>(&mobilizer) != nullptr) {
      const int q_index = mobilizer.position_start_in_q();
      t_val(i) = std::tan((q_val(q_index) - q_star_val(q_index)) / 2);
      if (clamp_angle) {
        if (q_val(q_index) - q_star_val(q_index) >= M_PI) {
          t_val(i) = kInf;
        } else if (q_val(q_index) - q_star_val(q_index) <= -M_PI) {
          t_val(i) = -kInf;
        }
      }
    } else {
      throw std::runtime_error("Other joint types are not supported yet.");
    }
  }
  return t_val;
}

VectorX<Expression> RationalForwardKinematics::ComputeTValue(
    const Eigen::Ref<const VectorX<symbolic::Expression>>& q_val,
    const Eigen::Ref<const Eigen::VectorXd>& q_star_val,
    bool clamp_angle) const {
  VectorX<Expression> t_val(t_.size());
  const double kInf = std::numeric_limits<double>::infinity();
  for (int i = 0; i < t_val.size(); ++i) {
    const Mobilizer<double>& mobilizer = GetInternalTree(plant_).get_mobilizer(
        map_t_to_mobilizer_.at(t_(i).get_id()));
    if (dynamic_cast<const RevoluteMobilizer<double>*>(&mobilizer) != nullptr) {
      const int q_index = mobilizer.position_start_in_q();
      t_val(i) = symbolic::tan((q_val(q_index) - q_star_val(q_index)) / 2);
      if (clamp_angle) {
        if (q_val(q_index) - q_star_val(q_index) >= M_PI) {
          t_val(i) = kInf;
        } else if (q_val(q_index) - q_star_val(q_index) <= -M_PI) {
          t_val(i) = -kInf;
        }
      }
    } else {
      throw std::runtime_error("Other joint types are not supported yet.");
    }
  }
  return t_val;
}

// TODO(hongkai.dai) templatize this code
Eigen::VectorXd RationalForwardKinematics::ComputeQValue(
    const Eigen::Ref<const Eigen::VectorXd>& t_val,
    const Eigen::Ref<const Eigen::VectorXd>& q_star_val) const {
  Eigen::VectorXd q_val(t_.size());
  for (int i = 0; i < t_val.size(); ++i) {
    const Mobilizer<double>& mobilizer = GetInternalTree(plant_).get_mobilizer(
        map_t_to_mobilizer_.at(t_(i).get_id()));
    if (dynamic_cast<const RevoluteMobilizer<double>*>(&mobilizer) != nullptr) {
      const int q_index = mobilizer.position_start_in_q();
      q_val(q_index) = std::atan2(2 * t_val(i) / (1 + std::pow(t_val(i), 2)),
                                  (1 - std::pow(t_val(i), 2)) /
                                      (1 + std::pow(t_val(i), 2))) +
                       q_star_val(q_index);
    } else {
      throw std::runtime_error("Other joint types are not supported yet.");
    }
  }
  return q_val;
}

VectorX<symbolic::Expression> RationalForwardKinematics::ComputeQValue(
    const Eigen::Ref<const VectorX<symbolic::Expression>>& t_val,
    const Eigen::Ref<const Eigen::VectorXd>& q_star_val) const {
  VectorX<symbolic::Expression> q_val(t_.size());
  for (int i = 0; i < t_val.size(); ++i) {
    const Mobilizer<double>& mobilizer = GetInternalTree(plant_).get_mobilizer(
        map_t_to_mobilizer_.at(t_(i).get_id()));
    if (dynamic_cast<const RevoluteMobilizer<double>*>(&mobilizer) != nullptr) {
      const int q_index = mobilizer.position_start_in_q();
      q_val(q_index) =
          symbolic::atan2(2 * t_val(i) / (1 + symbolic::pow(t_val(i), 2)),
                          (1 - symbolic::pow(t_val(i), 2)) /
                              (1 + symbolic::pow(t_val(i), 2))) +
          q_star_val(q_index);
    } else {
      throw std::runtime_error("Other joint types are not supported yet.");
    }
  }
  return q_val;
}

VectorX<AutoDiffXd> RationalForwardKinematics::ComputeQValue(
    const Eigen::Ref<const VectorX<AutoDiffXd>>& t_val,
    const Eigen::Ref<const Eigen::VectorXd>& q_star_val) const {
  VectorX<AutoDiffXd> q_val(t_.size());
  for (int i = 0; i < t_val.size(); ++i) {
    const Mobilizer<double>& mobilizer = GetInternalTree(plant_).get_mobilizer(
        map_t_to_mobilizer_.at(t_(i).get_id()));
    if (dynamic_cast<const RevoluteMobilizer<double>*>(&mobilizer) != nullptr) {
      const int q_index = mobilizer.position_start_in_q();
      q_val(q_index) = atan2(2 * t_val(i) / (1 + pow(t_val(i), 2)),
                             (1 - pow(t_val(i), 2)) / (1 + pow(t_val(i), 2))) +
                       q_star_val(q_index);
    } else {
      throw std::runtime_error("Other joint types are not supported yet.");
    }
  }
  return q_val;
}

VectorX<drake::symbolic::Variable> RationalForwardKinematics::FindTOnPath(
    BodyIndex start, BodyIndex end) const {
  const std::vector<MobilizerIndex> mobilizers =
      internal::FindMobilizersOnShortestPath(plant_, start, end);
  VectorX<drake::symbolic::Variable> t_on_path;
  for (int i = 0; i < static_cast<int>(mobilizers.size()); ++i) {
    auto it = map_mobilizer_to_t_index_.find(mobilizers[i]);
    if (it != map_mobilizer_to_t_index_.end()) {
      t_on_path.conservativeResize(t_on_path.size() + 1);
      t_on_path(t_on_path.size() - 1) = t_(it->second);
    }
  }

  return t_on_path;
}
}  // namespace multibody
}  // namespace drake
