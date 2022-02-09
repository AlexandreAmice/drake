#pragma once

#include <memory>
#include <vector>

#include "drake/geometry/optimization/cartesian_product.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/hyperellipsoid.h"
#include "drake/geometry/optimization/minkowski_sum.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/constraint.h"
#include "drake/systems/framework/context.h"

namespace drake {
namespace multibody {
using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::Vector3d;
using Eigen::VectorXd;
using geometry::optimization::ConvexSet;
using math::RigidTransform;
using multibody::Body;
using multibody::Frame;
using multibody::JacobianWrtVariable;
using multibody::MultibodyPlant;
using symbolic::Expression;
using systems::Context;

// copied from geometry/optimization/iris
// Takes q, p_AA, and p_BB and enforces that p_WA == p_WB.
class SamePointConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SamePointConstraint)

  SamePointConstraint(const MultibodyPlant<double>* plant,
                      const Context<double>& context)
      : solvers::Constraint(3, plant ? plant->num_positions() + 6 : 0,
                            Vector3d::Zero(), Vector3d::Zero()),
        plant_(plant),
        context_(plant->CreateDefaultContext()) {
    DRAKE_DEMAND(plant_ != nullptr);
    context_->SetTimeStateAndParametersFrom(context);
  }

  ~SamePointConstraint() override {}

  void set_frameA(const multibody::Frame<double>* frame) { frameA_ = frame; }

  void set_frameB(const multibody::Frame<double>* frame) { frameB_ = frame; }

  void EnableSymbolic() {
    if (symbolic_plant_ != nullptr) {
      return;
    }
    symbolic_plant_ = systems::System<double>::ToSymbolic(*plant_);
    symbolic_context_ = symbolic_plant_->CreateDefaultContext();
    symbolic_context_->SetTimeStateAndParametersFrom(*context_);
  }

 private:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override {
    DRAKE_DEMAND(frameA_ != nullptr);
    DRAKE_DEMAND(frameB_ != nullptr);
    VectorXd q = x.head(plant_->num_positions());
    Vector3d p_AA = x.template segment<3>(plant_->num_positions()),
             p_BB = x.template tail<3>();
    Vector3d p_WA, p_WB;
    plant_->SetPositions(context_.get(), q);
    plant_->CalcPointsPositions(*context_, *frameA_, p_AA,
                                plant_->world_frame(), &p_WA);
    plant_->CalcPointsPositions(*context_, *frameB_, p_BB,
                                plant_->world_frame(), &p_WB);
    *y = p_WA - p_WB;
  }

  // p_WA = X_WA(q)*p_AA
  // dp_WA = Jq_v_WA*dq + X_WA(q)*dp_AA
  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override {
    DRAKE_DEMAND(frameA_ != nullptr);
    DRAKE_DEMAND(frameB_ != nullptr);
    VectorX<AutoDiffXd> q = x.head(plant_->num_positions());
    Vector3<AutoDiffXd> p_AA = x.template segment<3>(plant_->num_positions()),
                        p_BB = x.template tail<3>();
    plant_->SetPositions(context_.get(), ExtractDoubleOrThrow(q));
    const RigidTransform<double>& X_WA =
        plant_->EvalBodyPoseInWorld(*context_, frameA_->body());
    const RigidTransform<double>& X_WB =
        plant_->EvalBodyPoseInWorld(*context_, frameB_->body());
    Eigen::Matrix3Xd Jq_v_WA(3, plant_->num_positions()),
        Jq_v_WB(3, plant_->num_positions());
    plant_->CalcJacobianTranslationalVelocity(
        *context_, JacobianWrtVariable::kQDot, *frameA_,
        ExtractDoubleOrThrow(p_AA), plant_->world_frame(),
        plant_->world_frame(), &Jq_v_WA);
    plant_->CalcJacobianTranslationalVelocity(
        *context_, JacobianWrtVariable::kQDot, *frameB_,
        ExtractDoubleOrThrow(p_BB), plant_->world_frame(),
        plant_->world_frame(), &Jq_v_WB);

    *y = X_WA.cast<AutoDiffXd>() * p_AA - X_WB.cast<AutoDiffXd>() * p_BB;
    // Now add it the dydq terms.  We don't use the standard autodiff tools
    // because these only impact a subset of the autodiff derivatives.
    for (int i = 0; i < 3; i++) {
      (*y)[i].derivatives().head(plant_->num_positions()) +=
          (Jq_v_WA.row(i) - Jq_v_WB.row(i)).transpose();
    }
  }

  void DoEval(const Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override {
    DRAKE_DEMAND(symbolic_plant_ != nullptr);
    DRAKE_DEMAND(frameA_ != nullptr);
    DRAKE_DEMAND(frameB_ != nullptr);
    const Frame<Expression>& frameA =
        symbolic_plant_->get_frame(frameA_->index());
    const Frame<Expression>& frameB =
        symbolic_plant_->get_frame(frameB_->index());
    VectorX<Expression> q = x.head(plant_->num_positions());
    Vector3<Expression> p_AA = x.template segment<3>(plant_->num_positions()),
                        p_BB = x.template tail<3>();
    Vector3<Expression> p_WA, p_WB;
    symbolic_plant_->SetPositions(symbolic_context_.get(), q);
    symbolic_plant_->CalcPointsPositions(*symbolic_context_, frameA, p_AA,
                                         symbolic_plant_->world_frame(), &p_WA);
    symbolic_plant_->CalcPointsPositions(*symbolic_context_, frameB, p_BB,
                                         symbolic_plant_->world_frame(), &p_WB);
    *y = p_WA - p_WB;
  }

 protected:
  const MultibodyPlant<double>* const plant_;
  const multibody::Frame<double>* frameA_{nullptr};
  const multibody::Frame<double>* frameB_{nullptr};
  std::unique_ptr<Context<double>> context_;

  std::unique_ptr<MultibodyPlant<Expression>> symbolic_plant_{nullptr};
  std::unique_ptr<Context<Expression>> symbolic_context_{nullptr};
};

// takes t, p_AA, and p_BB and enforces that p_WA == p_WB
class SamePointConstraintRational : public SamePointConstraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SamePointConstraintRational)

  SamePointConstraintRational(const multibody::RationalForwardKinematics*
                                  rational_forward_kinematics_ptr,
                              const Eigen::Ref<const Eigen::VectorXd>& q_star,
                              const Context<double>& context)
      : SamePointConstraint(&rational_forward_kinematics_ptr->plant(), context),
        rational_forward_kinematics_ptr_(rational_forward_kinematics_ptr),
        q_star_(q_star) {}

  ~SamePointConstraintRational() override {}

 private:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override {
    DRAKE_DEMAND(frameA_ != nullptr);
    DRAKE_DEMAND(frameB_ != nullptr);
    VectorXd t = x.head(plant_->num_positions());
    VectorXd q = rational_forward_kinematics_ptr_->ComputeQValue(t, q_star_);
    Vector3d p_AA = x.template segment<3>(plant_->num_positions()),
             p_BB = x.template tail<3>();
    Vector3d p_WA, p_WB;
    plant_->SetPositions(context_.get(), q);
    plant_->CalcPointsPositions(*context_, *frameA_, p_AA,
                                plant_->world_frame(), &p_WA);
    plant_->CalcPointsPositions(*context_, *frameB_, p_BB,
                                plant_->world_frame(), &p_WB);
    *y = p_WA - p_WB;
  }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override {
    DRAKE_DEMAND(frameA_ != nullptr);
    DRAKE_DEMAND(frameB_ != nullptr);
    VectorX<AutoDiffXd> t = x.head(plant_->num_positions());
    VectorX<AutoDiffXd> q =
        rational_forward_kinematics_ptr_->ComputeQValue(t, q_star_);

    Vector3<AutoDiffXd> p_AA = x.template segment<3>(plant_->num_positions()),
                        p_BB = x.template tail<3>();
    plant_->SetPositions(context_.get(), ExtractDoubleOrThrow(q));
    const RigidTransform<double>& X_WA =
        plant_->EvalBodyPoseInWorld(*context_, frameA_->body());
    const RigidTransform<double>& X_WB =
        plant_->EvalBodyPoseInWorld(*context_, frameB_->body());
    Eigen::Matrix3Xd Jq_v_WA(3, plant_->num_positions()),
        Jq_v_WB(3, plant_->num_positions());
    plant_->CalcJacobianTranslationalVelocity(
        *context_, JacobianWrtVariable::kQDot, *frameA_,
        ExtractDoubleOrThrow(p_AA), plant_->world_frame(),
        plant_->world_frame(), &Jq_v_WA);
    plant_->CalcJacobianTranslationalVelocity(
        *context_, JacobianWrtVariable::kQDot, *frameB_,
        ExtractDoubleOrThrow(p_BB), plant_->world_frame(),
        plant_->world_frame(), &Jq_v_WB);
    Eigen::Matrix3Xd Jt_v_WA(3, plant_->num_positions()),
        Jt_v_WB(3, plant_->num_positions());
    for (int i = 0; i < plant_->num_positions(); i++) {
      // dX_t_wa = J_q_WA * dq_dt
      Jt_v_WA.col(i) = Jq_v_WA.col(i) * q(i).derivatives()(i);
      Jt_v_WB.col(i) = Jq_v_WB.col(i) * q(i).derivatives()(i);
    }

    *y = X_WA.cast<AutoDiffXd>() * p_AA - X_WB.cast<AutoDiffXd>() * p_BB;
    // Now add it the dydq terms.  We don't use the standard autodiff tools
    // because these only impact a subset of the autodiff derivatives.
    for (int i = 0; i < 3; i++) {
      (*y)[i].derivatives().head(plant_->num_positions()) +=
          (Jt_v_WA.row(i) - Jt_v_WB.row(i)).transpose();
    }
  }

  void DoEval(const Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override {
    DRAKE_DEMAND(symbolic_plant_ != nullptr);
    DRAKE_DEMAND(frameA_ != nullptr);
    DRAKE_DEMAND(frameB_ != nullptr);
    const Frame<Expression>& frameA =
        symbolic_plant_->get_frame(frameA_->index());
    const Frame<Expression>& frameB =
        symbolic_plant_->get_frame(frameB_->index());
    VectorX<Expression> t = x.head(plant_->num_positions());
    VectorX<Expression> q =
        rational_forward_kinematics_ptr_->ComputeQValue(t, q_star_);
    Vector3<Expression> p_AA = x.template segment<3>(plant_->num_positions()),
                        p_BB = x.template tail<3>();
    Vector3<Expression> p_WA, p_WB;
    symbolic_plant_->SetPositions(symbolic_context_.get(), q);
    symbolic_plant_->CalcPointsPositions(*symbolic_context_, frameA, p_AA,
                                         symbolic_plant_->world_frame(), &p_WA);
    symbolic_plant_->CalcPointsPositions(*symbolic_context_, frameB, p_BB,
                                         symbolic_plant_->world_frame(), &p_WB);
    *y = p_WA - p_WB;
  }

 protected:
  const multibody::RationalForwardKinematics* rational_forward_kinematics_ptr_;
  const Eigen::VectorXd q_star_;
};

class IrisConvexSetMaker final : public geometry::ShapeReifier {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(IrisConvexSetMaker)

  IrisConvexSetMaker(const geometry::QueryObject<double>& query,
                     std::optional<geometry::FrameId> reference_frame)
      : query_{query}, reference_frame_{reference_frame} {};

  void set_reference_frame(const geometry::FrameId& reference_frame) {
    DRAKE_DEMAND(reference_frame.is_valid());
    *reference_frame_ = reference_frame;
  }

  void set_geometry_id(const geometry::GeometryId& geom_id) {
    geom_id_ = geom_id;
  }

  using ShapeReifier::ImplementGeometry;

  void ImplementGeometry(const geometry::Sphere&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<geometry::optimization::Hyperellipsoid>(
        query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const geometry::Cylinder&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<geometry::optimization::CartesianProduct>(
        query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const geometry::HalfSpace&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<geometry::optimization::HPolyhedron>(
        query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const geometry::Box&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    // Note: We choose HPolyhedron over VPolytope here, but the IRIS paper
    // discusses a significant performance improvement using a "least-distance
    // programming" instance from CVXGEN that exploited the VPolytope
    // representation.  So we may wish to revisit this.
    set = std::make_unique<geometry::optimization::HPolyhedron>(
        query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const geometry::Capsule&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<geometry::optimization::MinkowskiSum>(
        query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const geometry::Ellipsoid&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<geometry::optimization::Hyperellipsoid>(
        query_, geom_id_, reference_frame_);
  }

 private:
  const geometry::QueryObject<double>& query_{};
  std::optional<geometry::FrameId> reference_frame_{};
  geometry::GeometryId geom_id_{};
};

/**
 * find the max epsilon such that c_cost^T * x <= d_cost + epsilon is redundant
 * for the polytope defined by C_constraint * t <= d_constraint
 * @param c_cost
 * @param d_cost
 * @param C_constraint
 * @param d_constraint
 * @param t_lower_limits
 * @param t_upper_limits
 * @return
 */
std::optional<double> FindMaxEpsTilRedundant(
    const Eigen::Ref<const Eigen::VectorXd>& c_cost, const double d_cost,
    const Eigen::Ref<const Eigen::MatrixXd>& C_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& d_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits);

Eigen::VectorXd FindMaxEpsTilRedundantAllIneqs(
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits);

std::optional<double> FindMaxEpsTilCollisionForIneqForCollisionPair(
    std::shared_ptr<SamePointConstraintRational> same_point_constraint,
    const multibody::Frame<double>& frameA,
    const multibody::Frame<double>& frameB, const ConvexSet& setA,
    const ConvexSet& setB, const Eigen::Ref<const Eigen::VectorXd>& c_cost,
    const double d_cost, const Eigen::Ref<const Eigen::MatrixXd>& C_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& d_constraint,
//    const double eps_min,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
    const solvers::SolverInterface& non_linear_solver,
    const Eigen::Ref<const Eigen::VectorXd>& t_sample);

std::optional<double> FindMaxEpsTilCollisionForIneq(
    const std::set<std::pair<geometry::GeometryId, geometry::GeometryId>>&
        pairs,
    const std::unordered_map<geometry::GeometryId,
                             const multibody::Frame<double>*>& frames,
    const std::unordered_map<geometry::GeometryId,
                             copyable_unique_ptr<ConvexSet>>& sets,
    const multibody::RationalForwardKinematics& rational_forward_kinematics,
    const systems::Context<double>& context,
    Eigen::Ref<const Eigen::VectorXd>& q_star,
    const Eigen::Ref<const Eigen::VectorXd>& c_cost, const double d_cost,
    const Eigen::Ref<const Eigen::MatrixXd>& C_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& d_constraint,
//    const double eps_min,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
    const solvers::SolverInterface& solver,
    const Eigen::Ref<const Eigen::VectorXd>& t_sample);


Eigen::VectorXd FindEpsTilCollisionOrRedundantForAllIneqs(
    const multibody::MultibodyPlant<double>& plant,
    const systems::Context<double>& context,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
//    const Eigen::Ref<const Eigen::VectorXd>& eps_min,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_sample);

/**
 * Max epsilon such that C(t-t_center) <= eps.cwiseProduct(d) containts
 * collision
 * @param plant
 * @param context
 * @param q_star
 * @param C
 * @param d
 * @param t_center
 * @param t_lower_limits
 * @param t_upper_limits
 * @return
 */
Eigen::VectorXd FindMaxEpsScalingForAllIneqs(
    const multibody::MultibodyPlant<double>& plant,
    const systems::Context<double>& context,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& t_center,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits);

Eigen::VectorXd FindMaxEpsScalingForAllIneqsForCollisionPair(
    std::shared_ptr<SamePointConstraintRational> same_point_constraint,
    const multibody::Frame<double>& frameA,
    const multibody::Frame<double>& frameB, const ConvexSet& setA,
    const ConvexSet& setB, const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& t_center,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits);

/**
 * remove ith row of A
 * @param A
 * @param i
 * @return
 */
Eigen::MatrixXd RemoveMatrixRow(const Eigen::MatrixXd A, int i);


}  // namespace multibody
}  // namespace drake
