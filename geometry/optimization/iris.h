#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "drake/common/drake_deprecated.h"
#include "drake/common/symbolic.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"

namespace drake {
namespace geometry {
namespace optimization {

using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::Vector3d;
using Eigen::VectorXd;
using math::RigidTransform;
using multibody::Body;
using multibody::Frame;
using multibody::JacobianWrtVariable;
using multibody::MultibodyPlant;
using symbolic::Expression;
using systems::Context;

/** Configuration options for the IRIS algorithm.

@ingroup geometry_optimization
*/
struct IrisOptions {
  IrisOptions() = default;

  /** The initial polytope is guaranteed to contain the point if that point is
  collision-free. However, the IRIS alternation objectives do not include (and
  can not easily include) a constraint that the original sample point is
  contained. Therefore, the IRIS paper recommends that if containment is a
  requirement, then the algorithm should simply terminate early if alternations
  would ever cause the set to not contain the point. */
  bool require_sample_point_is_contained{false};

  /** Maximum number of iterations. */
  int iteration_limit{100};

  /** IRIS will terminate if the change in the *volume* of the hyperellipsoid
  between iterations is less that this threshold. This termination condition can
  be disabled by setting to a negative value. */
  double termination_threshold{2e-2};  // from rdeits/iris-distro.

  /** IRIS will terminate if the change in the *volume* of the hyperellipsoid
  between iterations is less that this percent of the previouse best volume.
  This termination condition can be disabled by setting to a negative value. */
  double relative_termination_threshold{1e-3};  // from rdeits/iris-distro.

  // TODO(russt): Improve the implementation so that we can clearly document the
  // units for this margin.
  /** For IRIS in configuration space, we retreat by this margin from each
  C-space obstacle in order to avoid the possibility of requiring an infinite
  number of faces to approximate a curved boundary.
  */
  double configuration_space_margin{1e-2};

  /** For IRIS in configuration space, we use IbexSolver to rigorously confirm
  that regions are collision-free. This step may be computationally
  demanding, so we allow it to be disabled for a faster algorithm for obtaining
  regions without the rigorous guarantee. */
  bool enable_ibex = true;

  /** Maximum number of faces added per collision pair, per iteration. Setting
     this option to -1 imposes no limit. Default value is -1. */
  int max_faces_per_collision_pair{-1};
};

struct IrisOptionsRationalSpace : public IrisOptions {
  IrisOptionsRationalSpace() = default;

  /** For IRIS in rational configuration space, we can certify that the regions
   * are truly collision free using SOS programming and the methods in
   * multibody/rational_forward_kinematics. Turning this option on
   * certifies the regions everytime a set of hyperplanes is added
   * TODO(Alex.Amice) support turning this option on */
  bool certify_region_with_sos_during_generation = false;

  /** For IRIS in rational configuration space, we can certify that the regions
   * are truly collision free using SOS programming and the methods in
   * multibody/rational_forward_kinematics. Turning this option on
   * certifies the regions at the end of the loop
   * TODO(Alex.Amice) support turning this option on */
  bool certify_region_with_sos_after_generation = false;

  /** For IRIS in rational configuration space we need a point around which to
   * perform the stereographic projection
   * */
  std::optional<Eigen::VectorXd> q_star;
};


// Takes q, p_AA, and p_BB and enforces that p_WA == p_WB.
class SamePointConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SamePointConstraint)

  SamePointConstraint(const MultibodyPlant<double>* plant,
                      const Context<double>& context)
      : solvers::Constraint(3, plant->num_positions() + 6,
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
    const Eigen::Vector3d y_val =
        X_WA * math::ExtractValue(p_AA) - X_WB * math::ExtractValue(p_BB);
    Eigen::Matrix3Xd dy(3, plant_->num_positions() + 6);
    dy << Jq_v_WA - Jq_v_WB, X_WA.rotation().matrix(),
        -X_WB.rotation().matrix();
    *y = math::InitializeAutoDiff(y_val, dy * math::ExtractGradient(x));
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

//    *y = X_WA.cast<AutoDiffXd>() * p_AA - X_WB.cast<AutoDiffXd>() * p_BB;
//    // Now add it the dydq terms.  We don't use the standard autodiff tools
//    // because these only impact a subset of the autodiff derivatives.
//    for (int i = 0; i < 3; i++) {
//      (*y)[i].derivatives().head(plant_->num_positions()) +=
//          (Jt_v_WA.row(i) - Jt_v_WB.row(i)).transpose();
//    }
    const Eigen::Vector3d y_val =
        X_WA * math::ExtractValue(p_AA) - X_WB * math::ExtractValue(p_BB);
    Eigen::Matrix3Xd dy(3, plant_->num_positions() + 6);
    dy << Jt_v_WA - Jt_v_WB, X_WA.rotation().matrix(),
        -X_WB.rotation().matrix();
    *y = math::InitializeAutoDiff(y_val, dy * math::ExtractGradient(x));
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

/** The IRIS (Iterative Region Inflation by Semidefinite programming) algorithm,
as described in

R. L. H. Deits and R. Tedrake, “Computing large convex regions of obstacle-free
space through semidefinite programming,” Workshop on the Algorithmic
Fundamentals of Robotics, Istanbul, Aug. 2014.
http://groups.csail.mit.edu/robotics-center/public_papers/Deits14.pdf

This algorithm attempts to locally maximize the volume of a convex polytope
representing obstacle-free space given a sample point and list of convex
obstacles. Rather than compute the volume of the polytope directly, the
algorithm maximizes the volume of an inscribed ellipsoid. It alternates between
finding separating hyperplanes between the ellipsoid and the obstacles and then
finding a new maximum-volume inscribed ellipsoid.

@param obstacles is a vector of convex sets representing the occupied space.
@param sample provides a point in the space; the algorithm is initialized using
a tiny sphere around this point. The algorithm is only guaranteed to succeed if
this sample point is collision free (outside of all obstacles), but in practice
the algorithm can often escape bad initialization (assuming the
require_sample_point_is_contained option is false).
@param domain describes the total region of interest; computed IRIS regions will
be inside this domain.  It must be bounded, and is typically a simple bounding
box (e.g. from HPolyhedron::MakeBox).

The @p obstacles, @p sample, and the @p domain must describe elements in the
same ambient dimension (but that dimension can be any positive integer).

@ingroup geometry_optimization
*/
HPolyhedron Iris(const ConvexSets& obstacles,
                 const Eigen::Ref<const Eigen::VectorXd>& sample,
                 const HPolyhedron& domain,
                 const IrisOptions& options = IrisOptions());

/** Constructs ConvexSet representations of obstacles for IRIS in 3D using the
geometry from a SceneGraph QueryObject. All geometry in the scene with a
proximity role, both anchored and dynamic, are consider to be *fixed* obstacles
frozen in the poses captured in the context used to create the QueryObject.

When multiple representations are available for a particular geometry (e.g. a
Box can be represented as either an HPolyhedron or a VPolytope), then this
method will prioritize the representation that we expect is most performant for
the current implementation of the IRIS algorithm.

@ingroup geometry_optimization
*/
ConvexSets MakeIrisObstacles(
    const QueryObject<double>& query_object,
    std::optional<FrameId> reference_frame = std::nullopt);

/** A variation of the Iris (Iterative Region Inflation by Semidefinite
programming) algorithm which finds collision-free regions in the *configuration
space* of @p plant.  @see Iris for details on the original algorithm.
The possibility of this configuration-space variant was suggested in the
original IRIS paper, but substantial new ideas have been employed here to
address the non-convexity of configuration-space obstacles; these will be
documented in a forth-coming publication.

@param plant describes the kinematics of configuration space.  It must be
connected to a SceneGraph in a systems::Diagram.
@param context is a context of the @p plant. The context must have the positions
of the plant set to the initialIRIS seed configuration.
@param options provides additional configuration options.  In particular,
`options.enabled_ibex` may have a significant impact on the runtime of the
algorithm.

@ingroup geometry_optimization
*/
HPolyhedron IrisInConfigurationSpace(
    const multibody::MultibodyPlant<double>& plant,
    const systems::Context<double>& context,
    const IrisOptions& options = IrisOptions());

/** A deprecated variation of the IrisInConfigurationSpace method where the
initial Iris seed configuration is provided explicitly instead of via the
context.

@ingroup geometry_optimization
*/
DRAKE_DEPRECATED("2022-03-01",
                 "Use IrisInConfigurationSpace() with sample set in context.")
HPolyhedron IrisInConfigurationSpace(
    const multibody::MultibodyPlant<double>& plant,
    const systems::Context<double>& context,
    const Eigen::Ref<const Eigen::VectorXd>& sample,
    const IrisOptions& options = IrisOptions());

/** A variation of the Iris (Iterative Region Inflation by Semidefinite
programming) algorithm which finds collision-free regions in the *rational
parametrization of the configuration space* of @p plant. @see Iris for details
on the original algorithm. This is a reimplementation of
IrisInConfigurationSpace for the rational reparametrization

@param plant describes the kinematics of configuration space.  It must be
connected to a SceneGraph in a systems::Diagram.
@param context is a context of the @p plant. The context must have the positions
of the plant set to the initial IRIS seed configuration.
@param options provides additional configuration options.  In particular,
`options.certify_region_during_generation` vs
`options.certify_region_after_generation' can have an impact on computation time
@param starting_hpolyhedron is an optional argument to constrain the initial
iris search. This defaults to the joint limits of the plants, but if there is a
reason to constrain it further this option is provided.
@ingroup geometry_optimization
*/
HPolyhedron IrisInRationalConfigurationSpace(
    const multibody::MultibodyPlant<double>& plant,
    const systems::Context<double>& context,
    const IrisOptionsRationalSpace& options = IrisOptionsRationalSpace(),
    const std::optional<HPolyhedron>& starting_hpolyhedron = std::nullopt);

/**
 * Internal method for actually running Iris. Be assume that the HPolyhedron in
 * *P_ptr has been set to a finite bounding box defining the joint limits of the
 * plant.
 * @param P_ptr must have its ambient dimension be the same size as the number
 * of joints in the plant.
 * @param E_ptr must have its ambient dimension be the same size as the nubmer
 * of joints in the plant.
 * @return
 */
void _DoIris_(const multibody::MultibodyPlant<double>& plant,
              const systems::Context<double>& context,
              const IrisOptions& options,
              const Eigen::Ref<const Eigen::VectorXd>& sample,
              const std::shared_ptr<SamePointConstraint>& same_point_constraint,
              HPolyhedron* P_ptr, Hyperellipsoid* E_ptr);

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
