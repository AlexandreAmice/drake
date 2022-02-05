//
// Created by amice on 2/4/22.
//

#include "drake/multibody/rational_forward_kinematics/t_in_collision_contraint.h"
#include "drake/solvers/choose_best_solver.h"

namespace drake {
namespace multibody {

using geometry::optimization::SamePointConstraintRational;
using geometry::optimization::ConvexSet;
using geometry::optimization::Hyperellipsoid;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;


/**
 * Find the maximum amount we can move the inequality c_cost^T t <= d_cost until we either collide between two collision
 * pairs or the inequality becomes redundant.
 * @param same_point_constraint: a same point constraint object
 * @param frameA:
 * @param frameB
 * @param setA
 * @param setB
 * @param c_cost
 * @param d_cost
 * @param C_constraint: C_constraint t <= d_constraint polytope we wish to stay within.
 * @param d_constraint
 * @param t_lower_limits: lower joint limits
 * @param t_upper_limits: upper joint limits
 * @param solver
 * @param t_guess: initial guess
 * @return
 */
std::optional<double> FindMaxEpsTilCollisionForIneqForCollisionPair(
    std::shared_ptr<SamePointConstraintRational> same_point_constraint,
    const multibody::Frame<double>& frameA,
    const multibody::Frame<double>& frameB, const ConvexSet& setA,
    const ConvexSet& setB,
    const Eigen::Ref<const Eigen::VectorXd>& c_cost,
    const double d_cost,
    const Eigen::Ref<const Eigen::MatrixXd>& C_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& d_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
    const solvers::SolverInterface& solver,
    const Eigen::Ref<const Eigen::VectorXd>& t_guess) {
  solvers::MathematicalProgram prog;
  auto t = prog.NewContinuousVariables(C_constraint.cols(), "t");

  prog.AddLinearConstraint(
      C_constraint, VectorXd::Constant(d_constraint.size(), -std::numeric_limits<double>::infinity()),
      d_constraint, t);
  prog.AddBoundingBoxConstraint(t_lower_limits, t_upper_limits, t);
  prog.AddLinearCost(c_cost, d_cost, t);

  auto p_AA = prog.NewContinuousVariables<3>("p_AA");
  auto p_BB = prog.NewContinuousVariables<3>("p_BB");
  setA.AddPointInSetConstraints(&prog, p_AA);
  setB.AddPointInSetConstraints(&prog, p_BB);

  same_point_constraint->set_frameA(&frameA);
  same_point_constraint->set_frameB(&frameB);
  prog.AddConstraint(same_point_constraint, {t, p_AA, p_BB});

  // Help nonlinear optimizers (e.g. SNOPT) avoid trivial local minima at the
  // origin.
  prog.SetInitialGuess(t, t_guess);
  prog.SetInitialGuess(p_AA, Vector3d::Constant(.01));
  prog.SetInitialGuess(p_BB, Vector3d::Constant(.01));


  solvers::MathematicalProgramResult result;
  solver.Solve(prog, std::nullopt, std::nullopt, &result);
  if (result.is_success()) {
    return result.get_optimal_cost();
  }
  return {};
}

std::optional<double> FindMaxEpsTilCollisionForIneq(
    std::shared_ptr<SamePointConstraintRational> same_point_constraint,
    const multibody::MultibodyPlant<double> &plant,
    const systems::Context<double> &context,
    Eigen::Ref<const Eigen::VectorXd>& q_star
//    const Eigen::Ref<const Eigen::VectorXd>& c_cost,
//    const double d_cost,
//    const Eigen::Ref<const Eigen::MatrixXd>& C_constraint,
//    const Eigen::Ref<const Eigen::VectorXd>& d_constraint,
//    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
//    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
//    const solvers::SolverInterface& solver,
//    const Eigen::Ref<const Eigen::VectorXd>& t_guess
    )
    {
      // Check the inputs.
      plant.ValidateContext(context);
      const int nt = plant.num_positions();
      const multibody::RationalForwardKinematics rational_forward_kinematics(plant);
      DRAKE_DEMAND(q_star.rows() == nt);

      // Make all of the convex sets and supporting quantities.
  // TODO(amice): should we provide a way that we don't have to run this for every t_sample point?
  auto query_object =
      plant.get_geometry_query_input_port().Eval<QueryObject<double>>(context);
  const geometry::SceneGraphInspector<double>& inspector = query_object.inspector();
  geometry::optimization::IrisConvexSetMaker maker(query_object, inspector.world_frame_id());
  std::unordered_map<geometry::GeometryId, copyable_unique_ptr<ConvexSet>> sets{};
  std::unordered_map<GeometryId, const multibody::Frame<double>*> frames{};
  const std::unordered_set<GeometryId> geom_ids = inspector.GetGeometryIds(
      GeometrySet(inspector.GetAllGeometryIds()), Role::kProximity);
  copyable_unique_ptr<ConvexSet> temp_set;
  for (GeometryId geom_id : geom_ids) {
    // Make all sets in the local geometry frame.
    FrameId frame_id = inspector.GetFrameId(geom_id);
    maker.set_reference_frame(frame_id);
    maker.set_geometry_id(geom_id);
    inspector.GetShape(geom_id).Reify(&maker, &temp_set);
    sets.emplace(geom_id, std::move(temp_set));
    frames.emplace(geom_id, &plant.GetBodyFromFrameId(frame_id)->body_frame());
  }

  auto pairs = inspector.GetCollisionCandidates();
  const int N = static_cast<int>(pairs.size());

  auto same_point_constraint =
      std::make_shared<SamePointConstraintRational>(&rational_forward_kinematics, q_star, context);


    }

}  // namespace multibody
}  // namespace drake