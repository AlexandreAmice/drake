//
// Created by amice on 2/4/22.
//

#include "drake/multibody/rational_forward_kinematics/t_in_collision_contraint.h"

#include "drake/solvers/choose_best_solver.h"

namespace drake {
namespace multibody {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using geometry::optimization::ConvexSet;
using geometry::optimization::Hyperellipsoid;
using geometry::optimization::SamePointConstraintRational;

/**
 * Find the maximum amount we can move the inequality c_cost^T t <= d_cost until
 * we either collide between two collision pairs or the inequality becomes
 * redundant.
 * @param same_point_constraint: a same point constraint object
 * @param frameA:
 * @param frameB
 * @param setA
 * @param setB
 * @param c_cost
 * @param d_cost
 * @param C_constraint: C_constraint t <= d_constraint polytope we wish to stay
 * within.
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
    const ConvexSet& setB, const Eigen::Ref<const Eigen::VectorXd>& c_cost,
    const double d_cost, const Eigen::Ref<const Eigen::MatrixXd>& C_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& d_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
    const solvers::SolverInterface& solver,
    const Eigen::Ref<const Eigen::VectorXd>& t_guess) {
  solvers::MathematicalProgram prog;
  solvers::MathematicalProgramResult result;
  double max_eps{0};
  auto t = prog.NewContinuousVariables(C_constraint.cols(), "t");

  prog.AddLinearConstraint(
      C_constraint,
      VectorXd::Constant(d_constraint.size(),
                         -std::numeric_limits<double>::infinity()),
      d_constraint, t);
  prog.AddBoundingBoxConstraint(t_lower_limits, t_upper_limits, t);
  prog.AddLinearCost(c_cost, d_cost, t);
  // solve just the LP to decide when the inequality c_cost^T t <= d_cost becomes redundant
  solver.Solve(prog, {}, {}, &result);
  if (result.is_success()) {
    max_eps = result.get_optimal_cost();
  }
  else {
    // this should only happen if C_constraint t <= d_constraint is empty so maybe it doesn't make sense to
    // return nullopt
    return {};
  }

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



  solver.Solve(prog, std::nullopt, std::nullopt, &result);
  if (result.is_success()) {
    // return most conservative of the max_eps
    return std::min(result.get_optimal_cost(), max_eps);
  }
  return max_eps;
}

/**
 * given a plant find the maximum epsilon such that c_cost^T t <= d_cost + eps is either in collision or redundant
 * with respect to C_constraint t <= d_constrain
 * @param plant
 * @param context
 * @param q_star
 * @param c_cost
 * @param d_cost
 * @param C_constraint
 * @param d_constraint
 * @param t_lower_limits
 * @param t_upper_limits
 * @param solver
 * @param t_guess
 * @return
 */
//std::optional<double> FindMaxEpsTilCollisionForIneq(
//    const multibody::MultibodyPlant<double> &plant,
//    const systems::Context<double> &context,
//    Eigen::Ref<const Eigen::VectorXd>& q_star,
//    const Eigen::Ref<const Eigen::VectorXd>& c_cost,
//    const double d_cost,
//    const Eigen::Ref<const Eigen::MatrixXd>& C_constraint,
//    const Eigen::Ref<const Eigen::VectorXd>& d_constraint,
//    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
//    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
//    const solvers::SolverInterface& solver,
//    const Eigen::Ref<const Eigen::VectorXd>& t_guess
//    ) {
//  // Check the inputs.
//  plant.ValidateContext(context);
//  const int nt = plant.num_positions();
//  const multibody::RationalForwardKinematics rational_forward_kinematics(plant);
//  DRAKE_DEMAND(q_star.rows() == nt);
//
//  // Make all of the convex sets and supporting quantities.
//  auto query_object =
//      plant.get_geometry_query_input_port()
//          .Eval < geometry::QueryObject < double >> (context);
//  const geometry::SceneGraphInspector<double> &inspector =
//      query_object.inspector();
//  geometry::optimization::IrisConvexSetMaker maker(query_object,
//                                                   inspector.world_frame_id());
//  std::unordered_map<geometry::GeometryId, copyable_unique_ptr<ConvexSet>>
//      sets{};
//  std::unordered_map<geometry::GeometryId, const multibody::Frame<double> *> frames{};
//  const std::unordered_set<geometry::GeometryId> geom_ids = inspector.GetGeometryIds(
//      geometry::GeometrySet(inspector.GetAllGeometryIds()), geometry::Role::kProximity);
//  copyable_unique_ptr<ConvexSet> temp_set;
//  for (geometry::GeometryId geom_id: geom_ids) {
//    // Make all sets in the local geometry frame.
//    geometry::FrameId frame_id = inspector.GetFrameId(geom_id);
//    maker.set_reference_frame(frame_id);
//    maker.set_geometry_id(geom_id);
//    inspector.GetShape(geom_id).Reify(&maker, &temp_set);
//    sets.emplace(geom_id, std::move(temp_set));
//    frames.emplace(geom_id, &plant.GetBodyFromFrameId(frame_id)->body_frame());
//  }
//
//  auto pairs = inspector.GetCollisionCandidates();
//
//  auto same_point_constraint = std::make_shared<SamePointConstraintRational>(
//      &rational_forward_kinematics, q_star, context);
//
//  std::optional<double> ret_val{};
//  std::optional<double> cur_val{};
//  for (const auto &pair: pairs) {
//    cur_val = FindMaxEpsTilCollisionForIneqForCollisionPair(
//        same_point_constraint, *frames.at(pair.first),
//                     *frames.at(pair.second), *sets.at(pair.first),
//                     *sets.at(pair.second), c_cost, d_cost, C_constraint, d_constraint,
//                     t_lower_limits, t_upper_limits, solver, t_guess
//        );
//
//    // keep the minimum of the max epsilons as this is the tightest condition
//    if (ret_val.has_value() and cur_val.has_value()){
//      ret_val.value() = std::min(ret_val.value(), cur_val.value());
//    }
//    else if(cur_val.has_value() and not ret_val.has_value()){
//      ret_val.value() = cur_val.value();
//    }
//  }
//  return ret_val;
//}

// same overloaded method except does not need to reconstruct the collision pairs
std::optional<double> FindMaxEpsTilCollisionForIneq(
    const std::set<std::pair<geometry::GeometryId, geometry::GeometryId>> &pairs,
    const std::unordered_map<geometry::GeometryId, const multibody::Frame<double> *> &frames,
    const std::unordered_map<geometry::GeometryId, copyable_unique_ptr<ConvexSet>>&
      sets,
    const multibody::RationalForwardKinematics& rational_forward_kinematics,
    const systems::Context<double> &context,
    Eigen::Ref<const Eigen::VectorXd>& q_star,
    const Eigen::Ref<const Eigen::VectorXd>& c_cost,
    const double d_cost,
    const Eigen::Ref<const Eigen::MatrixXd>& C_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& d_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
    const solvers::SolverInterface& solver,
    const Eigen::Ref<const Eigen::VectorXd>& t_guess
    ) {
  auto same_point_constraint = std::make_shared<SamePointConstraintRational>(
      &rational_forward_kinematics, q_star, context);

  std::optional<double> ret_val{};
  std::optional<double> cur_val{};
  for (const auto &pair: pairs) {
    cur_val = FindMaxEpsTilCollisionForIneqForCollisionPair(
        same_point_constraint, *frames.at(pair.first),
                     *frames.at(pair.second), *sets.at(pair.first),
                     *sets.at(pair.second), c_cost, d_cost, C_constraint, d_constraint,
                     t_lower_limits, t_upper_limits, solver, t_guess
        );

    // keep the minimum of the max epsilons as this is the tightest condition
    if (ret_val.has_value() and cur_val.has_value()){
      ret_val.value() = std::min(ret_val.value(), cur_val.value());
    }
    else if(cur_val.has_value() and not ret_val.has_value()){
      ret_val.value() = cur_val.value();
    }
  }
  return ret_val;
}

Eigen::VectorXd FindMaxEpsForAllIneqs(
    const multibody::MultibodyPlant<double> &plant,
    const systems::Context<double> &context,
    Eigen::Ref<const Eigen::VectorXd>& q_star,
    const Eigen::Ref<const Eigen::VectorXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
    const solvers::SolverInterface& solver,
    const Eigen::Ref<const Eigen::VectorXd>& t_guess){
  // Check the inputs.
  plant.ValidateContext(context);
  const int nt = plant.num_positions();
  const multibody::RationalForwardKinematics rational_forward_kinematics(plant);
  DRAKE_DEMAND(q_star.rows() == nt);

  // Make all of the convex sets and supporting quantities.
  auto query_object =
      plant.get_geometry_query_input_port()
          .Eval < geometry::QueryObject < double >> (context);
  const geometry::SceneGraphInspector<double> &inspector =
      query_object.inspector();
  geometry::optimization::IrisConvexSetMaker maker(query_object,
                                                   inspector.world_frame_id());
  std::unordered_map<geometry::GeometryId, copyable_unique_ptr<ConvexSet>>
      sets{};
  std::unordered_map<geometry::GeometryId, const multibody::Frame<double> *> frames{};
  const std::unordered_set<geometry::GeometryId> geom_ids = inspector.GetGeometryIds(
      geometry::GeometrySet(inspector.GetAllGeometryIds()), geometry::Role::kProximity);
  copyable_unique_ptr<ConvexSet> temp_set;
  for (geometry::GeometryId geom_id: geom_ids) {
    // Make all sets in the local geometry frame.
    geometry::FrameId frame_id = inspector.GetFrameId(geom_id);
    maker.set_reference_frame(frame_id);
    maker.set_geometry_id(geom_id);
    inspector.GetShape(geom_id).Reify(&maker, &temp_set);
    sets.emplace(geom_id, std::move(temp_set));
    frames.emplace(geom_id, &plant.GetBodyFromFrameId(frame_id)->body_frame());
  }

  auto pairs = inspector.GetCollisionCandidates();

  // TODO (Alex.Amice) parallelize
  Eigen::VectorXd eps_max = Eigen::VectorXd::Zero(d.rows());
  Eigen::MatrixXd C_constraint(C.rows()-1, C.cols());
  Eigen::VectorXd d_constraint(d.rows()-1);
  Eigen::VectorXd c_cost(C.cols());
  double d_cost{0};
  for (int i = 0; i < C.rows(); i++){
    if(i > 0){
          C_constraint.topRows(i) = C.topRows(i);
          d_constraint.topRows(i) = d.topRows(i);
    }
    c_cost = C.row(i);
    d_cost = d(i);

    if(i < C.rows()-1){
      C_constraint.bottomRows(i) = C.bottomRows(i+1);
      d_constraint.bottomRows(i) = d.bottomRows(i+1);
    }

    eps_max.row(i) = FindMaxEpsTilCollisionForIneq(pairs,frames,
      sets, rational_forward_kinematics, context, q_star,c_cost,d_cost,C_constraint,d_constraint,t_lower_limits,
    t_upper_limits,solver,t_guess
    );

  }


}


}  // namespace multibody
}  // namespace drake