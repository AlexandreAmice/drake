#include "drake/multibody/rational_forward_kinematics/maximal_epsilon_expansion.h"

#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"


namespace drake {
namespace multibody {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using geometry::optimization::ConvexSet;
using geometry::GeometryId;
using geometry::ShapeReifier;
using Eigen::Ref;
using math::RigidTransform;
using multibody::Body;
using multibody::Frame;
using multibody::JacobianWrtVariable;
using multibody::MultibodyPlant;
using symbolic::Expression;
using systems::Context;
const double kInf = std::numeric_limits<double>::infinity();


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
 * @param non_linear_solver
 * @param t_guess: initial guess
 * @return
 */
std::optional<double> FindMaxEpsTilCollisionForIneqForCollisionPair(
    std::shared_ptr<SamePointConstraintRational> same_point_constraint,
    const multibody::Frame<double>& frameA,
    const multibody::Frame<double>& frameB, const ConvexSet& setA,
    const ConvexSet& setB, const Eigen::Ref<const Eigen::VectorXd>& c_cost,
    const double d_cost, double const eps_min,
    const Eigen::Ref<const Eigen::MatrixXd>& C_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& d_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
    const solvers::SolverInterface& non_linear_solver,
    const Eigen::Ref<const Eigen::VectorXd>& t_sample) {
  solvers::MathematicalProgram lin_prog;
  solvers::MathematicalProgram non_lin_prog;
  solvers::MathematicalProgramResult result;
//  double max_eps{-52};
  auto t_non_lin = non_lin_prog.NewContinuousVariables(C_constraint.cols(), "t");
  auto t_lin = lin_prog.NewContinuousVariables(C_constraint.cols(), "t");

  // hack so itll still compile
  lin_prog.AddLinearConstraint(
      C_constraint,
      VectorXd::Constant(d_constraint.size(),
                         -std::numeric_limits<double>::infinity()),
      d_constraint, t_lin);

  lin_prog.AddBoundingBoxConstraint(t_lower_limits, t_upper_limits, t_lin);
  // max until become redundant
  lin_prog.AddLinearCost(-c_cost, d_cost, t_lin);
  // solve just the LP to decide when the inequality c_cost^T t_non_lin <= d_cost
  // becomes redundant. Do this as a convex program.
  result = solvers::Solve(lin_prog);

  non_lin_prog.AddBoundingBoxConstraint(t_lower_limits, t_upper_limits, t_non_lin);

  //ensure eps_max >= eps_min
  non_lin_prog.AddLinearConstraint(c_cost.transpose(), eps_min + d_cost, kInf, t_non_lin);
  //ensure that we don't pass the seed point
  non_lin_prog.AddLinearConstraint(c_cost.transpose(), c_cost.transpose()*t_sample, kInf, t_non_lin);

  // min until collision
  non_lin_prog.AddLinearCost(c_cost, -d_cost, t_non_lin);



  auto p_AA = non_lin_prog.NewContinuousVariables<3>("p_AA");
  auto p_BB = non_lin_prog.NewContinuousVariables<3>("p_BB");
  setA.AddPointInSetConstraints(&non_lin_prog, p_AA);
  setB.AddPointInSetConstraints(&non_lin_prog, p_BB);

  same_point_constraint->set_frameA(&frameA);
  same_point_constraint->set_frameB(&frameB);
  non_lin_prog.AddConstraint(same_point_constraint, {t_non_lin, p_AA, p_BB});

  // Help nonlinear optimizers (e.g. SNOPT) avoid trivial local minima at the
  // origin.
  non_lin_prog.SetInitialGuess(t_non_lin, t_sample);
  non_lin_prog.SetInitialGuess(p_AA, Vector3d::Constant(.01));
  non_lin_prog.SetInitialGuess(p_BB, Vector3d::Constant(.01));

  non_linear_solver.Solve(non_lin_prog, std::nullopt, std::nullopt, &result);

  if (result.is_success()) {
    // return most conservative of the max_eps
    return result.get_optimal_cost(); //std::min(result.get_optimal_cost(), max_eps);
  }
  return {};
}

/**
 * given a plant find the maximum epsilon such that c_cost^T t <= d_cost + eps
 * is either in collision or redundant with respect to C_constraint t <=
 * d_constrain
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
 * @param t_sample
 * @return
 */
std::optional<double> FindMaxEpsTilCollisionForIneq(
    const std::set<std::pair<geometry::GeometryId, geometry::GeometryId>>&
        pairs,
    const std::unordered_map<geometry::GeometryId,
                             const multibody::Frame<double>*>& frames,
    const std::unordered_map<geometry::GeometryId,
                             copyable_unique_ptr<ConvexSet>>& sets,
    const RationalForwardKinematics& rational_forward_kinematics,
    const systems::Context<double>& context,
    Eigen::Ref<const Eigen::VectorXd>& q_star,
    const Eigen::Ref<const Eigen::VectorXd>& c_cost, const double d_cost,
    const double eps_min,
    const Eigen::Ref<const Eigen::MatrixXd>& C_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& d_constraint,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
    const solvers::SolverInterface& solver,
    const Eigen::Ref<const Eigen::VectorXd>& t_sample) {
  auto same_point_constraint = std::make_shared<SamePointConstraintRational>(
      &rational_forward_kinematics, q_star, context);

  std::optional<double> ret_val{};
  std::optional<double> cur_val{};
  for (const auto& pair : pairs) {
    cur_val = FindMaxEpsTilCollisionForIneqForCollisionPair(
        same_point_constraint, *frames.at(pair.first), *frames.at(pair.second),
        *sets.at(pair.first), *sets.at(pair.second), c_cost, d_cost, eps_min,
        C_constraint, d_constraint, t_lower_limits, t_upper_limits, solver,
        t_sample);

    // keep the minimum of the max epsilons as this is the tightest condition
    if (ret_val.has_value() and cur_val.has_value()) {
      ret_val = std::min(ret_val.value(), cur_val.value());
    } else if (cur_val.has_value() and not ret_val.has_value()) {
      ret_val = cur_val.value();
    }
  }
  return ret_val;
}

/**
 * Find the maximal expansion of the polytope Ct <=d + eps such that every inequality either touches and obstacle
 * or becomes redundant with respect to the original Ct <= d
 * @param plant
 * @param context
 * @param q_star
 * @param C
 * @param d
 * @param t_lower_limits
 * @param t_upper_limits
 * @param solver
 * @param t_guess
 * @return
 */
Eigen::VectorXd FindMaxEpsForAllIneqs(
    const multibody::MultibodyPlant<double>& plant,
    const systems::Context<double>& context,
    Eigen::Ref<const Eigen::VectorXd>& q_star,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::VectorXd>& eps_min,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits,
    const Eigen::Ref<const Eigen::VectorXd>& t_sample) {
  // Check the inputs.
  plant.ValidateContext(context);
  const int nt = plant.num_positions();
  const RationalForwardKinematics rational_forward_kinematics(plant);
  auto non_linear_solver = solvers::MakeFirstAvailableSolver(
      {solvers::SnoptSolver::id(), solvers::IpoptSolver::id()});
  DRAKE_DEMAND(q_star.rows() == nt);

  // Make all of the convex sets and supporting quantities.
  auto query_object =
      plant.get_geometry_query_input_port().Eval<geometry::QueryObject<double>>(
          context);
  const geometry::SceneGraphInspector<double>& inspector =
      query_object.inspector();
  IrisConvexSetMaker maker(query_object,
                                                   inspector.world_frame_id());
  std::unordered_map<geometry::GeometryId, copyable_unique_ptr<ConvexSet>>
      sets{};
  std::unordered_map<geometry::GeometryId, const multibody::Frame<double>*>
      frames{};
  const std::unordered_set<geometry::GeometryId> geom_ids =
      inspector.GetGeometryIds(
          geometry::GeometrySet(inspector.GetAllGeometryIds()),
          geometry::Role::kProximity);
  copyable_unique_ptr<ConvexSet> temp_set;
  for (geometry::GeometryId geom_id : geom_ids) {
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
  std::optional<double> cur_ret{};
  Eigen::MatrixXd C_constraint(C.rows() - 1, C.cols());
  Eigen::VectorXd d_constraint(d.rows() - 1);
  Eigen::VectorXd c_cost(C.cols());
  Eigen::VectorXd t_guess(t_sample.rows());
  double d_cost{0};
  for (int i = 0; i < C.rows(); i++) {
    for (int j = 0; j < C.rows(); j++) {
      if (j < i) {
        C_constraint.row(j) = C.row(j);
        d_constraint(j) = d(j);
      }
      if (j == i) {
        c_cost = C.row(j);
        d_cost = d(j);
      }
      if (j > i) {
        C_constraint.row(j-1) = C.row(j);
        d_constraint(j-1) = d(j);
      }
    }

    t_guess = c_cost.transpose().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(d + c_cost*t_sample);

    cur_ret = FindMaxEpsTilCollisionForIneq(
        pairs, frames, sets, rational_forward_kinematics, context, q_star,
        c_cost, d_cost, eps_min(i), C_constraint, d_constraint, t_lower_limits,
        t_upper_limits, *non_linear_solver, t_guess);

    if (cur_ret.has_value()) {
      eps_max(i) = cur_ret.value();
    }
    std::cout << fmt::format("completed ineq {}/{}", i+1, C.rows()) << std::endl;
  }
  return eps_max;
}

//Eigen::VectorXd FindMaxEpsScalingForAllIneqsForCollisionPair(
//    std::shared_ptr<SamePointConstraintRational> same_point_constraint,
//    const multibody::Frame<double>& frameA,
//    const multibody::Frame<double>& frameB, const ConvexSet& setA,
//    const ConvexSet& setB,
//    const Eigen::Ref<const Eigen::MatrixXd>& C,
//    const Eigen::Ref<const Eigen::VectorXd>& d,
//    const Eigen::Ref<const Eigen::VectorXd>& t_center,
//    const Eigen::Ref<const Eigen::VectorXd>& t_lower_limits,
//    const Eigen::Ref<const Eigen::VectorXd>& t_upper_limits
//    ) {
//  solvers::MathematicalProgram prog;
//  solvers::MathematicalProgramResult result;
//  auto t = prog.NewContinuousVariables(C.cols(), "t");
//  auto eps = prog.NewContinuousVariables(C.rows(), "eps");
//
//
//
//  prog.AddBoundingBoxConstraint(t_lower_limits, t_upper_limits, t);
//  prog.AddBoundingBoxConstraint(Eigen::VectorXd::Zero(eps.rows()), Eigen::VectorXd::Constant(eps.rows(), 1, kInf), eps);
//  prog.AddLinearConstraint(C*(t-t_center) <= eps.cwiseProduct(d));
//
//  //TODO(Alex.Amice) add constraint that eps doesn't scale beyond bounding box
//  prog.AddLinearCost(Eigen::MatrixXd::One(eps.rows()), 0, eps);
//
//  // same point constraint
//  auto p_AA = non_lin_prog.NewContinuousVariables<3>("p_AA");
//  auto p_BB = non_lin_prog.NewContinuousVariables<3>("p_BB");
//  setA.AddPointInSetConstraints(&prog, p_AA);
//  setB.AddPointInSetConstraints(&prog, p_BB);
//
//  same_point_constraint->set_frameA(&frameA);
//  same_point_constraint->set_frameB(&frameB);
//  non_lin_prog.AddConstraint(same_point_constraint, {t_non_lin, p_AA, p_BB});
//
//  // Help nonlinear optimizers (e.g. SNOPT) avoid trivial local minima at the
//  // origin.
//  non_lin_prog.SetInitialGuess(t_non_lin, t_guess);
//  non_lin_prog.SetInitialGuess(p_AA, Vector3d::Constant(.01));
//  non_lin_prog.SetInitialGuess(p_BB, Vector3d::Constant(.01));
//
//  non_linear_solver.Solve(non_lin_prog, std::nullopt, std::nullopt, &result);
//
//  if (result.is_success()) {
//    // return most conservative of the max_eps
//    return result.get_optimal_cost(); //std::min(result.get_optimal_cost(), max_eps);
//  }
//  return {};
//}


}  // namespace multibody
}  // namespace drake