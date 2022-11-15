#include "drake/geometry/optimization/iris.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "drake/common/symbolic/expression.h"
#include "drake/geometry/optimization/cartesian_product.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/minkowski_sum.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/snopt_solver.h"

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
using solvers::Binding;
using solvers::Constraint;
using solvers::MathematicalProgram;
using symbolic::Expression;
using systems::Context;

HPolyhedron Iris(const ConvexSets& obstacles, const Ref<const VectorXd>& sample,
                 const HPolyhedron& domain, const IrisOptions& options) {
  const int dim = sample.size();
  const int N = obstacles.size();
  DRAKE_DEMAND(domain.ambient_dimension() == dim);
  for (int i = 0; i < N; ++i) {
    DRAKE_DEMAND(obstacles[i]->ambient_dimension() == dim);
  }
  DRAKE_DEMAND(domain.IsBounded());
  const double kEpsilonEllipsoid = 1e-2;
  Hyperellipsoid E = Hyperellipsoid::MakeHypersphere(kEpsilonEllipsoid, sample);
  HPolyhedron P = domain;

  // On each iteration, we will build the collision-free polytope represented as
  // {x | A * x <= b}.  Here we pre-allocate matrices of the maximum size.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      domain.A().rows() + N, dim);
  VectorXd b(domain.A().rows() + N);
  A.topRows(domain.A().rows()) = domain.A();
  b.head(domain.A().rows()) = domain.b();
  // Use pairs {scale, index}, so that I can back out the indices after a sort.
  std::vector<std::pair<double, int>> scaling(N);
  MatrixXd closest_points(dim, N);

  double best_volume = E.Volume();
  int iteration = 0;
  MatrixXd tangent_matrix;

  while (true) {
    DRAKE_ASSERT(best_volume > 0);
    // Find separating hyperplanes
    for (int i = 0; i < N; ++i) {
      const auto touch = E.MinimumUniformScalingToTouch(*obstacles[i]);
      scaling[i].first = touch.first;
      scaling[i].second = i;
      closest_points.col(i) = touch.second;
    }
    std::sort(scaling.begin(), scaling.end());

    int num_constraints = domain.A().rows();
    tangent_matrix = 2.0 * E.A().transpose() * E.A();
    for (int i = 0; i < N; ++i) {
      // Only add a constraint if this obstacle still has overlap with the set
      // that has been constructed so far on this iteration.
      if (HPolyhedron(A.topRows(num_constraints), b.head(num_constraints))
              .IntersectsWith(*obstacles[scaling[i].second])) {
        // Add the tangent to the (scaled) ellipsoid at this point as a
        // constraint.
        const VectorXd point = closest_points.col(scaling[i].second);
        A.row(num_constraints) =
            (tangent_matrix * (point - E.center())).normalized();
        b[num_constraints] = A.row(num_constraints) * point;
        num_constraints++;
      }
    }

    if (options.require_sample_point_is_contained &&
        ((A.topRows(num_constraints) * sample).array() >=
         b.head(num_constraints).array())
            .any()) {
      break;
    }
    P = HPolyhedron(A.topRows(num_constraints), b.head(num_constraints));

    iteration++;
    if (iteration >= options.iteration_limit) {
      break;
    }

    E = P.MaximumVolumeInscribedEllipsoid();
    const double volume = E.Volume();
    const double delta_volume = volume - best_volume;
    if (delta_volume <= options.termination_threshold) {
      break;
    }
    if (delta_volume / best_volume <= options.relative_termination_threshold) {
      break;
    }
    best_volume = volume;
  }

  return P;
}

HPolyhedron IrisMultiContainment(const ConvexSets& obstacles,
                                 const Ref<const MatrixXd>& samples,
                                 const Hyperellipsoid& initellipse,
                                 const HPolyhedron& domain,
                                 const IrisOptions& options) {
  const int dim = samples.cols();
  const int N = obstacles.size();

  DRAKE_DEMAND(domain.ambient_dimension() == dim);
  for (int i = 0; i < N; ++i) {
    DRAKE_DEMAND(obstacles[i]->ambient_dimension() == dim);
  }
  DRAKE_DEMAND(domain.IsBounded());
  // const double kEpsilonEllipsoid = 1e-2;
  Hyperellipsoid E = initellipse;
  // Hyperellipsoid::MakeHypersphere(kEpsilonEllipsoid,
  // sample);
  HPolyhedron P = domain;

  // On each iteration, we will build the collision-free polytope represented as
  // {x | A * x <= b}.  Here we pre-allocate matrices of the maximum size.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      domain.A().rows() + N, dim);
  VectorXd b(domain.A().rows() + N);
  A.topRows(domain.A().rows()) = domain.A();
  b.head(domain.A().rows()) = domain.b();
  // Use pairs {scale, index}, so that I can back out the indices after a sort.
  std::vector<std::pair<double, int>> scaling(N);
  MatrixXd closest_points(dim, N);

  double best_volume = E.Volume();
  int iteration = 0;
  MatrixXd tangent_matrix;

  while (true) {
    DRAKE_ASSERT(best_volume > 0);
    // Find separating hyperplanes
    for (int i = 0; i < N; ++i) {
      const auto touch = E.MinimumUniformScalingToTouch(*obstacles[i]);
      scaling[i].first = touch.first;
      scaling[i].second = i;
      closest_points.col(i) = touch.second;
    }
    std::sort(scaling.begin(), scaling.end());

    int num_constraints = domain.A().rows();
    tangent_matrix = 2.0 * E.A().transpose() * E.A();
    for (int i = 0; i < N; ++i) {
      const VectorXd point = closest_points.col(scaling[i].second);
      // Only add a constraint if this point is still within the set that has
      // been constructed so far on this iteration.
      if (((A.topRows(num_constraints) * point).array() <=
           b.head(num_constraints).array())
              .all()) {
        // Add the tangent to the (scaled) ellipsoid at this point as a
        // constraint.
        A.row(num_constraints) =
            (tangent_matrix * (point - E.center())).normalized();
        b[num_constraints] = A.row(num_constraints) * point;
        num_constraints++;
      }
    }

    for (int i = 0; i < samples.rows(); i++) {
      if (options.require_sample_point_is_contained &&
          ((A.topRows(num_constraints) * samples.row(i).transpose()).array() >=
           b.head(num_constraints).array())
              .any()) {
        return P;
      }
    }

    P = HPolyhedron(A.topRows(num_constraints), b.head(num_constraints));

    iteration++;
    if (iteration >= options.iteration_limit) {
      break;
    }

    E = P.MaximumVolumeInscribedEllipsoid();
    const double volume = E.Volume();
    const double delta_volume = volume - best_volume;
    if (delta_volume <= options.termination_threshold) {
      break;
    }
    if (delta_volume / best_volume <= options.relative_termination_threshold) {
      break;
    }
    best_volume = volume;
  }

  return P;
}

namespace {
// Constructs a ConvexSet for each supported Shape and adds it to the set.
class IrisConvexSetMaker final : public ShapeReifier {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(IrisConvexSetMaker)

  IrisConvexSetMaker(const QueryObject<double>& query,
                     std::optional<FrameId> reference_frame)
      : query_{query}, reference_frame_{reference_frame} {};

  void set_reference_frame(const FrameId& reference_frame) {
    DRAKE_DEMAND(reference_frame.is_valid());
    *reference_frame_ = reference_frame;
  }

  void set_geometry_id(const GeometryId& geom_id) { geom_id_ = geom_id; }

  using ShapeReifier::ImplementGeometry;

  void ImplementGeometry(const Sphere&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<Hyperellipsoid>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Cylinder&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set =
        std::make_unique<CartesianProduct>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const HalfSpace&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<HPolyhedron>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Box&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    // Note: We choose HPolyhedron over VPolytope here, but the IRIS paper
    // discusses a significant performance improvement using a "least-distance
    // programming" instance from CVXGEN that exploited the VPolytope
    // representation.  So we may wish to revisit this.
    set = std::make_unique<HPolyhedron>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Capsule&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<MinkowskiSum>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Ellipsoid&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<Hyperellipsoid>(query_, geom_id_, reference_frame_);
  }

 private:
  const QueryObject<double>& query_{};
  std::optional<FrameId> reference_frame_{};
  GeometryId geom_id_{};
};

}  // namespace

ConvexSets MakeIrisObstacles(const QueryObject<double>& query_object,
                             std::optional<FrameId> reference_frame) {
  const SceneGraphInspector<double>& inspector = query_object.inspector();
  const GeometrySet all_ids(inspector.GetAllGeometryIds());
  const std::unordered_set<GeometryId> geom_ids =
      inspector.GetGeometryIds(all_ids, Role::kProximity);
  ConvexSets sets(geom_ids.size());

  IrisConvexSetMaker maker(query_object, reference_frame);
  int count = 0;
  for (GeometryId geom_id : geom_ids) {
    maker.set_geometry_id(geom_id);
    inspector.GetShape(geom_id).Reify(&maker, &sets[count++]);
  }
  return sets;
}

namespace {
// Solves the optimization
// min_q (q-d)*CᵀC(q-d)
// s.t. setA in frameA and setB in frameB are in collision in q.
//      Aq ≤ b.
// where C, d are the matrix and center from the hyperellipsoid E.
// Returns true iff a collision is found.
// Sets `closest` to an optimizing solution q*, if a solution is found.
bool FindClosestCollision(
    std::shared_ptr<SamePointConstraint> same_point_constraint,
    const multibody::Frame<double>& frameA,
    const multibody::Frame<double>& frameB, const ConvexSet& setA,
    const ConvexSet& setB, const Hyperellipsoid& E,
    const Eigen::Ref<const Eigen::MatrixXd>& A,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const solvers::SolverInterface& solver,
    const Eigen::Ref<const Eigen::VectorXd>& q_guess, VectorXd* closest) {
  solvers::MathematicalProgram prog;
  auto q = prog.NewContinuousVariables(A.cols(), "q");

  prog.AddLinearConstraint(
      A, VectorXd::Constant(b.size(), -std::numeric_limits<double>::infinity()),
      b, q);
  // Scale the objective so the eigenvalues are close to 1, using
  // scale*lambda_min = 1/scale*lambda_max.
  const MatrixXd Asq = E.A().transpose() * E.A();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Asq);
  const double scale = 1.0 / std::sqrt(es.eigenvalues().maxCoeff() *
                                       es.eigenvalues().minCoeff());
  prog.AddQuadraticErrorCost(scale * Asq, E.center(), q);

  auto p_AA = prog.NewContinuousVariables<3>("p_AA");
  auto p_BB = prog.NewContinuousVariables<3>("p_BB");
  setA.AddPointInSetConstraints(&prog, p_AA);
  setB.AddPointInSetConstraints(&prog, p_BB);

  same_point_constraint->set_frameA(&frameA);
  same_point_constraint->set_frameB(&frameB);
  prog.AddConstraint(same_point_constraint, {q, p_AA, p_BB});

  // Help nonlinear optimizers (e.g. SNOPT) avoid trivial local minima at the
  // origin.
  prog.SetInitialGuess(q, q_guess);
  prog.SetInitialGuess(p_AA, Vector3d::Constant(.01));
  prog.SetInitialGuess(p_BB, Vector3d::Constant(.01));

  solvers::MathematicalProgramResult result;
  solver.Solve(prog, std::nullopt, std::nullopt, &result);
  if (result.is_success()) {
    *closest = result.GetSolution(q);
    return true;
  }
  return false;
}

// Add the tangent to the (scaled) ellipsoid at @p point as a
// constraint.
void AddTangentToPolytope(
    const Hyperellipsoid& E, const Eigen::Ref<const Eigen::VectorXd>& point,
    double configuration_space_margin,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* A,
    Eigen::VectorXd* b, int* num_constraints) {
  if (*num_constraints >= A->rows()) {
    // Increase pre-allocated polytope size.
    A->conservativeResize(A->rows() * 2, A->cols());
    b->conservativeResize(b->rows() * 2);
  }

  A->row(*num_constraints) =
      (E.A().transpose() * E.A() * (point - E.center())).normalized();
  (*b)[*num_constraints] =
      A->row(*num_constraints) * point - configuration_space_margin;
  if (A->row(*num_constraints) * E.center() > (*b)[*num_constraints]) {
    throw std::logic_error(
        "The current center of the IRIS region is within "
        "options.configuration_space_margin of being infeasible.  Check your "
        "sample point and/or any additional constraints you've passed in via "
        "the options. The configuration space surrounding the sample point "
        "must have an interior.");
  }
  *num_constraints += 1;
}

struct GeometryPairWithDistance {
  GeometryId geomA;
  GeometryId geomB;
  double distance;

  GeometryPairWithDistance(GeometryId gA, GeometryId gB, double dist)
      : geomA(gA), geomB(gB), distance(dist) {}

  bool operator<(const GeometryPairWithDistance& other) const {
    return distance < other.distance;
  }
};

}  // namespace

HPolyhedron IrisInConfigurationSpace(const MultibodyPlant<double>& plant,
                                     const Context<double>& context,
                                     const IrisOptions& options) {
  plant.ValidateContext(context);
  auto same_point_constraint =
      std::make_shared<SamePointConstraint>(&plant, context);
  const Eigen::VectorXd sample = plant.GetPositions(context);

  // Note: We require finite joint limits to define the bounding box for the
  // IRIS algorithm.
  DRAKE_DEMAND(plant.GetPositionLowerLimits().array().isFinite().all());
  DRAKE_DEMAND(plant.GetPositionUpperLimits().array().isFinite().all());

  // Make the polytope and ellipsoid.
  HPolyhedron P = HPolyhedron::MakeBox(plant.GetPositionLowerLimits(),
                                       plant.GetPositionUpperLimits());

  const double kEpsilonEllipsoid = 1e-2;
  Hyperellipsoid E = Hyperellipsoid::MakeHypersphere(kEpsilonEllipsoid, sample);
  _DoIris_(plant, context, options, sample, same_point_constraint, &P, &E);
  return P;
}

HPolyhedron IrisInRationalConfigurationSpace(
    const multibody::MultibodyPlant<double>& plant,
    const systems::Context<double>& context,
    const IrisOptionsRationalSpace& options,
    const std::optional<HPolyhedron>& starting_hpolyhedron) {
  // Check the inputs.
  plant.ValidateContext(context);
  const int nt = plant.num_positions();
  const multibody::RationalForwardKinematics rational_forward_kinematics(plant);
  Eigen::VectorXd q_star(nt);
  if (!options.q_star.has_value()) {
    q_star.setZero(nt);
  } else {
    q_star = options.q_star.value();
  }
  DRAKE_DEMAND(q_star.rows() == nt);
  DRAKE_DEMAND(plant.GetPositionLowerLimits().array().isFinite().all());
  DRAKE_DEMAND(plant.GetPositionUpperLimits().array().isFinite().all());

  const Eigen::VectorXd sample = plant.GetPositions(context);
  const Eigen::VectorXd t_sample = rational_forward_kinematics.ComputeTValue(
      plant.GetPositions(context), q_star);
  const Eigen::VectorXd t_lower_limits =
      rational_forward_kinematics.ComputeTValue(plant.GetPositionLowerLimits(),
                                                q_star);
  const Eigen::VectorXd t_upper_limits =
      rational_forward_kinematics.ComputeTValue(plant.GetPositionUpperLimits(),
                                                q_star);

  HPolyhedron P = HPolyhedron::MakeBox(t_lower_limits, t_upper_limits);
  if (starting_hpolyhedron.has_value()) {
    P = starting_hpolyhedron.value().Intersection(P);
  }

  const double kEpsilonEllipsoid = 1e-5;
  Hyperellipsoid E =
      Hyperellipsoid::MakeHypersphere(kEpsilonEllipsoid, t_sample);

  auto same_point_constraint = std::make_shared<SamePointConstraintRational>(
      &rational_forward_kinematics, q_star, context);
  _DoIris_(plant, context, options, t_sample, same_point_constraint, &P, &E);
  return P;
}

void _DoIris_(const multibody::MultibodyPlant<double>& plant,
              const systems::Context<double>& context,
              const IrisOptions& options,
              const Eigen::Ref<const Eigen::VectorXd>& sample,
              const std::shared_ptr<SamePointConstraint>& same_point_constraint,
              HPolyhedron* P_ptr, Hyperellipsoid* E_ptr) {
  // Check the inputs.
  plant.ValidateContext(context);
  const int num_joints = plant.num_positions();
  const int nc = static_cast<int>(options.configuration_obstacles.size());

  DRAKE_DEMAND(num_joints == P_ptr->ambient_dimension());
  DRAKE_DEMAND(P_ptr->ambient_dimension() == E_ptr->ambient_dimension());

  // Make all of the convex sets and supporting quantities.
  auto query_object =
      plant.get_geometry_query_input_port().Eval<QueryObject<double>>(context);
  const SceneGraphInspector<double>& inspector = query_object.inspector();
  IrisConvexSetMaker maker(query_object, inspector.world_frame_id());
  std::unordered_map<GeometryId, copyable_unique_ptr<ConvexSet>> sets{};
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

  for (int i = 0; i < nc; ++i) {
    DRAKE_DEMAND(options.configuration_obstacles[i]->ambient_dimension() ==
                 num_joints);
    if (options.configuration_obstacles[i]->PointInSet(sample)) {
      throw std::runtime_error(
          fmt::format("The seed point is in configuration obstacle {}", i));
    }
  }

  auto pairs = inspector.GetCollisionCandidates();
  const int N = static_cast<int>(pairs.size());

  // As a surrogate for the true objective, the pairs are sorted by the distance
  // between each collision pair from the sample point configuration. This could
  // improve computation times in Ibex here and produce regions with fewer
  // faces.
  std::vector<GeometryPairWithDistance> sorted_pairs;
  for (const auto& [geomA, geomB] : pairs) {
    sorted_pairs.emplace_back(
        geomA, geomB,
        query_object.ComputeSignedDistancePairClosestPoints(geomA, geomB)
            .distance);
  }
  std::sort(sorted_pairs.begin(), sorted_pairs.end());

  // On each iteration, we will build the collision-free polytope represented as
  // {x | A * x <= b}.  Here we pre-allocate matrices with a generous maximum
  // size.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      P_ptr->A().rows() + 2 * N, num_joints);
  VectorXd b(P_ptr->A().rows() + 2 * N);
  A.topRows(P_ptr->A().rows()) = P_ptr->A();
  b.head(P_ptr->A().rows()) = P_ptr->b();

  DRAKE_THROW_UNLESS(P_ptr->PointInSet(sample, 1e-12));

  double best_volume = E_ptr->Volume();
  int iteration = 0;
  VectorXd closest(num_joints);

  RandomGenerator generator(options.random_seed);
  std::vector<std::pair<double, int>> scaling(nc);
  MatrixXd closest_points(num_joints, nc);

  auto solver = solvers::MakeFirstAvailableSolver(
      {solvers::SnoptSolver::id(), solvers::IpoptSolver::id()});

  while (true) {
    int num_constraints = 2 * num_joints;  // Start with just the joint limits.
    bool sample_point_requirement = true;
    DRAKE_ASSERT(best_volume > 0);
    // Find separating hyperplanes

    // Add constraints from configuration space obstacles to reduce the domain
    // for later optimization.
    if (options.configuration_obstacles.size() > 0) {
      const ConvexSets& obstacles = options.configuration_obstacles;
      for (int i = 0; i < nc; ++i) {
        const auto touch = E_ptr->MinimumUniformScalingToTouch(*obstacles[i]);
        scaling[i].first = touch.first;
        scaling[i].second = i;
        closest_points.col(i) = touch.second;
      }
      std::sort(scaling.begin(), scaling.end());

      for (int i = 0; i < nc; ++i) {
        // Only add a constraint if this obstacle still has overlap with the set
        // that has been constructed so far on this iteration.
        if (HPolyhedron(A.topRows(num_constraints), b.head(num_constraints))
                .IntersectsWith(*obstacles[scaling[i].second])) {
          const VectorXd point = closest_points.col(scaling[i].second);
          AddTangentToPolytope(*E_ptr, point, 0.0, &A, &b, &num_constraints);
        }
      }

      if (options.require_sample_point_is_contained) {
        sample_point_requirement =
            ((A.topRows(num_constraints) * sample).array() <=
             b.head(num_constraints).array())
                .any();
      }
    }

    // First use a fast nonlinear optimizer to add as many constraint as it
    // can find.
    for (const auto& pair : sorted_pairs) {
      int num_faces = 0;
      while (sample_point_requirement &&
             FindClosestCollision(
                 same_point_constraint, *frames.at(pair.geomA),
                 *frames.at(pair.geomB), *sets.at(pair.geomA),
                 *sets.at(pair.geomB), *E_ptr, A.topRows(num_constraints),
                 b.head(num_constraints), *solver, sample, &closest) &&
             ((options.max_faces_per_collision_pair < 0) ||
              num_faces <= options.max_faces_per_collision_pair)) {
        AddTangentToPolytope(*E_ptr, closest,
                             options.configuration_space_margin, &A, &b,
                             &num_constraints);
        if (options.require_sample_point_is_contained) {
          sample_point_requirement =
              A.row(num_constraints - 1) * sample <= b(num_constraints - 1);
        }
        num_faces++;
      }
    }

    if (!sample_point_requirement) {
      break;
    }
    *P_ptr = HPolyhedron(A.topRows(num_constraints), b.head(num_constraints));

    iteration++;
    if (iteration >= options.iteration_limit) {
      break;
    }

    *E_ptr = P_ptr->MaximumVolumeInscribedEllipsoid();
    const double volume = E_ptr->Volume();
    const double delta_volume = volume - best_volume;
    if (delta_volume <= options.termination_threshold) {
      break;
    }
    if (delta_volume / best_volume <= options.relative_termination_threshold) {
      break;
    }
    best_volume = volume;
  }
}

void SamePointConstraint::DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
                                 Eigen::VectorXd* y) const {
  DRAKE_DEMAND(frameA_ != nullptr);
  DRAKE_DEMAND(frameB_ != nullptr);
  VectorXd q = x.head(plant_->num_positions());
  Vector3d p_AA = x.template segment<3>(plant_->num_positions()),
           p_BB = x.template tail<3>();
  Vector3d p_WA, p_WB;
  plant_->SetPositions(context_.get(), q);
  plant_->CalcPointsPositions(*context_, *frameA_, p_AA, plant_->world_frame(),
                              &p_WA);
  plant_->CalcPointsPositions(*context_, *frameB_, p_BB, plant_->world_frame(),
                              &p_WB);
  *y = p_WA - p_WB;
}

void SamePointConstraint::DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
                                 AutoDiffVecXd* y) const {
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
      ExtractDoubleOrThrow(p_AA), plant_->world_frame(), plant_->world_frame(),
      &Jq_v_WA);
  plant_->CalcJacobianTranslationalVelocity(
      *context_, JacobianWrtVariable::kQDot, *frameB_,
      ExtractDoubleOrThrow(p_BB), plant_->world_frame(), plant_->world_frame(),
      &Jq_v_WB);
  const Eigen::Vector3d y_val =
      X_WA * math::ExtractValue(p_AA) - X_WB * math::ExtractValue(p_BB);
  Eigen::Matrix3Xd dy(3, plant_->num_positions() + 6);
  dy << Jq_v_WA - Jq_v_WB, X_WA.rotation().matrix(), -X_WB.rotation().matrix();
  *y = math::InitializeAutoDiff(y_val, dy * math::ExtractGradient(x));
}

void SamePointConstraint::DoEval(
    const Ref<const VectorX<symbolic::Variable>>& x,
    VectorX<symbolic::Expression>* y) const {
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

void SamePointConstraintRational::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* y) const {
  DRAKE_DEMAND(frameA_ != nullptr);
  DRAKE_DEMAND(frameB_ != nullptr);
  VectorXd t = x.head(plant_->num_positions());
  VectorXd q = rational_forward_kinematics_ptr_->ComputeQValue(t, q_star_);
  Vector3d p_AA = x.template segment<3>(plant_->num_positions()),
           p_BB = x.template tail<3>();
  Vector3d p_WA, p_WB;
  plant_->SetPositions(context_.get(), q);
  plant_->CalcPointsPositions(*context_, *frameA_, p_AA, plant_->world_frame(),
                              &p_WA);
  plant_->CalcPointsPositions(*context_, *frameB_, p_BB, plant_->world_frame(),
                              &p_WB);
  *y = p_WA - p_WB;
}

void SamePointConstraintRational::DoEval(
    const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const {
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
      ExtractDoubleOrThrow(p_AA), plant_->world_frame(), plant_->world_frame(),
      &Jq_v_WA);
  plant_->CalcJacobianTranslationalVelocity(
      *context_, JacobianWrtVariable::kQDot, *frameB_,
      ExtractDoubleOrThrow(p_BB), plant_->world_frame(), plant_->world_frame(),
      &Jq_v_WB);
  Eigen::Matrix3Xd Jt_v_WA(3, plant_->num_positions()),
      Jt_v_WB(3, plant_->num_positions());
  for (int i = 0; i < plant_->num_positions(); i++) {
    // dX_t_wa = J_q_WA * dq_dt
    Jt_v_WA.col(i) = Jq_v_WA.col(i) * q(i).derivatives()(i);
    Jt_v_WB.col(i) = Jq_v_WB.col(i) * q(i).derivatives()(i);
  }

  const Eigen::Vector3d y_val =
      X_WA * math::ExtractValue(p_AA) - X_WB * math::ExtractValue(p_BB);
  Eigen::Matrix3Xd dy(3, plant_->num_positions() + 6);
  dy << Jt_v_WA - Jt_v_WB, X_WA.rotation().matrix(), -X_WB.rotation().matrix();
  *y = math::InitializeAutoDiff(y_val, dy * math::ExtractGradient(x));
}

void SamePointConstraintRational::DoEval(
    const Ref<const VectorX<symbolic::Variable>>& x,
    VectorX<symbolic::Expression>* y) const {
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

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
