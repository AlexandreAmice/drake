#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "drake/common/parallelism.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/iris.h"
#include "drake/planning/graph_algorithms/max_clique_solver_base.h"
#include "drake/planning/graph_algorithms/max_clique_solver_via_mip.h"
#include "drake/planning/scene_graph_collision_checker.h"

namespace drake {
namespace planning {
/**
 * The default configurations for running Iris when building a convex set from a
 * clique. Currently, it is recommended to only run Iris for one iteration when
 * building from a clique so as to avoid discarding the information gained from
 * the clique.
 */
struct IrisFromCliqueCoverOptions {
  /**
   * The options used on internal calls to Iris.
   *
   * Note that `IrisOptions` can optionally include a meshcat instance to
   * provide debugging visualization. If this is provided `IrisFromCliqueCover`
   * will provide debug visualization in meshcat showing where in configuration
   * space it is drawing from. However, if the parallelism option is set to
   * allow more than 1 thread, then the debug visualizations of internal Iris
   * calls will be disabled. This is due to a limitation of drawing to meshcat
   * from outside the main thread.
   */
  geometry::optimization::IrisOptions iris_options{.iteration_limit = 1};

  /**
   * The fraction of the domain that must be covered before we terminate the
   * algorithm.
   */
  double coverage_termination_threshold{0.7};

  /**
   * The maximum number of iterations before the algorithm terminates.
   */
  int iteration_limit{100};

  /**
   * The number of points to sample when testing coverage.
   */
  int num_points_per_coverage_check{static_cast<int>(1e3)};

  /**
   * The amount of parallelism to use. This algorithm makes heavy use of
   * parallelism at many points and thus it is highly recommended to set this to
   * the maximum tolerable parallelism.
   */
  Parallelism parallelism{Parallelism::Max()};

  /**
   * The minimum size of the cliques used to construct a region. If this is set
   * lower than the ambient dimension of the space we are trying to cover, then
   * this option will be overridden to be at least 1 + the ambient dimension.
   */
  int minimum_clique_size{3};

  /**
   * Number of points to sample when building visibilty cliques. If this option
   * is less than twice the minimum clique size, it will be overridden to be at
   * least twice the minimum clique size. If the algorithm ever fails to find a
   * single clique in a visibility round, then the number of points in a
   * visibility round will be doubled.
   */
  int num_points_per_visibility_round{200};

  /**
   * The max clique solver used. If parallelism is set to allow more than 1
   * thread, then this class **must** be implemented in C++.
   */
  std::unique_ptr<planning::graph_algorithms::MaxCliqueSolverBase>
      max_clique_solver{
          new planning::graph_algorithms::MaxCliqueSolverViaMip()};

  /**
   * The rank tolerance used for computing the
   * MinimumVolumeCircumscribedEllipsoid of a clique. See
   * @MinimumVolumeCircumscribedEllipsoid.
   */
  double rank_tol_for_lowner_john_ellipse{1e-6};

  /**
   * The tolerance used for checking whether a point is contained inside an
   * HPolyhedron. See @ConvexSet::PointInSet.
   */
  double point_in_set_tol{1e-6};
};

/**
 * Cover the configuration space in Iris regions using the Visibility Clique
 * Cover Algorithm as described in
 *
 * P. Werner, A. Amice, T. Marcucci, D. Rus, R. Tedrake "Approximating Robot
 * Configuration Spaces with few Convex Sets using Clique Covers of Visibility
 * Graphs" In 2024 IEEE Internation Conference on Robotics and Automation.
 * https://arxiv.org/abs/2310.02875
 *
 * @param checker The collision checker containing the plant and it's associated
 * scene_graph.
 * @param There are points in the algorithm requiring randomness. The generator
 * controls this source of randomness.
 * @param sets [in/out] initial sets covering the space (potentially empty).
 * The cover is written into this vector.
 */
void IrisInConfigurationSpaceFromCliqueCover(
    const CollisionChecker& checker, const IrisFromCliqueCoverOptions& options,
    RandomGenerator* generator,
    std::vector<geometry::optimization::HPolyhedron>* sets);

}  // namespace planning
}  // namespace drake
