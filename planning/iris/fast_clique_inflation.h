#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "drake/common/parallelism.h"
#include "drake/geometry/meshcat.h"
// #include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/affine_ball.h"
#include "drake/geometry/optimization/affine_subspace.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/hyperellipsoid.h"
#include "drake/planning/collision_checker.h"

namespace drake {
namespace planning {

using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;

struct FastCliqueInflationOptions {
  /** Passes this object to an Archive.
  Refer to @ref yaml_serialization "YAML Serialization" for background.
  Note: This only serializes options that are YAML built-in types. */
  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(num_particles));
    a->Visit(DRAKE_NVP(tau));
    a->Visit(DRAKE_NVP(delta));
    a->Visit(DRAKE_NVP(admissible_proportion_in_collision));
    a->Visit(DRAKE_NVP(max_iterations_separating_planes));
    a->Visit(DRAKE_NVP(max_separating_planes_per_iteration));
    a->Visit(DRAKE_NVP(bisection_steps));
    a->Visit(DRAKE_NVP(parallelize));
    a->Visit(DRAKE_NVP(verbose));
    a->Visit(DRAKE_NVP(configuration_space_margin));
    a->Visit(DRAKE_NVP(random_seed));
  }

  FastCliqueInflationOptions() = default;

  /** Number of particles used to estimate the closest collision*/
  int num_particles = 4e2;

  /** Descision threshold for unadaptive test.*/
  double tau = 0.5;

  /** Upper bound on the error probability that the admissible_proportion in
   * collision is not met.*/
  double delta = 5e-2;

  /** Admissible fraction of the region volume allowed to be in collision.*/
  double admissible_proportion_in_collision = 1e-2;

  /** Number of resampling steps for the gradient updates*/
  // int num_resampling_steps = 1;

  /** Number Iris Iterations*/
  int max_iterations{2};

  /** Maximum number of rounds of adding faces to the polytope*/
  int max_iterations_separating_planes{100};

  /** Maximum number of faces to add per round of samples, -1 is unlimited*/
  int max_separating_planes_per_iteration{10};

  /** Maximum number of bisection steps per gradient step*/
  int bisection_steps{9};

  /** Parallelize the updates of the particles*/
  bool parallelize{true};

  /* Enables print statements indicating the progress of fast iris**/
  bool verbose{false};

  /** For IRIS in configuration space, we retreat by this margin from each
  C-space obstacle in order to avoid the possibility of requiring an infinite
  number of faces to approximate a curved boundary.
  */
  double configuration_space_margin{1e-2};

  /** The only randomization in IRIS is the random sampling done to find
  counter-examples for the additional constraints using in
  IrisInConfigurationSpace. Use this option to set the initial seed. */
  int random_seed{1234};

  /** Passing a meshcat instance may enable debugging visualizations; this
  currently and when the
  configuration space is <= 3 dimensional.*/
  std::shared_ptr<geometry::Meshcat> meshcat{};
};

/** Given a set of points that form a clique on a visibility graph, use sampling
based optimization to find a collision free polytope in cspace.*/

HPolyhedron FastCliqueInflation(
    const CollisionChecker& checker, const Eigen::MatrixXd& clique,
    const HPolyhedron& domain,
    const FastCliqueInflationOptions& options = FastCliqueInflationOptions());

}  // namespace planning
}  // namespace drake