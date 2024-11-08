#include "drake/planning/iris/iris_from_clique_cover_template.h"

#include <algorithm>
#include <set>

#include "drake/common/drake_throw.h"
#include "drake/geometry/optimization/hyperellipsoid.h"
#include "drake/perception/point_cloud.h"
#include "drake/planning/graph_algorithms/max_clique_solver_via_greedy.h"
#include "drake/planning/graph_algorithms/min_clique_cover_solver_via_greedy.h"
#include "drake/planning/iris/iris_from_clique_cover_template_internal.h"
#include "drake/planning/visibility_graph.h"

namespace drake {
namespace planning {
namespace iris {
using geometry::optimization::HPolyhedron;

std::vector<geometry::optimization::HPolyhedron> PointsToCliqueCoverSets(
    const Eigen::Ref<const Eigen::MatrixXd>& points,
    const CollisionChecker& checker,
    graph_algorithms::MinCliqueCoverSolverBase* min_clique_cover_solver,
    RegionFromCliqueBase* set_builder, Parallelism parallelism, bool partition,
    std::shared_ptr<geometry::Meshcat> meshcat) {
  DRAKE_THROW_UNLESS(min_clique_cover_solver != nullptr);
  DRAKE_THROW_UNLESS(set_builder != nullptr);
  drake::log()->debug("Building visibility graph for {} points", points.cols());
  Eigen::SparseMatrix<bool> visibility_graph =
      VisibilityGraph(checker, points, parallelism);
  drake::log()->debug("Building visibility graph completed. Have {} edges",
                      points.nonZeros() / 2);
  // Compute the clique cover
  drake::log()->debug("Launching Clique Cover Solver");
  std::vector<std::set<int>> clique_cover =
      min_clique_cover_solver->SolveMinCliqueCover(visibility_graph, partition);
  drake::log()->debug("Clique Cover Solver Complete");
  bool do_debugging_visualization =
      meshcat != nullptr && (checker.plant().num_positions() == 3 ||
                             checker.plant().num_positions() == 2);
  if (do_debugging_visualization) {
    Eigen::Matrix3Xd start = Eigen::Matrix3Xd::Zero(
        3, static_cast<int>(visibility_graph.nonZeros()));
    Eigen::Matrix3Xd end = Eigen::Matrix3Xd::Zero(
        3, static_cast<int>(visibility_graph.nonZeros()));
    int this_ctr = 0;
    for (int i = 0; i < visibility_graph.outerSize(); ++i) {
      for (Eigen::SparseMatrix<bool>::InnerIterator it(visibility_graph, i); it;
           ++it) {
        start.topRows(points.rows()).col(this_ctr) = points.col(it.row());
        end.topRows(points.rows()).col(this_ctr) = points.col(it.col());
        ++this_ctr;
      }
    }
    meshcat->SetLineSegments("/visibility_graph", start, end, 1,
                             geometry::Rgba(0, 1, 0, 1));
    int clique_ctr = 0;
    for (const auto& clique : clique_cover) {
      Eigen::Matrix3Xf plot_points(3, ssize(clique));
      int ctr = 0;
      for (const auto& ind : clique) {
        if (points.rows() == 3) {
          plot_points.col(ctr++) = points.col(ind).cast<float>();
        } else {
          plot_points.col(ctr).topRows(2) = points.col(ind).cast<float>();
          plot_points.col(ctr++)(2) = 0.0;
        }
      }
      auto randd = []() {
        return static_cast<double>(std::rand()) / (RAND_MAX + 1.0);
      };
      geometry::Rgba color(randd(), randd(), randd(), 1);
      //      geometry::Rgba color(0, 1, 0, 1);
      drake::perception::PointCloud point_cloud(plot_points.cols(),
                                                perception::pc_flags::kXYZs);
      point_cloud.mutable_xyzs() = plot_points;
      meshcat->SetObject(fmt::format("/clique_points/{}", clique_ctr),
                         point_cloud, 0.1, color);
      ++clique_ctr;
    }
  }

  std::vector<HPolyhedron> sets;
  sets.reserve(clique_cover.size());

  // Inflate the regions.
  // Pre-allocate a data-structure to hold the current clique.
  Eigen::MatrixXd clique(points.rows(), points.cols());
  int ctr = 0;
  for (const auto& clique_inds : clique_cover) {
    drake::log()->debug("Building Clique {}/{}", ctr, ssize(clique_cover));
    int i = 0;
    for (const auto& ind : clique_inds) {
      clique.col(i++) = points.col(ind);
    }
    sets.push_back(
        set_builder->BuildRegion(clique.leftCols(clique_inds.size())));
    ctr += 1;
  }
  return sets;
}

std::vector<HPolyhedron> PointsToCliqueCoverSets(
    const Eigen::Ref<const Eigen::MatrixXd>& points,
    AdjacencyMatrixBuilderBase* graph_builder,
    graph_algorithms::MinCliqueCoverSolverBase* min_clique_cover_solver,
    RegionFromCliqueBase* set_builder, bool partition,
    std::shared_ptr<geometry::Meshcat> meshcat) {
  unused(meshcat);
  DRAKE_THROW_UNLESS(graph_builder != nullptr);
  DRAKE_THROW_UNLESS(min_clique_cover_solver != nullptr);
  DRAKE_THROW_UNLESS(set_builder != nullptr);
  drake::log()->debug("Building visibility graph for {} points", points.cols());
  Eigen::SparseMatrix<bool> visibility_graph =
      graph_builder->BuildAdjacencyMatrix(points);
  drake::log()->debug("Building visibility graph completed. Have {} edges",
                      points.nonZeros() / 2);

  // Compute the clique cover
  std::vector<std::set<int>> clique_cover =
      min_clique_cover_solver->SolveMinCliqueCover(visibility_graph, partition);

  std::vector<HPolyhedron> sets;
  sets.reserve(clique_cover.size());

  // Inflate the regions.
  // Pre-allocate a data-structure to hold the current clique.
  Eigen::MatrixXd clique(points.rows(), points.cols());
  for (const auto& clique_inds : clique_cover) {
    int i = 0;
    for (const auto& ind : clique_inds) {
      clique.col(i++) = points.col(ind);
    }
    sets.push_back(
        set_builder->BuildRegion(clique.leftCols(clique_inds.size())));
  }
  return sets;
}

void IrisInConfigurationSpaceFromCliqueCoverTemplate(
    const IrisFromCliqueCoverOptions& options, const CollisionChecker& checker,
    RandomGenerator* generator, PointSamplerBase* point_sampler,
    graph_algorithms::MinCliqueCoverSolverBase* min_clique_cover_solver,
    RegionFromCliqueBase* set_builder,
    std::vector<geometry::optimization::HPolyhedron>* sets,
    AdjacencyMatrixBuilderBase* adjacency_matrix_builder) {
  DRAKE_THROW_UNLESS(generator != nullptr);
  DRAKE_THROW_UNLESS(point_sampler != nullptr);
  DRAKE_THROW_UNLESS(min_clique_cover_solver != nullptr);
  DRAKE_THROW_UNLESS(set_builder != nullptr);
  DRAKE_THROW_UNLESS(sets != nullptr);

  // Override options which are set too aggressively.
  const int minimum_clique_size = std::max(options.minimum_clique_size,
                                           checker.plant().num_positions() + 1);

  int num_points_per_visibility_round = std::max(
      options.num_points_per_visibility_round, 2 * minimum_clique_size);

  Parallelism max_collision_checker_parallelism{std::min(
      options.parallelism.num_threads(), checker.num_allocated_contexts())};

  // These are initialized by the samplers.
  Eigen::MatrixXd uncovered_collision_free_sampled_points;
  Eigen::MatrixXd collision_free_sampled_points;
  int num_uncovered_collision_free_sampled_points{0};
  int num_collision_free_sampled_points{0};

  Eigen::MatrixXd visibility_graph_points(checker.plant().num_positions(),
                                          num_points_per_visibility_round);
  int ctr = 0;
  auto delta_threshold = [&ctr, &options]() {
    return 6 * options.confidence / (M_PI * M_PI * (ctr + 1) * (ctr + 1));
  };
  while (ctr < options.iteration_limit &&
         !internal::IsSufficientlyCovered(
             options.coverage_termination_threshold, delta_threshold(), checker,
             *sets, generator, point_sampler, &collision_free_sampled_points,
             &num_collision_free_sampled_points,
             &uncovered_collision_free_sampled_points,
             &num_uncovered_collision_free_sampled_points,
             options.sampling_batch_size, max_collision_checker_parallelism)) {
    log()->debug("IrisFromCliqueCover Iteration {}/{}", ctr + 1,
                 options.iteration_limit);
    ctr++;
    // Sample new points. Reuse the work of IsSufficientlyCovered, and if
    // there are not enough points then continue sampling until we have
    // enough.
    int new_points_start_index{0};
    int num_new_points{0};
    if (options.sample_outside_of_sets) {
      num_new_points = std::min(num_uncovered_collision_free_sampled_points,
                                num_points_per_visibility_round);
      visibility_graph_points.middleCols(new_points_start_index,
                                         num_new_points) =
          uncovered_collision_free_sampled_points.leftCols(num_new_points);
    } else {
      num_new_points = std::min(num_collision_free_sampled_points,
                                num_points_per_visibility_round);
      visibility_graph_points.middleCols(new_points_start_index,
                                         num_new_points) =
          collision_free_sampled_points.leftCols(num_new_points);
    }
    new_points_start_index += num_new_points;

    while (new_points_start_index < num_points_per_visibility_round) {
      internal::SampleCollisionFreePoints(
          num_points_per_visibility_round, checker, generator, point_sampler,
          &collision_free_sampled_points, nullptr, options.sampling_batch_size,
          max_collision_checker_parallelism);
      if (options.sample_outside_of_sets) {
        internal::ClassifyCollisionFreePoints(
            collision_free_sampled_points, *sets,
            &uncovered_collision_free_sampled_points,
            &num_uncovered_collision_free_sampled_points);
        num_new_points =
            std::min(num_uncovered_collision_free_sampled_points,
                     num_points_per_visibility_round - new_points_start_index);
        visibility_graph_points.middleCols(new_points_start_index,
                                           num_new_points) =
            uncovered_collision_free_sampled_points.leftCols(num_new_points);
      } else {
        num_new_points = std::min(num_collision_free_sampled_points,
                                  num_points_per_visibility_round);
        visibility_graph_points.middleCols(new_points_start_index,
                                           num_new_points) =
            collision_free_sampled_points.leftCols(num_new_points);
      }
      new_points_start_index += num_new_points;
    }
    bool do_debugging_visualization = options.iris_options.meshcat != nullptr &&
                                      (checker.plant().num_positions() == 3 ||
                                       checker.plant().num_positions() == 2);
    if (do_debugging_visualization) {
      Eigen::Matrix3Xf plot_points(3, visibility_graph_points.cols());
      if (visibility_graph_points.rows() == 3) {
        plot_points = visibility_graph_points.cast<float>();
      } else {
        plot_points.topRows(2) = visibility_graph_points.cast<float>();
        plot_points.bottomRows(1) =
            Eigen::MatrixXf::Constant(1, visibility_graph_points.cols(), 0.0);
      }
      drake::perception::PointCloud point_cloud(plot_points.cols(),
                                                perception::pc_flags::kXYZs);
      point_cloud.mutable_xyzs() = plot_points;
      options.iris_options.meshcat->SetObject(
          fmt::format("/Visibility Points round {}", ctr), point_cloud, 0.01,
          geometry::Rgba(0, 0, 1, 1));
    }

    std::vector<HPolyhedron> new_sets;
    if (adjacency_matrix_builder == nullptr) {
      new_sets = PointsToCliqueCoverSets(
          visibility_graph_points, checker, min_clique_cover_solver,
          set_builder, max_collision_checker_parallelism, options.partition,
          options.iris_options.meshcat);
    } else {
      new_sets = PointsToCliqueCoverSets(
          visibility_graph_points, adjacency_matrix_builder,
          min_clique_cover_solver, set_builder, options.partition,
          options.iris_options.meshcat);
    }
    sets->insert(sets->end(), new_sets.begin(), new_sets.end());
    log()->debug(
        "IrisInConfigurationSpaceFromCliqueCoverTemplate has built {} sets.",
        sets->size());
  }
}

}  // namespace iris
}  // namespace planning
}  // namespace drake
