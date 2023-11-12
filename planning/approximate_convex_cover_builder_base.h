#pragma once

#include <memory>
#include <queue>
#include <vector>

#include "drake/geometry/optimization/convex_set.h"
#include "drake/planning/graph_algorithms/max_clique_solver_base.h"

namespace drake {
namespace planning {

using geometry::optimization::ConvexSet;
using graph_algorithms::MaxCliqueSolverBase;

/**
 * An abstract base class for implementing methods for checking whether a set of
 * convex sets provides sufficient coverage for solving approximate convex
 * cover.
 */
class CoverageCheckerBase {
 public:
  /**
   * Check that the current sets provide sufficient coverage.
   */
  virtual bool CheckCoverage(
      const std::vector<ConvexSet>& current_sets) const = 0;

  /**
   * Check that the current sets provide sufficient coverage.
   */
  virtual bool CheckCoverage(
      const std::queue<ConvexSet>& current_sets) const = 0;

  virtual bool CheckCoverage(
      const std::queue<std::unique_ptr<ConvexSet>>& current_sets) const = 0;

   virtual bool CheckCoverage(
      const std::vector<std::unique_ptr<ConvexSet>>& current_sets) const = 0;

  virtual ~CoverageCheckerBase() {}


 protected:
  // We put the copy/move/assignment constructors as protected to avoid copy
  // slicing. The inherited final subclasses should put them in public
  // functions.
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(CoverageCheckerBase);
};

/**
 * An abstract base class for implementing ways to sample points from the
 * underlying space an approximate convex cover builder wishes to cover.
 */
class PointSamplerBase {
 public:
  /**
   * Sample num_points from the underlying space and return these points as a
   * matrix where each column represents an underlying point.
   * @param num_threads the number of threads used to sample points.
   */
  virtual Eigen::MatrixXd SamplePoints(int num_points) const = 0;

  virtual ~PointSamplerBase() {}

 protected:
  // We put the copy/move/assignment constructors as protected to avoid copy
  // slicing. The inherited final subclasses should put them in public
  // functions.
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PointSamplerBase);
};

class AdjacencyMatrixBuilderBase {
 public:
  /**
   * Given points in the space, build the adjacency matrix of these points
   * describing the desired graphical relationship.
   * @param points to be used as the vertices of the graph.
   */
  virtual Eigen::SparseMatrix<bool>& BuildAdjacencyMatrix(
      const Eigen::Ref<const Eigen::MatrixXd>& points) const = 0;

  virtual ~AdjacencyMatrixBuilderBase() {}

 protected:
  // We put the copy/move/assignment constructors as protected to avoid copy
  // slicing. The inherited final subclasses should put them in public
  // functions.
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(AdjacencyMatrixBuilderBase);
};

/**
 * An abstract base class for implementing ways to build a convex set from a
 * clique of points
 */
class ConvexSetFromCliqueBuilderBase {
 public:
  /**
   * Given a set of points, build a convex set.
   */
  virtual std::unique_ptr<ConvexSet> BuildConvexSet(
      const Eigen::Ref<const Eigen::MatrixXd>& clique_points) const = 0;

  virtual ~ConvexSetFromCliqueBuilderBase() {}

 protected:
  // We put the copy/move/assignment constructors as protected to avoid copy
  // slicing. The inherited final subclasses should put them in public
  // functions.
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ConvexSetFromCliqueBuilderBase);
};
/**
 *
 * @param checker
 * @param points
 * @param options
 * @param minimum_clique_size
 * @param parallelize
 * @return
 */
class ApproximateConvexCoverFromCliqueCoverOptions {
 public:
  ApproximateConvexCoverFromCliqueCoverOptions(
      std::unique_ptr<CoverageCheckerBase> coverage_checker,
      std::unique_ptr<PointSamplerBase> point_sampler,
      std::unique_ptr<AdjacencyMatrixBuilderBase>
          adjacency_matrix_builder,
      std::unique_ptr<MaxCliqueSolverBase>
          max_clique_solver,
      std::unique_ptr<ConvexSetFromCliqueBuilderBase> set_builder,
      int num_sampled_points, int minimum_clique_size = 3,
      int num_threads = -1);

  [[nodiscard]] const CoverageCheckerBase* coverage_checker() const {
    return coverage_checker_.get();
  }

  [[nodiscard]] const PointSamplerBase* point_sampler() const {
    return point_sampler_.get();
  }

  [[nodiscard]] const AdjacencyMatrixBuilderBase* adjacency_matrix_builder()
      const {
    return adjacency_matrix_builder_.get();
  }

  [[nodiscard]] const MaxCliqueSolverBase* max_clique_solver()
      const {
    return max_clique_solver_.get();
  }

  [[nodiscard]] const ConvexSetFromCliqueBuilderBase* set_builder() const {
    return set_builder_.get();
  }

  [[nodiscard]] int num_sampled_points() const { return num_sampled_points_; }

  [[nodiscard]] int minimum_clique_size() const { return minimum_clique_size_; }

  [[nodiscard]] int num_threads() const { return num_threads_; }

 private:
  const std::unique_ptr<CoverageCheckerBase> coverage_checker_;

  const std::unique_ptr<PointSamplerBase> point_sampler_;

  const std::unique_ptr<AdjacencyMatrixBuilderBase> adjacency_matrix_builder_;

  const std::unique_ptr<MaxCliqueSolverBase> max_clique_solver_;

  const std::unique_ptr<ConvexSetFromCliqueBuilderBase> set_builder_;

  const int num_sampled_points_;

  const int minimum_clique_size_;

  const int num_threads_;
};

//class DummyClass {
//  DummyClass() = default;
//};
//
//class Tmp {
// public:
//  Tmp(std::unique_ptr<DummyClass> dummy);
//
// private:
//  const std::unique_ptr<DummyClass> dummy_;
//
//};

std::vector<std::unique_ptr<ConvexSet>> ApproximateConvexCoverFromCliqueCover(
    const ApproximateConvexCoverFromCliqueCoverOptions& options);

}  // namespace planning
}  // namespace drake