#include "drake/planning/coverage_checker_via_bernoulli_test.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <thread>
#include <iostream>

namespace drake {
namespace planning {
using geometry::optimization::ConvexSets;

CoverageCheckerViaBernoulliTest::CoverageCheckerViaBernoulliTest(
    const double alpha, const int num_points_per_check,
    std::shared_ptr<PointSamplerBase> point_sampler, const int num_threads,
    const double point_in_set_tol)
    : alpha_(alpha),
      num_points_per_check_(num_points_per_check),
      point_sampler_(std::move(point_sampler)),
      num_threads_{num_threads},
      point_in_set_tol_{point_in_set_tol} {};

double CoverageCheckerViaBernoulliTest::GetSampledCoverageFraction(
    const ConvexSets& current_sets) const {
  Eigen::MatrixXd sampled_points =
      point_sampler_->SamplePoints(num_points_per_check_);
  // Leave this count as a double since we will use it to perform division
  // later.
  std::atomic<double> num_in_sets{0};
  const int max_concurrency{
      std::max(static_cast<int>(std::thread::hardware_concurrency()), 1)};
  const int num_threads{num_threads_ > 0 ? num_threads_ : max_concurrency};
#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
  for (int i = 0; i < sampled_points.cols(); ++i) {
    for (const auto& set : current_sets) {
      if (set->PointInSet(sampled_points.col(i), point_in_set_tol_)) {
        num_in_sets.fetch_add(1);
        break;
      }
    }
  }
  std::cout << "current coverage = " << num_in_sets / num_points_per_check_ << std::endl;
  return num_in_sets / num_points_per_check_;
}

bool CoverageCheckerViaBernoulliTest::DoCheckCoverage(
    const ConvexSets& current_sets) const {
  return GetSampledCoverageFraction(current_sets) >= alpha_;
}

}  // namespace planning
}  // namespace drake
