#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/geometry/optimization_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/planning/adjacency_matrix_builder_base.h"
#include "drake/planning/approximate_convex_cover_builder_base.h"
#include "drake/planning/convex_set_from_clique_builder_base.h"
#include "drake/planning/coverage_checker_base.h"
#include "drake/planning/coverage_checker_via_bernoulli_test.h"
#include "drake/planning/iris_from_clique_cover.h"
#include "drake/planning/point_sampler_base.h"
#include "drake/planning/rejection_sampler.h"
#include "drake/planning/uniform_set_sampler.h"
#include "drake/geometry/optimization/hyperrectangle.h"
#include "drake/planning/visibility_graph_builder.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefinePlanningIrisFromCliqueCover(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::planning;
  constexpr auto& doc = pydrake_doc.drake.planning;

  // PointSamplerBase, UniformSetSampler, RejectionSampler
  {
    class PyPointSamplerBase : public py::wrapper<PointSamplerBase> {
     public:
      // Trampoline virtual methods.
      // The private virtual method of DoSamplePoints is made public to enable
      // Python implementations to override it.
      Eigen::MatrixXd DoSamplePoints(int num_points) override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::MatrixXd, PointSamplerBase, DoSamplePoints, num_points);
      }
    };
    const auto& cls_doc = doc.PointSamplerBase;
    py::class_<PointSamplerBase, PyPointSamplerBase>(
        m, "PointSamplerBase", cls_doc.doc)
        .def(py::init<>(), cls_doc.ctor.doc)
        .def("SamplePoints", &PointSamplerBase::SamplePoints,
            py::arg("num_points"), cls_doc.SamplePoints.doc);
  }
  {
    const auto& cls_doc = doc.UniformSetSampler;
    py::class_<UniformSetSampler<geometry::optimization::HPolyhedron>>
        hpoly_sampler(m, "UniformHPolyhedronSampler", cls_doc.doc);
    py::class_<UniformSetSampler<geometry::optimization::Hyperrectangle>>
        hyperrectangle_sampler(m, "UniformHyperrectangleSampler", cls_doc.doc);
  }
  {
//    const auto& cls_doc = doc.RejectionSampler;
//    py::class
  }

  // CoverageCheckerBase and CoverageCheckerViaBernoulliTest
  {
    class PyCoverageCheckerBase : public py::wrapper<CoverageCheckerBase> {
     public:
      // Trampoline virtual methods.
      // The private virtual method of DoCheckCoverage is made public to enable
      // Python implementations to override it.
      bool DoCheckCoverage(const ConvexSets& current_sets) const override {
        PYBIND11_OVERRIDE_PURE(
            bool, CoverageCheckerBase, DoCheckCoverage, current_sets);
      }
    };
    const auto& cls_doc = doc.CoverageCheckerBase;
    py::class_<CoverageCheckerBase, PyCoverageCheckerBase>(
        m, "CoverageCheckerBase", cls_doc.doc)
        .def(py::init<>(), cls_doc.ctor.doc)
        .def(
            "CheckCoverage",
            [](CoverageCheckerBase& self,
                const std::vector<geometry::optimization::ConvexSet*>& sets) {
              return self.CheckCoverage(CloneConvexSets(sets));
            },
            py::arg("sets"), cls_doc.CheckCoverage.doc);
  }

}  // DefinePlanningIrisFromCliqueCover
}  // namespace internal
}  // namespace pydrake
}  // namespace drake
