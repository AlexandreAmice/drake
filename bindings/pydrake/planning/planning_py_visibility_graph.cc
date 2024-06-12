#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/planning/visibility_graph.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefinePlanningVisibilityGraph(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::planning;
  constexpr auto& doc = pydrake_doc.drake.planning;

  m.def("VisibilityGraph",
      py::overload_cast<const CollisionChecker&,
          const Eigen::Ref<const Eigen::MatrixXd>&, const Parallelism>(
          &planning::VisibilityGraph),
      py::arg("checker"), py::arg("points"), 
      py::arg("parallelize") = Parallelism::Max(),
      py::call_guard<py::gil_scoped_release>()
  , doc.VisibilityGraph.doc_3args);

  m.def("VisibilityGraph",
      py::overload_cast<
          std::function<void(const int, const int64_t, std::vector<uint8_t>*)>,
          std::function<void(const int, const int64_t,
              const std::vector<uint8_t>&, const int,
              std::vector<std::vector<int>>*)>,
          const Eigen::Ref<const Eigen::MatrixXd>&,
          const Parallelism>(&planning::VisibilityGraph), 
          py::arg("point_check_work"), 
          py::arg("edge_check_work"),
      py::arg("points"), py::arg("parallelize") = Parallelism::Max(),
      py::call_guard<py::gil_scoped_release>()
   , doc.VisibilityGraph.doc_4args);
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
