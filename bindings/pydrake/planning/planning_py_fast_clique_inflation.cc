#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/planning/iris/fast_clique_inflation.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefinePlanningFastCliqueInflation(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::planning;
  constexpr auto& doc = pydrake_doc.drake.planning;

  // FastCliqueInflationOptions
  const auto& cls_doc = doc.FastCliqueInflationOptions;
  py::class_<FastCliqueInflationOptions>(m, "FastCliqueInflationOptions", cls_doc.doc)
      .def(py::init<>())
      .def_readwrite("num_particles", &FastCliqueInflationOptions::num_particles,
          cls_doc.num_particles.doc)
      .def_readwrite("tau", &FastCliqueInflationOptions::tau, cls_doc.tau.doc)
      .def_readwrite("delta", &FastCliqueInflationOptions::delta, cls_doc.delta.doc)
      .def_readwrite("admissible_proportion_in_collision",
          &FastCliqueInflationOptions::admissible_proportion_in_collision,
          cls_doc.admissible_proportion_in_collision.doc)
      .def_readwrite("max_iterations_separating_planes",
          &FastCliqueInflationOptions::max_iterations_separating_planes,
          cls_doc.max_iterations_separating_planes.doc)
      .def_readwrite("max_separating_planes_per_iteration",
          &FastCliqueInflationOptions::max_separating_planes_per_iteration,
          cls_doc.max_separating_planes_per_iteration.doc)
      .def_readwrite("bisection_steps", &FastCliqueInflationOptions::bisection_steps,
          cls_doc.bisection_steps.doc)
      .def_readwrite(
          "parallelize", &FastCliqueInflationOptions::parallelize, cls_doc.parallelize.doc)
      .def_readwrite("verbose", &FastCliqueInflationOptions::verbose, cls_doc.verbose.doc)
      .def_readwrite("configuration_space_margin",
          &FastCliqueInflationOptions::configuration_space_margin,
          cls_doc.configuration_space_margin.doc)
      .def_readwrite(
          "random_seed", &FastCliqueInflationOptions::random_seed, cls_doc.random_seed.doc)
      .def("__repr__", [](const FastCliqueInflationOptions& self) {
        return py::str(
            "FastCliqueInflationOptions("
            "num_particles={}, "
            "tau={}, "
            "delta={}, "
            "admissible_proportion_in_collision={}, "
            "max_iterations_separating_planes={}, "
            "max_separating_planes_per_iteration={}, "
            "bisection_steps={}, "
            "parallelize={}, "
            "verbose={}, "
            "configuration_space_margin={}, "
            "random_seed={} "
            ")")
            .format(self.num_particles, self.tau, self.delta,
                self.admissible_proportion_in_collision,
                self.max_iterations_separating_planes,
                self.max_separating_planes_per_iteration, self.bisection_steps,
                self.parallelize, self.verbose,
                self.configuration_space_margin,
                self.random_seed);
      });

  m.def("FastCliqueInflation", &FastCliqueInflation, py::arg("checker"),
      py::arg("clique"), py::arg("domain"),
      py::arg("options") = FastCliqueInflationOptions(), doc.FastCliqueInflation.doc);
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake