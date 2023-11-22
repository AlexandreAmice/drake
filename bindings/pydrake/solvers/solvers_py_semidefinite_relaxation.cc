#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/bindings/pydrake/solvers/solvers_py.h"
#include "drake/solvers/semidefinite_relaxation.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefineSolversSemidefiniteRelaxation(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::solvers;
  constexpr auto& doc = pydrake_doc.drake.solvers;
  unused(doc);

  m.def("MakeSemidefiniteRelaxation",
      py::overload_cast<const MathematicalProgram&, bool>(
          &solvers::MakeSemidefiniteRelaxation),
      py::arg("prog"), py::arg("use_term_sparsity") = true);

  m.def("MakeSemidefiniteRelaxation",
      py::overload_cast<const MathematicalProgram&,
          const std::map<symbolic::Variables, bool>&>(
          &solvers::MakeSemidefiniteRelaxation),
      py::arg("prog"), py::arg("variables_to_enforce_sparsity"));
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
