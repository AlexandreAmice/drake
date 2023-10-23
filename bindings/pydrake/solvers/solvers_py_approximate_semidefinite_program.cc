#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/bindings/pydrake/solvers/solvers_py.h"
#include "drake/solvers/approximate_semidefinite_program.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefineApproximateSemidefiniteProgram(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::solvers;
  constexpr auto& doc = pydrake_doc.drake.solvers;

  m.def("MakeDiagonallyDominantInnerApproximation",
      &solvers::MakeDiagonallyDominantInnerApproximation, py::arg("prog"),
      doc.MakeDiagonallyDominantInnerApproximation.doc);
  m.def("MakeScaledDiagonallyDominantInnerApproximation",
      &solvers::MakeScaledDiagonallyDominantInnerApproximation, py::arg("prog"),
      doc.MakeScaledDiagonallyDominantInnerApproximation.doc);
  m.def("MakeDiagonallyDominantDualConeOuterApproximation",
      &solvers::MakeDiagonallyDominantDualConeOuterApproximation, py::arg("prog"),
      doc.MakeDiagonallyDominantDualConeOuterApproximation.doc);
  m.def("MakeScaledDiagonallyDominantDualConeOuterApproximation",
      &solvers::MakeScaledDiagonallyDominantDualConeOuterApproximation, py::arg("prog"),
      doc.MakeScaledDiagonallyDominantDualConeOuterApproximation.doc);

}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
