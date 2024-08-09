#include "drake/bindings/pydrake/common/eigen_pybind.h"
#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/bindings/pydrake/solvers/solvers_py.h"
#include "drake/bindings/pydrake/symbolic_types_pybind.h"
#include "drake/solvers/conic_standard_form.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefineSolversConicStandardForm(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::solvers;
  constexpr auto& doc = pydrake_doc.drake.solvers;
  const auto& cls_doc = doc.ConicStandardForm;
  py::class_<ConicStandardForm>(m, "ConicStandardForm", cls_doc.doc)
      .def(py::init<const MathematicalProgram&>(), py::arg("prog"),
          cls_doc.ctor.doc)
      .def("A", &ConicStandardForm::A, cls_doc.A.doc)
      // TODO(Alexandre.Amice) when pybind 11 supports sparse vectors simplify
      // this binding.
      .def(
          "b",
          [](const ConicStandardForm& self) {
            Eigen::SparseMatrix<double> b(self.b().rows(), 1);
            for (Eigen::SparseVector<double>::InnerIterator it(self.b()); it;
                 ++it) {
              b.insert(it.index(), 0) = it.value();
            }
            return b;
          },
          cls_doc.b.doc)
      .def(
          "c",
          [](const ConicStandardForm& self) {
            Eigen::SparseMatrix<double> c(self.c().rows(), 1);
            for (Eigen::SparseVector<double>::InnerIterator it(self.c()); it;
                 ++it) {
              c.insert(it.index(), 0) = it.value();
            }
            return c;
          },
          cls_doc.c.doc)
      .def("d", &ConicStandardForm::d, cls_doc.d.doc)
      .def("x", &ConicStandardForm::x, cls_doc.x.doc)
      .def("attributes_to_start_end_pairs",
          &ConicStandardForm::attributes_to_start_end_pairs,
          cls_doc.attributes_to_start_end_pairs.doc)
      .def("MakeProgram", &ConicStandardForm::MakeProgram,
          cls_doc.MakeProgram.doc);
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
