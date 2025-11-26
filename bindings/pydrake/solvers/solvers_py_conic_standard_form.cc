#include "drake/bindings/generated_docstrings/solvers.h"
#include "drake/bindings/pydrake/common/eigen_pybind.h"
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
  // constexpr auto& doc = pydrake_doc_solvers.drake.solvers;
  // const auto& cls_doc = doc.ConicStandardForm;
  py::class_<ConicStandardFormOptions>(m, "ConicStandardFormOptions")
      .def(py::init(
               [](bool keep_quadratic_costs,
                   bool parse_bounding_box_constraints_as_positive_orthant) {
                 ConicStandardFormOptions options;
                 options.keep_quadratic_costs = keep_quadratic_costs;
                 options.parse_bounding_box_constraints_as_positive_orthant =
                     parse_bounding_box_constraints_as_positive_orthant;
                 return options;
               }),
          py::arg("keep_quadratic_costs") = false,
          py::arg("parse_bounding_box_constraints_as_positive_orthant") = true)
      .def_readwrite("keep_quadratic_costs",
          &ConicStandardFormOptions::keep_quadratic_costs)
      .def_readwrite("parse_bounding_box_constraints_as_positive_orthant",
          &ConicStandardFormOptions::
              parse_bounding_box_constraints_as_positive_orthant);
  py::class_<ConicStandardForm>(m, "ConicStandardForm")
      .def(py::init<const MathematicalProgram&>(), py::arg("prog"))
      .def(py::init<const MathematicalProgram&, ConicStandardFormOptions>(),
          py::arg("prog"), py::arg("options"))
      .def("A", &ConicStandardForm::A)
      // TODO(Alexandre.Amice) when pybind 11 supports sparse vectors simplify
      // this binding.
      .def("b",
          [](const ConicStandardForm& self) {
            Eigen::SparseMatrix<double> b(self.b().rows(), 1);
            for (Eigen::SparseVector<double>::InnerIterator it(self.b()); it;
                ++it) {
              b.insert(it.index(), 0) = it.value();
            }
            return b;
          })
      .def("c",
          [](const ConicStandardForm& self) {
            Eigen::SparseMatrix<double> c(self.c().rows(), 1);
            for (Eigen::SparseVector<double>::InnerIterator it(self.c()); it;
                ++it) {
              c.insert(it.index(), 0) = it.value();
            }
            return c;
          })
      .def("d", &ConicStandardForm::d)
      .def("x", &ConicStandardForm::x)
      .def("attributes_to_start_end_pairs",
          &ConicStandardForm::attributes_to_start_end_pairs)
      .def("MakeProgram", &ConicStandardForm::MakeProgram);
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
