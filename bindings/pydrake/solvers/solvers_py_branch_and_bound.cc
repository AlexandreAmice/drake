#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/bindings/pydrake/symbolic_types_pybind.h"
#include "drake/solvers/branch_and_bound.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefineSolversBranchAndBound(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::solvers;
  constexpr auto& doc = pydrake_doc.drake.solvers;

  {
    using Class = MixedIntegerBranchAndBound;
    constexpr auto& cls_doc = doc.MixedIntegerBranchAndBound;
    py::class_<Class> bnb_class(m, "MixedIntegerBranchAndBound", cls_doc.doc);
    bnb_class
        .def(py::init<const MathematicalProgram&, const SolverId&>(),
            py::arg("prog"), py::arg("solver_id"), cls_doc.ctor.doc)
        .def("Solve", &Class::Solve, cls_doc.Solve.doc)
        .def("GetOptimalCost", &Class::GetOptimalCost,
            cls_doc.GetOptimalCost.doc)
        .def(
            "GetSubOptimalCost",
            [](const Class& self, int nth_suboptimal_cost) {
              return self.GetSubOptimalCost(nth_suboptimal_cost);
            },
            py::arg("nth_suboptimal_cost"), cls_doc.GetSubOptimalCost.doc)
        .def(
            "GetSolution",
            [](const Class& self, const symbolic::Variable& mip_var,
                int nth_best_solution) {
              return self.GetSolution(mip_var, nth_best_solution);
            },
            py::arg("mip_var"), py::arg("nth_best_solution") = 0,
            cls_doc.GetSolution.doc_2args_mip_var_nth_best_solution)
        .def(
            "GetSolution",
            [](const Class& self, const VectorXDecisionVariable& mip_vars,
                int nth_best_solution) {
              return self.GetSolution(mip_vars, nth_best_solution);
            },
            py::arg("mip_vars"), py::arg("nth_best_solution") = 0,
            cls_doc.GetSolution.doc_2args_constEigenMatrixBase_int)
        .def(
            "GetSolution",
            [](const Class& self, const MatrixXDecisionVariable& mip_vars,
                int nth_best_solution) {
              return self.GetSolution(mip_vars, nth_best_solution);
            },
            py::arg("mip_vars"), py::arg("nth_best_solution") = 0,
            cls_doc.GetSolution.doc_2args_constEigenMatrixBase_int)
        .def("SetUserDefinedNodeSelectionFunction",
            &Class::SetUserDefinedNodeSelectionFunction,
            py::arg("fun"),
            cls_doc.SetUserDefinedNodeSelectionFunction.doc)
        .def("SetNodeSelectionMethod", &Class::SetNodeSelectionMethod,
            cls_doc.SetNodeSelectionMethod.doc)
        .def("root", &Class::root);
    py::enum_<Class::NodeSelectionMethod>(
        bnb_class, "NodeSelectionMethod", cls_doc.NodeSelectionMethod.doc)
        .value("kUserDefined", Class::NodeSelectionMethod::kUserDefined)
        .value("kDepthFirst", Class::NodeSelectionMethod::kDepthFirst)
        .value("kMinLowerBound", Class::NodeSelectionMethod::kMinLowerBound);
  }
  {
    using Class = MixedIntegerBranchAndBoundNode;
    constexpr auto& cls_doc = doc.MixedIntegerBranchAndBoundNode;
    py::class_<Class> bnb_node_class(m, "MixedIntegerBranchAndBoundNode", cls_doc.doc);
    bnb_node_class
        .def("left_child", &Class::left_child, py_rvp::reference, cls_doc.left_child.doc)
        .def("right_child", &Class::right_child, py_rvp::reference, cls_doc.right_child.doc);
  }
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
