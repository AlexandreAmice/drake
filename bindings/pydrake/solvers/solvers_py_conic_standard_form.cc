#include "drake/bindings/pydrake/common/eigen_pybind.h"
#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/bindings/pydrake/solvers/solvers_py.h"
#include "drake/bindings/pydrake/symbolic_types_pybind.h"
#include "drake/solvers/dual_convex_program.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefineSolversConicStandardForm(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::solvers;
  constexpr auto& doc = pydrake_doc.drake.solvers;

  m.def(
      "ParseToConicStandardForm",
      [](const MathematicalProgram& prog) {
        Eigen::SparseVector<double> c;
        double d{0};
        Eigen::SparseMatrix<double> A;
        Eigen::SparseVector<double> b;
        std::unordered_map<ProgramAttribute, std::vector<std::pair<int, int>>>
            attributes_to_start_end_pairs;
        ParseToConicStandardForm(prog, &c, &d, &A, &b,
                           &attributes_to_start_end_pairs);
        return constraint_to_dual_variable_map;
      },
      py::arg("prog"), doc.CreateDualConvexProgram.doc);
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
