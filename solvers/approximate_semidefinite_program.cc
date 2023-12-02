#include "drake/solvers/approximate_semidefinite_program.h"

#include <functional>
#include <vector>

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {

namespace {

// Replace all the psd constraints of prog using one of
// prog->TightenPSDConstraintToDDConstraint,
// prog->TightenPSDConstraintToSDDConstraint,
// prog->RelaxPSDConstraintToDDDualConeConstraint,
// prog->RelaxPSDConstraintToSDDDualConeConstraint.
// Expect that prog and the implicit prog in psd_constraint_replacing_function
// are the same.
template <typename T>
void ApproximateProgram(
    const std::function<T(const Binding<PositiveSemidefiniteConstraint>&)>&
        psd_constraint_replacing_function,
    MathematicalProgram* prog) {
  const std::vector<Binding<PositiveSemidefiniteConstraint>> constraints =
      prog->positive_semidefinite_constraints();
  for (const auto& psd_constraint : constraints) {
    psd_constraint_replacing_function(psd_constraint);
  }
}
}  // namespace

void MakeDiagonallyDominantInnerApproximation(MathematicalProgram* prog) {
  auto fun =
      [&prog](const Binding<PositiveSemidefiniteConstraint>& constraint) {
        return prog->TightenPsdConstraintToDd(constraint);
      };
  ApproximateProgram<MatrixX<symbolic::Expression>>(fun, prog);
}

void MakeScaledDiagonallyDominantInnerApproximation(MathematicalProgram* prog) {
  auto fun =
      [&prog](const Binding<PositiveSemidefiniteConstraint>& constraint) {
        return prog->TightenPsdConstraintToSdd(constraint);
      };
  ApproximateProgram<std::vector<std::vector<Matrix2<symbolic::Variable>>>>(
      fun, prog);
}

void MakeDiagonallyDominantDualConeOuterApproximation(
    MathematicalProgram* prog) {
  auto fun =
      [&prog](const Binding<PositiveSemidefiniteConstraint>& constraint) {
        return prog->RelaxPsdConstraintToDdDualCone(constraint);
      };
  ApproximateProgram<Binding<LinearConstraint>>(fun, prog);
}

void MakeScaledDiagonallyDominantDualConeOuterApproximation(
    MathematicalProgram* prog) {
  auto fun =
      [&prog](const Binding<PositiveSemidefiniteConstraint>& constraint) {
        return prog->RelaxPsdConstraintToSddDualCone(constraint);
      };
  ApproximateProgram<std::vector<Binding<RotatedLorentzConeConstraint>>>(fun,
                                                                         prog);
}

}  // namespace solvers
}  // namespace drake
