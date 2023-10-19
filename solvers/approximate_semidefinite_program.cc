#include <functional>
#include <memory>

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
  for (const auto& psd_constraint : prog->positive_semidefinite_constraints()) {
    psd_constraint_replacing_function(psd_constraint);
  }
}
}  // namespace

void MakeDiagonallyDominantInnerApproximation(
    std::unique_ptr<MathematicalProgram> prog) {
  auto fun = [&prog](const Binding<PositiveSemidefiniteConstraint>& constraint) {
    return prog.get()->TightenPSDConstraintToDDConstraint(constraint);
  };
  ApproximateProgram<MatrixX<symbolic::Expression>>(fun, prog.get());
}

 void
 MakeScaledDiagonallyDominantInnerApproximation(std::unique_ptr<MathematicalProgram>
 prog){
  auto fun = [&prog](const Binding<PositiveSemidefiniteConstraint>& constraint) {
    return prog.get()->TightenPSDConstraintToSDDConstraint(constraint);
  };
  ApproximateProgram<std::vector<std::vector<Matrix2<symbolic::Variable>>>>(fun, prog.get());
};

 void
 MakeDiagonallyDominantDualConeOuterApproximation(
    std::unique_ptr<MathematicalProgram> prog){
  auto fun = [&prog](const Binding<PositiveSemidefiniteConstraint>& constraint) {
    return prog.get()->RelaxPSDConstraintToDDDualConeConstraint(constraint);
  };
  ApproximateProgram<Binding<LinearConstraint>>(fun, prog.get());
}

void
 MakeScaledDiagonallyDominantDualConeOuterApproximation(
    std::unique_ptr<MathematicalProgram> prog){
auto fun = [&prog](const Binding<PositiveSemidefiniteConstraint>& constraint) {
    return prog.get()->RelaxPSDConstraintToSDDDualConeConstraint(constraint);
  };
  ApproximateProgram<std::vector<Binding<RotatedLorentzConeConstraint>>>(fun, prog.get());
}

}  // namespace solvers
}  // namespace drake
