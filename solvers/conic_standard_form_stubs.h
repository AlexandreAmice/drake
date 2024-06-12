#pragma once

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {

std::unordered_map<Binding<L2NormCost>, symbolic::Variable>
ParseL2NormCostsToEpigraphForm(MathematicalProgram* prog);

std::unordered_map<Binding<L1NormCost>, symbolic::Variable>
ParseL1NormCostsToEpigraphForm(MathematicalProgram* prog);

std::unordered_map<Binding<QuadraticCost>, symbolic::Variable>
ParseQuadraticCostsToEpigraphForm(MathematicalProgram* prog);

/* Most convex solvers require only support linear and quadratic costs when
operating with nonlinear constraints. This removes costs and adds variables and
constraints as needed by the solvers. */
void ParseNonlinearCostsToEpigraphForm(MathematicalProgram* prog);
}  // namespace solvers
}  // namespace drake
