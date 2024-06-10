#pragma once

#include <memory>
#include <unordered_map>

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {

std::unique_ptr<MathematicalProgram> RelaxToPhase1Prog(
    const MathematicalProgram& prog,
     std::unordered_map<Binding<Constraint>, VectorXDecisionVariable>*
        constraint_to_slack_variables);

}  // namespace solvers
}  // namespace drake