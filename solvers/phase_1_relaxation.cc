#include "drake/solvers/phase_1_relaxation.h"

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "drake/common/fmt_eigen.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/program_attribute.h"
namespace drake {
namespace solvers {

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using symbolic::Expression;
using symbolic::Variable;
using symbolic::Variables;

// Given a convex program
// min f(x) subject to
// gᵢ(x) ≤ 0
// relax to
// min 1ᵀs subject to
// gᵢ(x) ≤ sᵢ
// sᵢ ≥ 0
std::unique_ptr<MathematicalProgram> RelaxToPhase1Prog(
    const MathematicalProgram& prog,
    std::unordered_map<Binding<Constraint>, VectorXDecisionVariable>*
        constraint_to_slack_variables) {
  const double kInf = std::numeric_limits<double>::infinity();
  std::unique_ptr<MathematicalProgram> relaxed_prog =
      std::make_unique<MathematicalProgram>();
  relaxed_prog->AddDecisionVariables(prog.decision_variables());

  //  for (const auto& c : prog.GetAllCosts()) {
  //    relaxed_prog->AddCost(c);
  //  }

  auto add_s_vars = [&relaxed_prog, &kInf](int num_new_s) {
    const auto s = relaxed_prog->NewContinuousVariables(num_new_s, "s");
    relaxed_prog->AddBoundingBoxConstraint(
        Eigen::VectorXd::Zero(num_new_s),
        Eigen::VectorXd::Constant(num_new_s, kInf), s);
    relaxed_prog->AddLinearCost(Eigen::VectorXd::Ones(num_new_s), 0, s);
    return s;
  };

  for (const auto& c : prog.linear_equality_constraints()) {
    // Ax = b converts to Ax - s = b
    const int s_size = c.evaluator()->lower_bound().size();
    const auto s = add_s_vars(s_size);
    Eigen::MatrixXd A_new(c.evaluator()->GetDenseA().rows(),
                          c.evaluator()->GetDenseA().cols() + s_size);
    A_new.leftCols(c.evaluator()->GetDenseA().cols()) =
        c.evaluator()->GetDenseA();
    A_new.rightCols(s_size) = -Eigen::MatrixXd::Identity(s_size, s_size);
    VectorXDecisionVariable xs(A_new.cols());
    xs << c.variables(), s;
    relaxed_prog->AddLinearEqualityConstraint(A_new,
                                              c.evaluator()->lower_bound(), xs);
    A_new.rightCols(s_size) = Eigen::MatrixXd::Identity(s_size, s_size);
    relaxed_prog->AddLinearEqualityConstraint(A_new,
                                              c.evaluator()->lower_bound(), xs);
    constraint_to_slack_variables->emplace(c, s);
  }
  for (const auto& c : prog.bounding_box_constraints()) {
    // lb ≤ x ≤ ub converts to x + s ≥ lb and x - s ≤ ub
    const int s_size = c.evaluator()->lower_bound().size();
    const auto sl = add_s_vars(s_size);
    const auto su = add_s_vars(s_size);
    Eigen::MatrixXd Al(s_size, s_size + s_size);
    Al.leftCols(s_size) = Eigen::MatrixXd::Identity(s_size, s_size);
    Al.rightCols(s_size) = Eigen::MatrixXd::Identity(s_size, s_size);
    VectorXDecisionVariable xsl(s_size + s_size);
    xsl << c.variables(), sl;
    relaxed_prog->AddLinearConstraint(Al, c.evaluator()->lower_bound(),
                                      Eigen::VectorXd::Constant(s_size, kInf),
                                      xsl);

    Eigen::MatrixXd Au(s_size, s_size + s_size);
    Au.leftCols(s_size) = Eigen::MatrixXd::Identity(s_size, s_size);
    Au.rightCols(s_size) = -Eigen::MatrixXd::Identity(s_size, s_size);
    VectorXDecisionVariable xsu(Au.cols());
    xsu << c.variables(), su;
    relaxed_prog->AddLinearConstraint(Au,
                                      Eigen::VectorXd::Constant(s_size, -kInf),
                                      c.evaluator()->upper_bound(), xsu);
    VectorXDecisionVariable sl_su(s_size + s_size);
    sl_su << sl, su;
    constraint_to_slack_variables->emplace(c, sl_su);
  }
  for (const auto& c : prog.lorentz_cone_constraints()) {
    // Ax + b ∈ Lorentz cone is relaxed to  Ax + b + [s, 0, ... 0] in Lorentz
    // cone.
    const auto s = add_s_vars(1);
    Eigen::MatrixXd A_new(c.evaluator()->A().rows(),
                          c.evaluator()->A().cols() + 1);
    A_new.leftCols(c.evaluator()->A().cols()) = c.evaluator()->A();
    A_new.rightCols(1) = Eigen::VectorXd::Zero(A_new.rows());
    A_new(0, A_new.cols() - 1) = 1;
    VectorXDecisionVariable xs(A_new.cols());
    xs << c.variables(), s;
    relaxed_prog->AddLorentzConeConstraint(A_new, c.evaluator()->b(), xs);
    constraint_to_slack_variables->emplace(c, s);
  }
  for (const auto& c : prog.rotated_lorentz_cone_constraints()) {
    // Ax + b ∈ Rotated Lorentz cone is relaxed to  Ax + b + [s/2, s/2, ... 0]
    // in Rotated Lorentz cone.
    const auto s = add_s_vars(1);
    Eigen::MatrixXd A_new(c.evaluator()->A().rows(),
                          c.evaluator()->A().cols() + 1);
    A_new.leftCols(c.evaluator()->A().cols()) = c.evaluator()->A();
    A_new.rightCols(1) = Eigen::VectorXd::Zero(A_new.rows());
    A_new(0, A_new.cols() - 1) = 0.5;
    A_new(1, A_new.cols() - 1) = 0.5;
    VectorXDecisionVariable xs(A_new.cols());
    xs << c.variables(), s;
    relaxed_prog->AddRotatedLorentzConeConstraint(A_new, c.evaluator()->b(),
                                                  xs);
    constraint_to_slack_variables->emplace(c, s);
  }

  return relaxed_prog;
}

}  // namespace solvers
}  // namespace drake
