#include "drake/solvers/semidefinite_relaxation.h"

#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "drake/common/fmt_eigen.h"
#include "drake/common/ssize.h"
#include "drake/common/text_logging.h"
#include "drake/solvers/program_attribute.h"
#include "drake/solvers/semidefinite_relaxation_internal.h"


namespace drake {
namespace solvers {

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using symbolic::Expression;
using symbolic::Variable;
using symbolic::Variables;

namespace {

const double kInf = std::numeric_limits<double>::infinity();


// This constrains the minors of X corresponding to the groups of variables in
// sparsity_to_apply to be PSD.
void AddMinorsArePsdConstraints(
    const MatrixX<Variable>& X,
    const std::set<symbolic::Variables>& sparsity_to_apply,
    MathematicalProgram* prog) {
  unused(X, sparsity_to_apply, prog);
  //  std::cout << fmt::format("X size = {} x {}", X.rows(), X.cols()) <<
  //  std::endl;

  const VectorX<Variable> x{X.bottomRows<1>()};

  for (const auto& vars : sparsity_to_apply) {
    //    std::cout << vars << std::endl;
    // Find the indices of the minor for these variables.
    std::vector<int> cur_X_inds;
    cur_X_inds.reserve(ssize(vars));
    for (const auto& v : vars) {
      for (int i = 0; i < x.rows(); ++i) {
        if (v.equal_to(x(i))) {
          cur_X_inds.emplace_back(i);
          break;
        }
      }
    }
    //      std::cout << cur_X_inds << std::endl;
    MatrixX<Variable> minor(ssize(cur_X_inds), ssize(cur_X_inds));
    //      std::cout << fmt::format("minor size = {} x {}", minor.rows(),
    //                               minor.cols())
    //                << std::endl;
    for (int minor_r = 0; minor_r < ssize(cur_X_inds); ++minor_r) {
      for (int minor_c = minor_r; minor_c < ssize(cur_X_inds); ++minor_c) {
        //          std::cout
        //              << fmt::format(
        //                     "initializing minor idx = ({}, {}) with X_idx =
        //                     ({}, {})", minor_r, minor_c,
        //                     cur_X_inds.at(minor_r), cur_X_inds.at(minor_c))
        //              << std::endl;
        minor(minor_r, minor_c) =
            X(cur_X_inds.at(minor_r), cur_X_inds.at(minor_c));
        minor(minor_c, minor_r) =
            X(cur_X_inds.at(minor_c), cur_X_inds.at(minor_r));
      }
    }
    prog->AddPositiveSemidefiniteConstraint(minor);
  }
}

}  // namespace

std::unique_ptr<MathematicalProgram> MakeSemidefiniteRelaxation(
    const MathematicalProgram& prog,
    const SemidefiniteRelaxationSparsityType& sparsity) {
  std::set<symbolic::Variables> computed_sparsity_groups;
  auto prog_and_X = internal::MakeSemidefiniteRelaxationLinearConstraintsAndComputeMinorCliques(
      prog, sparsity != SemidefiniteRelaxationSparsityType::kDense
            ? std::optional{&computed_sparsity_groups}
            : std::nullopt);
  switch (sparsity) {
    case kDense:
      prog_and_X.first->AddPositiveSemidefiniteConstraint(prog_and_X.second);
      break;
    case kTermSparse:
      AddMinorsArePsdConstraints(prog_and_X.second, computed_sparsity_groups,
                                 prog_and_X.first.get());
      break;
      DRAKE_UNREACHABLE();
  }
  return std::move(prog_and_X.first);
}

std::unique_ptr<MathematicalProgram> MakeSemidefiniteRelaxation(
    const MathematicalProgram& prog,
    const std::map<symbolic::Variables, bool>& variables_to_enforce_sparsity) {
  auto prog_and_X =
      internal::MakeSemidefiniteRelaxationLinearConstraintsAndComputeMinorCliques(prog, std::nullopt);
  std::set<symbolic::Variables> sparsity_to_apply;
  const Variables prog_vars{prog.decision_variables()};
  for (const auto& [vars, use_one] : variables_to_enforce_sparsity) {
    Variables local_vars{vars};
    DRAKE_THROW_UNLESS(local_vars.IsSubsetOf(prog_vars));
    if (use_one) {
      local_vars.insert(prog_and_X.second.bottomRightCorner<1, 1>()(0));
    }
    internal::InsertIfNotSubsetOrReplaceIfSuperset(local_vars, &sparsity_to_apply);
  };
  //  for (const auto& vars : sparsity_to_apply) {
  //    std::cout << vars << std::endl;
  //  }
  AddMinorsArePsdConstraints(prog_and_X.second, sparsity_to_apply,
                             prog_and_X.first.get());
  return std::move(prog_and_X.first);
};
}  // namespace solvers
}  // namespace drake
