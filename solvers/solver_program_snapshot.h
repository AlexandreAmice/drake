#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {
namespace internal {

// Borrowed views of current evaluator coefficients, never of Eigen's data
// buffers. Obtain new views after an evaluator changes its storage.
struct SolverBindingCoefficients {
  const Eigen::SparseMatrix<double>* A{};
  const Eigen::MatrixXd* Q{};
  const Eigen::VectorXd* v{};
  const Eigen::VectorXd* w{};
  double constant{};
  bool recognized{true};
};

class SolverCoefficientReader {
 public:
  virtual ~SolverCoefficientReader() = default;
  virtual SolverBindingCoefficients Read() const = 0;
};

// Exact numerical snapshots of the mutable evaluators translated by OSQP and
// SCS. Comparisons do not construct new coefficient matrices. Unknown types
// require a rebuild; immutable SCS evaluators only need binding checks.
struct SolverBindingSnapshot {
  explicit SolverBindingSnapshot(
      Binding<EvaluatorBase> binding_in, const MathematicalProgram& prog,
      std::unique_ptr<SolverCoefficientReader> reader);
  const Eigen::SparseMatrix<double>* current_A() const;
  const Eigen::MatrixXd* current_Q() const;
  const Eigen::VectorXd* current_v() const;
  const Eigen::VectorXd* current_w() const;
  double current_constant() const;
  bool MatrixMatches() const;
  bool VectorsMatch() const;
  bool DimensionsMatch() const;
  void Refresh();
  void RefreshVectors();

  Binding<EvaluatorBase> binding;
  std::vector<int> variable_indices;
  Eigen::SparseMatrix<double> A;
  Eigen::MatrixXd Q;
  Eigen::VectorXd v, w;
  double constant{};
  bool recognized{};

 private:
  std::unique_ptr<SolverCoefficientReader> reader_;
};

class SolverProgramSnapshot {
 public:
  explicit SolverProgramSnapshot(const MathematicalProgram& prog);
  std::string CheckStructure(const MathematicalProgram& prog) const;
  bool IsUnchanged() const;
  std::vector<SolverBindingSnapshot> costs;
  std::vector<SolverBindingSnapshot> constraints;

 private:
  VectorXDecisionVariable variables_;
  std::unordered_map<int, double> scaling_;
};

}  // namespace internal
}  // namespace solvers
}  // namespace drake
