#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace geometry {
namespace optimization {
class AxisAlignedBox {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(AxisAlignedBox)

  AxisAlignedBox(const Eigen::Ref<const Eigen::VectorXd>& lo,
                 const Eigen::Ref<const Eigen::VectorXd>& up);

  /** Compute the tightest outer box that contains the polyhedron C*x<=d. Throw
   * an error if C*x<=d is unbounded.
   */
  static AxisAlignedBox OuterBox(const Eigen::Ref<const Eigen::MatrixXd>& C,
                                 const Eigen::Ref<const Eigen::VectorXd>& d);

  AxisAlignedBox Scale(double factor) const;

  const Eigen::VectorXd& lo() const { return lo_; }

  const Eigen::VectorXd& up() const { return up_; }

 private:
  Eigen::VectorXd lo_;
  Eigen::VectorXd up_;
};

/**
 * Find a box lo<=x<=up contained in the polyhedron C*x<=d.
 */
class FindInscribedBox {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FindInscribedBox)

  /**
   * Find a box contained in the polyhedron C*x<=d while avoiding all obstacles.
   * @param outer_box An optional argument. This box contains the polyhedron
   * C*x<=d.
   */
  FindInscribedBox(const Eigen::Ref<const Eigen::MatrixXd>& C,
                   const Eigen::Ref<const Eigen::VectorXd>& d,
                   std::vector<AxisAlignedBox> obstacles,
                   const std::optional<AxisAlignedBox>& outer_box);

  const solvers::MathematicalProgram& prog() const { return *prog_; }

  solvers::MathematicalProgram* mutable_prog() { return prog_.get(); }

  const VectorX<symbolic::Variable>& box_lo() const { return box_lo_; }

  const VectorX<symbolic::Variable>& box_up() const { return box_up_; }

 private:
  std::unique_ptr<solvers::MathematicalProgram> prog_;
  Eigen::MatrixXd C_;
  Eigen::VectorXd d_;
  std::vector<AxisAlignedBox> obstacles_;
  VectorX<symbolic::Variable> box_lo_;
  VectorX<symbolic::Variable> box_up_;
};

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
