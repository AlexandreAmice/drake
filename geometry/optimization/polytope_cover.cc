#include "drake/geometry/optimization/polytope_cover.h"

#include <limits>
#include <utility>

#include "drake/math/gray_code.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace geometry {
namespace optimization {

const double kInf = std::numeric_limits<double>::infinity();

AxisAlignedBox::AxisAlignedBox(const Eigen::Ref<const Eigen::VectorXd>& lo,
                               const Eigen::Ref<const Eigen::VectorXd>& up)
    : lo_(lo), up_(up) {
  DRAKE_DEMAND((lo.array() <= up.array()).all());
}

AxisAlignedBox AxisAlignedBox::OuterBox(
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d) {
  solvers::MathematicalProgram prog;
  const int nx = C.cols();
  auto x = prog.NewContinuousVariables(nx);
  prog.AddLinearConstraint(C, Eigen::VectorXd::Constant(d.rows(), -kInf), d, x);
  Eigen::VectorXd coeff = Eigen::VectorXd::Zero(nx);
  auto cost = prog.AddLinearCost(coeff, x);
  Eigen::VectorXd box_lo(nx);
  Eigen::VectorXd box_up(nx);
  for (int i = 0; i < nx; ++i) {
    coeff.setZero();
    coeff(i) = 1;
    cost.evaluator()->UpdateCoefficients(coeff, 0);
    auto result = solvers::Solve(prog);
    if (!result.is_success()) {
      throw std::invalid_argument(
          "OuterBox fails, please check if the input polyhedron is bounded.");
    }
    box_lo(i) = result.get_optimal_cost();
    coeff(i) = -1;
    cost.evaluator()->UpdateCoefficients(coeff, 0);
    result = solvers::Solve(prog);
    if (!result.is_success()) {
      throw std::invalid_argument(
          "OuterBox fails, please check if the input polyhedron is bounded.");
    }
    box_up(i) = -result.get_optimal_cost();
  }
  return AxisAlignedBox(box_lo, box_up);
}

AxisAlignedBox AxisAlignedBox::Scale(double factor) const {
  DRAKE_DEMAND(factor >= 0);
  const Eigen::VectorXd center = (lo_ + up_) / 2;
  return AxisAlignedBox(center - (up_ - lo_) / 2 * factor,
                        center + (up_ - lo_) / 2 * factor);
}

namespace {
// Add the constraint that box_lo <= x <= box_up doesn't overlap with the other
// obstacle box. The box box_lo <= x <= box_up has to lie within outer_box.
VectorX<symbolic::Variable> AddBoxNoOverlapping(
    solvers::MathematicalProgram* prog,
    const VectorX<symbolic::Variable>& box_lo,
    const VectorX<symbolic::Variable>& box_up, const AxisAlignedBox& obstacle,
    const AxisAlignedBox& outer_box) {
  // Two boxes don't overlap if at least on one dimenstion we have box_lo1(i) >=
  // box_up2(i) or box_up1(i) <= box_lo2(i). We use a binary variable to
  // indicate whether this condition is true.
  const int dim = box_lo.rows();
  const auto b = prog->NewBinaryVariables(2 * dim);
  for (int i = 0; i < dim; ++i) {
    // b(2i) = 1 implies box_lo(i) >= obstacle.up(i).
    // We write box_lo(i) >= outer_box.lo(i) + (obstale.up(i) - outer_box.lo(i))
    // * b(2i)
    prog->AddLinearConstraint(
        Eigen::RowVector2d(1, outer_box.lo()(i) - obstacle.up()(i)),
        outer_box.lo()(i), kInf,
        Vector2<symbolic::Variable>(box_lo(i), b(2 * i)));
    // b(2i+1) = 1 implies box_up(i) <= obstacle.lo(i).
    // We write box_up(i) <= outer_box.up(i) - (outer_box.up(i) -
    // obstacle.lo(i)) * b(2i+1)
    prog->AddLinearConstraint(
        Eigen::RowVector2d(1, outer_box.up()(i) - obstacle.lo()(i)), -kInf,
        outer_box.up()(i),
        Vector2<symbolic::Variable>(box_up(i), b(2 * i + 1)));
  }
  prog->AddLinearConstraint(Eigen::RowVectorXd::Ones(2 * dim), 1, kInf, b);
  return b;
}
}  // namespace

FindInscribedBox::FindInscribedBox(
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    std::vector<AxisAlignedBox> obstacles,
    const std::optional<AxisAlignedBox>& outer_box)
    : prog_{new solvers::MathematicalProgram()},
      C_{C},
      d_{d},
      obstacles_{std::move(obstacles)} {
  const int dim = C_.cols();
  box_lo_ = prog_->NewContinuousVariables(dim, "lo");
  box_up_ = prog_->NewContinuousVariables(dim, "up");
  // Add the constraint that box_up >= box_lo
  Eigen::MatrixXd A(dim, 2 * dim);
  A << Eigen::MatrixXd::Identity(dim, dim),
      -Eigen::MatrixXd::Identity(dim, dim);
  prog_->AddLinearConstraint(A, Eigen::VectorXd::Zero(dim),
                             Eigen::VectorXd::Constant(dim, kInf),
                             {box_up_, box_lo_});
  // Add the constraint that the box is in the polytope, namely all vertices of
  // the box in the polytope. To obtain all vertices, we need all of power(2,
  // dim) permutations to select between box_lo(i) and box_up(i) along each
  // dimension. Such permuation can be obtained through the gray code with
  // num_digits = dim, then for each gray code, if code(i) = 0 then we take
  // box_lo_(i); if code(i) = 1 then we take box_up_(i).
  const auto gray_code = math::CalculateReflectedGrayCodes(dim);
  VectorX<symbolic::Variable> vertex(dim);
  const Eigen::VectorXd lb_inf = Eigen::VectorXd::Constant(d.rows(), -kInf);
  for (int i = 0; i < gray_code.rows(); ++i) {
    for (int j = 0; j < dim; ++j) {
      vertex(j) = gray_code(i, j) == 0 ? box_lo_(j) : box_up_(j);
    }
    prog_->AddLinearConstraint(C, lb_inf, d, vertex);
  }
  // Now add the constraint that the box doesn't overlap with any obstacles.
  // First we compute the outer box if the user doesn't provide one
  if (!obstacles_.empty()) {
    const AxisAlignedBox outer_box_ = outer_box.has_value()
                                          ? outer_box.value()
                                          : AxisAlignedBox::OuterBox(C_, d_);
    for (const auto& obstacle : obstacles_) {
      AddBoxNoOverlapping(prog_.get(), box_lo_, box_up_, obstacle, outer_box_);
    }
  }
}
}  // namespace optimization
}  // namespace geometry
}  // namespace drake
