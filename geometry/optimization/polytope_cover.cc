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

double AxisAlignedBox::volume() const { return (up_ - lo_).array().prod(); }

FindInscribedBox::FindInscribedBox(const Eigen::Ref<const Eigen::MatrixXd>& C,
                                   const Eigen::Ref<const Eigen::VectorXd>& d,
                                   const std::vector<AxisAlignedBox>& obstacles,
                                   std::optional<AxisAlignedBox> outer_box)
    : prog_{new solvers::MathematicalProgram()},
      C_{C},
      d_{d},
      obstacles_{},
      outer_box_{nullptr} {
  // First we compute the outer box if the user doesn't provide one
  if (outer_box.has_value()) {
    outer_box_ = std::make_unique<AxisAlignedBox>(std::move(outer_box.value()));
  } else {
    const AxisAlignedBox outer_box_tmp = AxisAlignedBox::OuterBox(C_, d_);
    outer_box_ = std::make_unique<AxisAlignedBox>(std::move(outer_box_tmp));
  }
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
  if (!obstacles.empty()) {
    obstacles_.reserve(obstacles.size());
    for (const auto& obstacle : obstacles) {
      AddObstacle(obstacle);
    }
  }
}

VectorX<symbolic::Variable> FindInscribedBox::AddObstacle(
    const AxisAlignedBox& obstacle) {
  obstacles_.push_back(obstacle);
  // Two boxes don't overlap if at least on one dimenstion we have box_lo1(i) >=
  // box_up2(i) or box_up1(i) <= box_lo2(i). We use a binary variable to
  // indicate whether this condition is true.
  const int dim = box_lo_.rows();
  const auto b = prog_->NewBinaryVariables(2 * dim);
  for (int i = 0; i < dim; ++i) {
    // b(2i) = 1 implies box_lo(i) >= obstacle.up(i).
    // We write box_lo(i) >= outer_box.lo(i) + (obstacle.up(i) -
    // outer_box.lo(i))
    // * b(2i)
    prog_->AddLinearConstraint(
        Eigen::RowVector2d(1, outer_box_->lo()(i) - obstacle.up()(i)),
        outer_box_->lo()(i), kInf,
        Vector2<symbolic::Variable>(box_lo_(i), b(2 * i)));
    // b(2i+1) = 1 implies box_up(i) <= obstacle.lo(i).
    // We write box_up(i) <= outer_box.up(i) - (outer_box.up(i) -
    // obstacle.lo(i)) * b(2i+1)
    prog_->AddLinearConstraint(
        Eigen::RowVector2d(1, outer_box_->up()(i) - obstacle.lo()(i)), -kInf,
        outer_box_->up()(i),
        Vector2<symbolic::Variable>(box_up_(i), b(2 * i + 1)));
    // It is impossible if box_lo(i) > obstacle.up(i) and box_up(i) <
    // obstacle.lo(i), so we can add b(2i) + b(2i+1) <=1
    prog_->AddLinearConstraint(Eigen::RowVector2d::Ones(), -kInf, 1,
                               b.segment<2>(2 * i));
  }
  prog_->AddLinearConstraint(Eigen::RowVectorXd::Ones(2 * dim), 1, kInf, b);
  return b;
}

void FindInscribedBox::MaximizeBoxVolume() {
  // Maximize product(box_up(i) - box_lo(i)).
  const int dim = box_lo_.rows();
  Eigen::MatrixXd A(dim, 2 * dim);
  A.leftCols(dim) = Eigen::MatrixXd::Identity(dim, dim);
  A.rightCols(dim) = -Eigen::MatrixXd::Identity(dim, dim);
  VectorX<symbolic::Variable> vars(2 * dim);
  vars << box_up_, box_lo_;
  prog_->AddMaximizeGeometricMeanCost(A, Eigen::VectorXd::Zero(dim), vars);
}
}  // namespace optimization
}  // namespace geometry
}  // namespace drake
