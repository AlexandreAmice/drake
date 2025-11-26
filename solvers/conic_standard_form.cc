#include "drake/solvers/conic_standard_form.h"

#include <initializer_list>
#include <limits>
#include <memory>
#include <string>

#include "drake/common/never_destroyed.h"
#include "drake/common/ssize.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/aggregate_costs_constraints.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {
using Eigen::MatrixXd;
using Eigen::VectorXd;
using symbolic::Variable;
const double kInf = std::numeric_limits<double>::infinity();

namespace {
// If the program is compatible with conic standard forms.
void CheckSupported(const MathematicalProgram& prog) {
  std::string unsupported_message{};
  const ProgramAttributes supported_attributes(
      std::initializer_list<ProgramAttribute>{
          // Supported Constraints.
          ProgramAttribute::kLinearEqualityConstraint,
          ProgramAttribute::kLinearConstraint,
          ProgramAttribute::kLorentzConeConstraint,
          ProgramAttribute::kRotatedLorentzConeConstraint,
          ProgramAttribute::kPositiveSemidefiniteConstraint,
          // Supported Costs.
          ProgramAttribute::kLinearCost,
          ProgramAttribute::kQuadraticCost,
          ProgramAttribute::kL2NormCost,
      });
  if (!AreRequiredAttributesSupported(prog.required_capabilities(),
                                      supported_attributes,
                                      &unsupported_message)) {
    throw std::runtime_error(fmt::format(
        "ConicStandardForm() does not (yet) support this program: {}.",
        unsupported_message));
  }
}

}  // namespace

ConicStandardForm::ConicStandardForm(
    const MathematicalProgram& prog_input,
    ConicStandardFormOptions constructor_options)
    : x_{prog_input.decision_variables()} {
  auto prog = prog_input.Clone();
  CheckSupported(*prog);

  internal::ConvexConstraintAggregationInfo info;
  internal::ConvexConstraintAggregationOptions options;
  options.cast_rotated_lorentz_to_lorentz = true;
  options.preserve_psd_inner_product_vectorization = true;
  options.parse_psd_using_upper_triangular = false;
  options.parse_bounding_box_constraints_as_positive_orthant =
      constructor_options.parse_bounding_box_constraints_as_positive_orthant;

  // Remove the quadratic costs and convert them to a linear cost.
  if (!constructor_options.keep_quadratic_costs) {
    std::vector<Binding<QuadraticCost>> quadratic_cost_copy =
        prog->quadratic_costs();
    for (const auto& quadratic_cost : quadratic_cost_copy) {
      DRAKE_THROW_UNLESS(quadratic_cost.evaluator()->is_convex());
      int num_costs_removed = prog->RemoveCost(quadratic_cost);
      if (num_costs_removed == 0) {
        // If a cost is duplicated, it is possible that this loop tries to
        // remove it twice. We don't want to repeat adding a quadratic
        // constraint for this cost.
        continue;
      }
      // We convert the quadratic cost to a linear cost in the standard form.
      symbolic::Variable t = prog->NewContinuousVariables(1, "t")[0];
      prog->AddLinearCost(num_costs_removed * t);
      const VectorXDecisionVariable& x = quadratic_cost.variables();
      Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(x.size() + 1, x.size() + 1);
      Q.topLeftCorner(x.size(), x.size()) = quadratic_cost.evaluator()->Q();
      Eigen::VectorXd b(x.size() + 1);
      b.head(x.size()) = quadratic_cost.evaluator()->b();
      b(x.size()) = -1;
      VectorXDecisionVariable xt(x.size() + 1);
      xt.head(x.size()) = x;
      xt(x.size()) = t;
      prog->AddQuadraticAsRotatedLorentzConeConstraint(
          Q, b, quadratic_cost.evaluator()->c(), xt);
    }
  }
  // Remove the L2 norm costs and convert them to Lorentz cone constraints.
  std::vector<Binding<L2NormCost>> l2norm_cost_copy = prog->l2norm_costs();
  for (const auto& l2norm_cost : l2norm_cost_copy) {
    int num_costs_removed = prog->RemoveCost(l2norm_cost);
    if (num_costs_removed == 0) {
      // If a cost is duplicated, it is possible that this loop tries to remove
      // it twice. We don't want to repeat adding a Lorentz cone constraint for
      // this cost.
      continue;
    }
    prog->AddL2NormCostUsingConicConstraint(
        l2norm_cost.evaluator()->GetDenseA(), l2norm_cost.evaluator()->b(),
        l2norm_cost.variables());
  }
  x_ = prog->decision_variables();

  std::vector<Eigen::Triplet<double>> P_std_upper_triplets;
  std::vector<double> c_std(prog->num_vars(), 0.0);
  internal::ParseLinearCosts(*prog, &c_std, &d_);
  internal::ParseQuadraticCosts(*prog, &P_std_upper_triplets, &c_std, &d_);
  c_.resize(static_cast<int>(c_std.size()));
  for (int i = 0; i < ssize(c_std); ++i) {
    if (c_std[i] != 0.0) {
      c_.insert(i) = c_std[i];
    }
  }
  P_.resize(prog->num_vars(), prog->num_vars());
  P_.setFromTriplets(P_std_upper_triplets.begin(), P_std_upper_triplets.end());

  internal::DoAggregateConvexConstraints(*prog, options, &info);
  // We need to negate the A_triplets since they return as -Ax + b ∈ K.
  for (int i = 0; i < ssize(info.A_triplets); ++i) {
    info.A_triplets[i] = Eigen::Triplet<double>(info.A_triplets[i].row(),
                                                info.A_triplets[i].col(),
                                                -info.A_triplets[i].value());
  }
  A_.resize(info.A_row_count, prog->num_vars());
  A_.setFromTriplets(info.A_triplets.begin(), info.A_triplets.end());

  b_.resize(info.b_std.size());
  for (int i = 0; i < ssize(info.b_std); ++i) {
    if (info.b_std[i] != 0.0) {
      b_.insert(i) = info.b_std[i];
    }
  }

  int expected_A_row_count = 0;
  attributes_to_start_end_pairs_.emplace(
      ProgramAttribute::kLinearEqualityConstraint,
      std::vector<std::pair<int, int>>{});
  if (info.num_linear_equality_constraint_rows > 0) {
    attributes_to_start_end_pairs_
        .at(ProgramAttribute::kLinearEqualityConstraint)
        .emplace_back(0, info.num_linear_equality_constraint_rows);
  }
  expected_A_row_count += info.num_linear_equality_constraint_rows;

  const int total_num_linear_constraints =
      info.num_linear_constraint_rows +
      info.num_bounding_box_inequality_constraint_rows;

  attributes_to_start_end_pairs_.emplace(ProgramAttribute::kLinearConstraint,
                                         std::vector<std::pair<int, int>>{});
  if (total_num_linear_constraints > 0) {
    if (constructor_options
            .parse_bounding_box_constraints_as_positive_orthant) {
      attributes_to_start_end_pairs_.at(ProgramAttribute::kLinearConstraint)
          .emplace_back(expected_A_row_count,
                        expected_A_row_count + total_num_linear_constraints);
      expected_A_row_count += total_num_linear_constraints;

    } else {
      attributes_to_start_end_pairs_.at(ProgramAttribute::kLinearConstraint)
          .emplace_back(expected_A_row_count,
                        expected_A_row_count + info.num_linear_constraint_rows);
      expected_A_row_count += info.num_linear_constraint_rows;
      attributes_to_start_end_pairs_.at(ProgramAttribute::kLinearConstraint)
          .emplace_back(expected_A_row_count,
                        expected_A_row_count +
                            info.num_bounding_box_inequality_constraint_rows);
      expected_A_row_count += info.num_bounding_box_inequality_constraint_rows;

      bb_lower_bounds_ =
          Eigen::Map<Eigen::VectorXd>(info.bb_lb.data(), info.bb_lb.size());
      bb_upper_bounds_ =
          Eigen::Map<Eigen::VectorXd>(info.bb_ub.data(), info.bb_ub.size());
      DRAKE_DEMAND(bb_lower_bounds_.size() ==
                   info.num_bounding_box_inequality_constraint_rows);
      DRAKE_DEMAND(bb_upper_bounds_.size() ==
                   info.num_bounding_box_inequality_constraint_rows);
    }
  }

  attributes_to_start_end_pairs_.emplace(
      ProgramAttribute::kLorentzConeConstraint,
      std::vector<std::pair<int, int>>{});
  attributes_to_start_end_pairs_.at(ProgramAttribute::kLorentzConeConstraint)
      .reserve(info.second_order_cone_lengths.size());
  for (const int soc_length : info.second_order_cone_lengths) {
    attributes_to_start_end_pairs_.at(ProgramAttribute::kLorentzConeConstraint)
        .emplace_back(expected_A_row_count, expected_A_row_count + soc_length);
    expected_A_row_count += soc_length;
  }

  attributes_to_start_end_pairs_.emplace(
      ProgramAttribute::kPositiveSemidefiniteConstraint,
      std::vector<std::pair<int, int>>{});
  attributes_to_start_end_pairs_
      .at(ProgramAttribute::kPositiveSemidefiniteConstraint)
      .reserve(info.psd_row_size.size());
  for (const std::optional<int> row_size : info.psd_row_size) {
    DRAKE_THROW_UNLESS(row_size.has_value());
    int psd_length = *row_size * (*row_size + 1) / 2;
    attributes_to_start_end_pairs_
        .at(ProgramAttribute::kPositiveSemidefiniteConstraint)
        .emplace_back(expected_A_row_count, expected_A_row_count + psd_length);
    expected_A_row_count += psd_length;
  }
  DRAKE_DEMAND(expected_A_row_count == A_.rows());
}

std::unique_ptr<MathematicalProgram> ConicStandardForm::MakeProgram() const {
  std::unique_ptr<MathematicalProgram> prog_standard_form =
      std::make_unique<MathematicalProgram>();
  prog_standard_form->AddDecisionVariables(x_);
  prog_standard_form->AddLinearCost(c_.toDense(), d_, x_);
  // If quadratic costs were kept, encode them here. P_ stores the (upper
  // triangular) Hessian terms, while the associated linear and constant terms
  // are already in c_ and d_ above.
  if (P_.nonZeros() > 0) {
    prog_standard_form->AddQuadraticCost(
        Eigen::MatrixXd(P_), Eigen::VectorXd::Zero(x_.size()), 0.0, x_);
  }

  for (const auto& [attribute, index_pairs] : attributes_to_start_end_pairs_) {
    int index_pair_index = 0;
    for (const auto& [start, end] : index_pairs) {
      const int length = end - start;
      if (attribute == ProgramAttribute::kLinearEqualityConstraint) {
        prog_standard_form->AddLinearEqualityConstraint(
            A_.middleRows(start, length), -b_.segment(start, length).toDense(),
            x_);
      } else if (attribute == ProgramAttribute::kLinearConstraint) {
        if (index_pair_index == 0) {
          prog_standard_form->AddLinearConstraint(
              A_.middleRows(start, length),
              -b_.segment(start, length).toDense(),
              Eigen::VectorXd::Constant(length, kInf), x_);
        } else if (index_pair_index == 1) {
          DRAKE_DEMAND(length == bb_lower_bounds_.size());
          DRAKE_DEMAND(length == bb_upper_bounds_.size());
          // Bounding box rows in A_ already have the correct sign (see
          // ParseAllLinearConstraintsIntoZeroPositiveOrthantAndBoundingBox),
          // so we should not negate them again here.
          prog_standard_form->AddLinearConstraint(A_.middleRows(start, length),
                                                  bb_lower_bounds_,
                                                  bb_upper_bounds_, x_);
        } else {
          throw std::runtime_error(
              "ConicStandardForm::MakeProgram(): More than two linear "
              "constraint segments found when parsing bounding box "
              "constraints.");
        }
      } else if (attribute == ProgramAttribute::kLorentzConeConstraint) {
        prog_standard_form->AddLorentzConeConstraint(
            A_.middleRows(start, length).toDense(),
            b_.segment(start, length).toDense(), x_);
      } else if (attribute ==
                 ProgramAttribute::kPositiveSemidefiniteConstraint) {
        const MatrixX<symbolic::Expression> y_vec =
            A_.middleRows(start, length).toDense() *
                x_.cast<symbolic::Expression>() +
            b_.segment(start, length).toDense();
        MatrixX<symbolic::Expression> Y =
            math::ToSymmetricMatrixFromLowerTriangularColumns(y_vec);
        const double sqrt2 = std::sqrt(2);
        for (int i = 0; i < Y.rows(); ++i) {
          for (int j = i + 1; j < Y.cols(); ++j) {
            Y(i, j) = (Y(i, j) / sqrt2).Expand();
            Y(j, i) = Y(i, j);
          }
        }
        prog_standard_form->AddPositiveSemidefiniteConstraint(Y);
      }
      index_pair_index++;
    }
  }
  return prog_standard_form;
}
}  // namespace solvers
}  // namespace drake
