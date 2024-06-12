#include "drake/solvers/conic_standard_form.h"

#include <initializer_list>
#include <limits>
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
          ProgramAttribute::kLinearCost});
  if (!AreRequiredAttributesSupported(prog.required_capabilities(),
                                      supported_attributes,
                                      &unsupported_message)) {
    throw std::runtime_error(fmt::format(
        "ConicStandardForm() does not (yet) support this program: {}.",
        unsupported_message));
  }
}

}  // namespace

ConicStandardForm::ConicStandardForm(const MathematicalProgram& prog) {
  CheckSupported(prog);

  internal::ConvexConstraintAggregationInfo info;
  internal::ConvexConstraintAggregationOptions options;
  options.cast_rotated_lorentz_to_lorentz = true;
  options.preserve_psd_inner_product_vectorization = true;
  options.parse_psd_using_upper_triangular = false;

  std::vector<double> c_std(prog.num_vars(), 0.0);
  internal::ParseLinearCosts(prog, &c_std, &d_);
  c_.resize(c_std.size());
  for (int i = 0; i < ssize(c_std); ++i) {
    if (c_std[i] != 0.0) {
      c_.insert(i) = c_std[i];
    }
  }

  internal::DoAggregateConvexConstraints(prog, options, &info);
  // We need to negate the A_triplets since they return as -Ax + b ∈ K.
  for (int i = 0; i < ssize(info.A_triplets); ++i) {
    info.A_triplets[i] = Eigen::Triplet<double>(info.A_triplets[i].row(),
                                                info.A_triplets[i].col(),
                                                -info.A_triplets[i].value());
  }
  A_.resize(info.A_row_count, prog.num_vars());
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
    attributes_to_start_end_pairs_.at(ProgramAttribute::kLinearConstraint)
        .emplace_back(expected_A_row_count,
                      expected_A_row_count + total_num_linear_constraints);
  }
  expected_A_row_count += total_num_linear_constraints;

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
  for (const int row_size : info.psd_row_size) {
    int psd_length = row_size * (row_size + 1) / 2;
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
  auto x = prog_standard_form->NewContinuousVariables(A_.cols(), "x");
  prog_standard_form->AddLinearCost(c_.toDense(), d_, x);

  for (const auto& [attribute, index_pairs] : attributes_to_start_end_pairs_) {
    for (const auto& [start, end] : index_pairs) {
      const int length = end - start;
      if (attribute == ProgramAttribute::kLinearEqualityConstraint) {
        prog_standard_form->AddLinearEqualityConstraint(
            A_.middleRows(start, length), -b_.segment(start, length).toDense(),
            x);
      } else if (attribute == ProgramAttribute::kLinearConstraint) {
        prog_standard_form->AddLinearConstraint(
            A_.middleRows(start, length), -b_.segment(start, length).toDense(),
            Eigen::VectorXd::Constant(length, kInf), x);
      } else if (attribute == ProgramAttribute::kLorentzConeConstraint) {
        prog_standard_form->AddLorentzConeConstraint(
            A_.middleRows(start, length).toDense(),
            b_.segment(start, length).toDense(), x);
      } else if (attribute ==
                 ProgramAttribute::kPositiveSemidefiniteConstraint) {
        const MatrixX<symbolic::Expression> y_vec =
            A_.middleRows(start, length).toDense() *
                x.cast<symbolic::Expression>() +
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
    }
  }
  return prog_standard_form;
}
}  // namespace solvers
}  // namespace drake
