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
          ProgramAttribute::kLinearCost});
  if (!AreRequiredAttributesSupported(prog.required_capabilities(),
                                      supported_attributes,
                                      &unsupported_message)) {
    throw std::runtime_error(fmt::format(
        "ConicStandardForm() does not (yet) support this program: {}.",
        unsupported_message));
  }
}

void ParseConvexCosts(const MathematicalProgram& prog,
                      const ConicStandardFormOptions& options,
                      internal::CostAggregationInfo* cost_info,
                      internal::DualInfo* dual_info) {
  internal::ConstraintAggregationInfo* constraint_info =
      &cost_info->constraint_info;
  cost_info->num_x = prog.num_vars();

  internal::ParseLinearCosts(prog, &cost_info->c_std, &cost_info->d);
  if (options.use_quadratic_cost) {
    internal::ParseQuadraticCosts(prog, &cost_info->P_upper_triplets,
                                  &cost_info->c_std, &cost_info->d);
  } else {
    int row_start = constraint_info->A_row_count;
    internal::ParseQuadraticCostsWithRotatedLorentzCone(
        prog, &cost_info->c_std, &constraint_info->A_triplets,
        &constraint_info->b_std, &constraint_info->A_row_count,
        &constraint_info->second_order_cone_lengths, &cost_info->num_x);
    constraint_info->attributes_to_start_end_pairs
        .at(ProgramAttribute::kLorentzConeConstraint)
        .reserve(constraint_info->second_order_cone_lengths.size());
    for (const auto& length : constraint_info->second_order_cone_lengths) {
      constraint_info->attributes_to_start_end_pairs
          .at(ProgramAttribute::kLorentzConeConstraint)
          .emplace_back(row_start, row_start + length);
      row_start += length;
    }
  }

  int row_start = constraint_info->A_row_count;
  internal::ParseL2NormCosts(
      prog, &cost_info->num_x, &constraint_info->A_triplets,
      &constraint_info->b_std, &constraint_info->A_row_count,
      &constraint_info->second_order_cone_lengths,
      &dual_info->l2norm_costs_lorentz_cone_y_start_indices, &cost_info->c_std,
      &dual_info->l2norm_costs_t_slack_indices);
  constraint_info->attributes_to_start_end_pairs
      .at(ProgramAttribute::kLorentzConeConstraint)
      .reserve(constraint_info->second_order_cone_lengths.size());
  for (const auto& length : constraint_info->second_order_cone_lengths) {
    constraint_info->attributes_to_start_end_pairs
        .at(ProgramAttribute::kLorentzConeConstraint)
        .emplace_back(row_start, row_start + length);
    row_start += length;
  }
}

void ParseConvexConstraints(const MathematicalProgram& prog,
                            const ConicStandardFormOptions& options,
                            internal::ConstraintAggregationInfo* info,
                            internal::DualInfo* dual_info) {
  info->parse_psd_using_upper_triangular =
      options.parse_psd_using_upper_triangular;
  int total_cone_row_count = 0;

  // Parse Linear Equality Constraints.
  internal::ParseLinearEqualityConstraints(
      prog, &(info->A_triplets), &(info->b_std), &(info->A_row_count),
      &(dual_info->linear_eq_dual_variable_start_indices),
      &(info->num_linear_equality_constraint_rows));
  // Parse the Bounding Box constraints. The bounding box constraints which are
  // equalities are immediately added to A and b. The bounding box constraints
  // which are inequality constraints are stored in A_bb_ineq_triplets and
  // b_bb_ineq. They will be added to A and b after the linear constraints are
  // added. The reason for this is we wish A to be compatible with the strict
  // ordering of the cones given by SCS:
  // https://www.cvxgrp.org/scs/api/cones.html.
  std::vector<Eigen::Triplet<double>> A_bb_ineq_triplets;
  std::vector<double> b_bb_ineq;
  const int A_row_count_before_parsing_bb{info->A_row_count};
  internal::ParseBoundingBoxConstraints(
      prog,
      // We can directly add the bounding box equality constraints
      // to A and b.
      &(info->A_triplets), &(info->b_std), &(info->A_row_count),
      // We delay adding the bounding box inequality constraints to A and b due
      // to the strict ordering required by SCS.
      &A_bb_ineq_triplets, &b_bb_ineq,
      &(info->num_bounding_box_inequality_constraint_rows),
      &(dual_info->bounding_box_constraint_dual_indices));
  const int num_bb_equality_constraint =
      info->A_row_count - A_row_count_before_parsing_bb;
  info->num_linear_equality_constraint_rows += num_bb_equality_constraint;

  info->attributes_to_start_end_pairs
      .at(ProgramAttribute::kLinearEqualityConstraint)
      .emplace_back(
          total_cone_row_count,
          total_cone_row_count + info->num_linear_equality_constraint_rows);
  total_cone_row_count += info->num_linear_equality_constraint_rows;

  // Parse Linear Constraints
  internal::ParseLinearConstraints(prog, &(info->A_triplets), &(info->b_std),
                                   &(info->A_row_count),
                                   &(dual_info->linear_constraint_dual_indices),
                                   &(info->num_linear_constraint_rows));
  // Now we can add the bounding box constraints.
  for (int i = 0; i < info->num_bounding_box_inequality_constraint_rows; ++i) {
    info->A_triplets.emplace_back(
        A_bb_ineq_triplets[i].row() + info->A_row_count,
        A_bb_ineq_triplets[i].col(), A_bb_ineq_triplets[i].value());
    info->b_std.push_back(b_bb_ineq[i]);
  }
  // We need to increment the dual variable index of only the inequality
  // bounding box constraints.
  for (int i = 0; i < ssize(prog.bounding_box_constraints()); ++i) {
    for (int j = 0; j < prog.bounding_box_constraints()[i].variables().rows();
         ++j) {
      if (prog.bounding_box_constraints()[i].evaluator()->lower_bound()[j] !=
          prog.bounding_box_constraints()[i].evaluator()->upper_bound()[j]) {
        if (dual_info->bounding_box_constraint_dual_indices[i][j].first != -1) {
          dual_info->bounding_box_constraint_dual_indices[i][j].first +=
              info->A_row_count;
        }
        if (dual_info->bounding_box_constraint_dual_indices[i][j].second !=
            -1) {
          dual_info->bounding_box_constraint_dual_indices[i][j].second +=
              info->A_row_count;
        }
      }
    }
  }
  info->A_row_count += info->num_bounding_box_inequality_constraint_rows;

  // Parse the scalar PSD constraint as linear constraints. Do this now to be
  // compatible with the strict ordering required by SCS.
  internal::ParseScalarPositiveSemidefiniteConstraints(
      prog, &info->A_triplets, &info->b_std, &info->A_row_count,
      &info->scalar_psd_positive_cone_length,
      &dual_info->scalar_psd_dual_indices,
      &(dual_info->scalar_lmi_dual_indices));

  const int total_num_linear_inequality_constraint_rows =
      info->num_linear_constraint_rows +
      info->num_bounding_box_inequality_constraint_rows +
      info->scalar_psd_positive_cone_length;
  info->attributes_to_start_end_pairs.at(ProgramAttribute::kLinearConstraint)
      .emplace_back(
          total_cone_row_count,
          total_cone_row_count + total_num_linear_inequality_constraint_rows);
  total_cone_row_count += total_num_linear_inequality_constraint_rows;

  // Parse Second-Order cone constraints
  internal::ParseSecondOrderConeConstraints(
      prog, &info->A_triplets, &info->b_std, &info->A_row_count,
      &info->second_order_cone_lengths,
      &dual_info->lorentz_cone_dual_variable_start_indices,
      &dual_info->rotated_lorentz_cone_dual_variable_start_indices);

  // Parse the 2x2 PSD constraint as second order cone constraints. Do this now
  // to be compatible with the strict ordering required by SCS.
  internal::Parse2x2PositiveSemidefiniteConstraints(
      prog, &info->A_triplets, &info->b_std, &info->A_row_count,
      &info->num_twobytwo_psd_and_lmi_constraints,
      &dual_info->twobytwo_psd_dual_start_indices,
      &dual_info->twobytwo_lmi_dual_start_indices);
  for (const int soc_length : info->second_order_cone_lengths) {
    info->attributes_to_start_end_pairs
        .at(ProgramAttribute::kLorentzConeConstraint)
        .emplace_back(total_cone_row_count, total_cone_row_count + soc_length);
    total_cone_row_count += soc_length;
  }
  for (int i = 0; i < info->num_twobytwo_psd_and_lmi_constraints; ++i) {
    // Each two by two matrix corresponds to a size 3 second order cone
    // constraint.
    info->attributes_to_start_end_pairs
        .at(ProgramAttribute::kLorentzConeConstraint)
        .emplace_back(total_cone_row_count, total_cone_row_count + 3);
    total_cone_row_count += 3;
  }

  // Parse PSD cone constraints
  internal::ParsePositiveSemidefiniteConstraints(
      prog, options.parse_psd_using_upper_triangular, &info->A_triplets,
      &info->b_std, &info->A_row_count, &info->psd_cone_length,
      &info->lmi_cone_length, &dual_info->psd_y_start_indices,
      &dual_info->lmi_y_start_indices);
  auto matrix_rows_to_length = [&](const std::optional<int> length) {
    return (*length * (*length + 1)) / 2;
  };
  for (const std::optional<int>& matrix_rows : info->psd_cone_length) {
    if (matrix_rows.has_value()) {
      const int length = matrix_rows_to_length(matrix_rows);
      info->attributes_to_start_end_pairs
          .at(ProgramAttribute::kPositiveSemidefiniteConstraint)
          .emplace_back(total_cone_row_count, total_cone_row_count + length);
      total_cone_row_count += +length;
    }
  }
  for (const std::optional<int>& matrix_rows : info->lmi_cone_length) {
    if (matrix_rows.has_value()) {
      const int length = matrix_rows_to_length(matrix_rows);
      info->attributes_to_start_end_pairs
          .at(ProgramAttribute::kPositiveSemidefiniteConstraint)
          .emplace_back(total_cone_row_count, total_cone_row_count + length);
      total_cone_row_count += +length;
    }
  }

  // Parse Exponential Cone Constraints
  internal::ParseExponentialConeConstraints(
      prog, &(info->A_triplets), &(info->b_std), &(info->A_row_count));
  info->num_exponential_cone_constraints =
      ssize(prog.exponential_cone_constraints());
}
}  // namespace

namespace internal {
std::unique_ptr<ConicStandardFormParsingInfo> ParseConicStandardForm(
    const MathematicalProgram& prog, const ConicStandardFormOptions& options) {
  auto info = std::make_unique<ConicStandardFormParsingInfo>();
  ParseConvexCosts(prog, options, &info->cost_info, &info->dual_info);
  ParseConvexConstraints(prog, options, &info->constraint_info,
                         &info->dual_info);
  // We need to combine the conic standard forms of the constraints generated
  // with the costs and the original constraints.


  return info;
}
}  // namespace internal

ConicStandardForm::ConicStandardForm(const MathematicalProgram& prog,
                                     const ConicStandardFormOptions& options)
    : x_{prog.decision_variables()} {
  CheckSupported(prog);
  internal::CostAggregationInfo cost_info;
  internal::ConstraintAggregationInfo constraint_info;
  internal::DualInfo dual_info;

  ParseConvexCosts(prog, options, &cost_info, &dual_info);
  ParseConvexConstraints(prog, options, &constraint_info, &dual_info);
  //
  //  if (options.sort_cones) {
  //  }
  //
  //  internal::ConvexConstraintAggregationInfo info;
  //  internal::ConvexConstraintAggregationOptions options;
  //  options.cast_rotated_lorentz_to_lorentz = true;
  //  options.preserve_psd_inner_product_vectorization = true;
  //  options.parse_psd_using_upper_triangular = false;
  //
  //  std::vector<double> c_std(prog.num_vars(), 0.0);
  //  internal::ParseLinearCosts(prog, &c_std, &d_);
  //  c_.resize(c_std.size());
  //  for (int i = 0; i < ssize(c_std); ++i) {
  //    if (c_std[i] != 0.0) {
  //      c_.insert(i) = c_std[i];
  //    }
  //  }
  //
  //  internal::DoAggregateConvexConstraints(prog, options, &info);
  //  // We need to negate the A_triplets since they return as -Ax + b ∈ K.
  //  for (int i = 0; i < ssize(info.A_triplets); ++i) {
  //    info.A_triplets[i] = Eigen::Triplet<double>(info.A_triplets[i].row(),
  //                                                info.A_triplets[i].col(),
  //                                                -info.A_triplets[i].value());
  //  }
  //  A_.resize(info.A_row_count, prog.num_vars());
  //  A_.setFromTriplets(info.A_triplets.begin(), info.A_triplets.end());
  //
  //  b_.resize(info.b_std.size());
  //  for (int i = 0; i < ssize(info.b_std); ++i) {
  //    if (info.b_std[i] != 0.0) {
  //      b_.insert(i) = info.b_std[i];
  //    }
  //  }
  //
  //  int expected_A_row_count = 0;
  //  attributes_to_start_end_pairs_.emplace(
  //      ProgramAttribute::kLinearEqualityConstraint,
  //      std::vector<std::pair<int, int>>{});
  //  if (info.num_linear_equality_constraint_rows > 0) {
  //    attributes_to_start_end_pairs_
  //        .at(ProgramAttribute::kLinearEqualityConstraint)
  //        .emplace_back(0, info.num_linear_equality_constraint_rows);
  //  }
  //  expected_A_row_count += info.num_linear_equality_constraint_rows;
  //
  //  const int total_num_linear_constraints =
  //      info.num_linear_constraint_rows +
  //      info.num_bounding_box_inequality_constraint_rows;
  //
  //  attributes_to_start_end_pairs_.emplace(ProgramAttribute::kLinearConstraint,
  //                                         std::vector<std::pair<int,
  //                                         int>>{});
  //  if (total_num_linear_constraints > 0) {
  //    attributes_to_start_end_pairs_.at(ProgramAttribute::kLinearConstraint)
  //        .emplace_back(expected_A_row_count,
  //                      expected_A_row_count + total_num_linear_constraints);
  //  }
  //  expected_A_row_count += total_num_linear_constraints;
  //
  //  attributes_to_start_end_pairs_.emplace(
  //      ProgramAttribute::kLorentzConeConstraint,
  //      std::vector<std::pair<int, int>>{});
  //  attributes_to_start_end_pairs_.at(ProgramAttribute::kLorentzConeConstraint)
  //      .reserve(info.second_order_cone_lengths.size());
  //  for (const int soc_length : info.second_order_cone_lengths) {
  //    attributes_to_start_end_pairs_.at(ProgramAttribute::kLorentzConeConstraint)
  //        .emplace_back(expected_A_row_count, expected_A_row_count +
  //        soc_length);
  //    expected_A_row_count += soc_length;
  //  }
  //
  //  attributes_to_start_end_pairs_.emplace(
  //      ProgramAttribute::kPositiveSemidefiniteConstraint,
  //      std::vector<std::pair<int, int>>{});
  //  attributes_to_start_end_pairs_
  //      .at(ProgramAttribute::kPositiveSemidefiniteConstraint)
  //      .reserve(info.psd_row_size.size());
  //  for (const int row_size : info.psd_row_size) {
  //    int psd_length = row_size * (row_size + 1) / 2;
  //    attributes_to_start_end_pairs_
  //        .at(ProgramAttribute::kPositiveSemidefiniteConstraint)
  //        .emplace_back(expected_A_row_count, expected_A_row_count +
  //        psd_length);
  //    expected_A_row_count += psd_length;
  //  }
  //  DRAKE_DEMAND(expected_A_row_count == A_.rows());
}

std::unique_ptr<MathematicalProgram> ConicStandardForm::MakeProgram() const {
  std::unique_ptr<MathematicalProgram> prog_standard_form =
      std::make_unique<MathematicalProgram>();
  prog_standard_form->AddDecisionVariables(x_);
  prog_standard_form->AddLinearCost(c_.toDense(), d_, x_);

  for (const auto& [attribute, index_pairs] : attributes_to_start_end_pairs_) {
    for (const auto& [start, end] : index_pairs) {
      const int length = end - start;
      if (attribute == ProgramAttribute::kLinearEqualityConstraint) {
        prog_standard_form->AddLinearEqualityConstraint(
            A_.middleRows(start, length), -b_.segment(start, length).toDense(),
            x_);
      } else if (attribute == ProgramAttribute::kLinearConstraint) {
        prog_standard_form->AddLinearConstraint(
            A_.middleRows(start, length), -b_.segment(start, length).toDense(),
            Eigen::VectorXd::Constant(length, kInf), x_);
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
    }
  }
  return prog_standard_form;
}
}  // namespace solvers
}  // namespace drake
