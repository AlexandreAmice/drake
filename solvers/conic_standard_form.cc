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
                      internal::ConvexAggregationInfo* aggregation_info) {
  aggregation_info->num_x = prog.num_vars();

  internal::ParseLinearCosts(prog, &aggregation_info->cost_info.c_std,
                             &aggregation_info->cost_info.d);
  if (options.use_quadratic_cost) {
    internal::ParseQuadraticCosts(
        prog, &aggregation_info->cost_info.P_upper_triplets,
        &aggregation_info->cost_info.c_std, &aggregation_info->cost_info.d);
  } else {
    internal::ParseQuadraticCostsWithRotatedLorentzCone(
        prog, &aggregation_info->cost_info.c_std,
        &aggregation_info->soc_constraint_info.A_triplets,
        &aggregation_info->soc_constraint_info.b_std,
        &aggregation_info->soc_constraint_info.A_row_count,
        &aggregation_info->soc_constraint_info.second_order_cone_lengths,
        &aggregation_info->num_x);
  }

  internal::ParseL2NormCosts(
      prog, &aggregation_info->num_x,
      &aggregation_info->soc_constraint_info.A_triplets,
      &aggregation_info->soc_constraint_info.b_std,
      &aggregation_info->soc_constraint_info.A_row_count,
      &aggregation_info->soc_constraint_info.second_order_cone_lengths,
      &aggregation_info->soc_constraint_info
           .l2norm_costs_lorentz_cone_y_start_indices,
      &aggregation_info->cost_info.c_std,
      &aggregation_info->soc_constraint_info.l2norm_costs_t_slack_indices);
}

void ParseConvexConstraints(const MathematicalProgram& prog,
                            const ConicStandardFormOptions& options,
                            internal::ConvexAggregationInfo* info) {
  info->psd_constraint_info.parse_psd_using_upper_triangular =
      options.parse_psd_using_upper_triangular;

  // Parse Linear Equality Constraints.
  int num_linear_equality_constraint_rows{0};
  internal::ParseLinearEqualityConstraints(
      prog, &info->equality_constraint_info.A_triplets,
      &info->equality_constraint_info.b_std,
      &info->equality_constraint_info.A_row_count,
      &info->linear_eq_dual_variable_start_indices,
      &num_linear_equality_constraint_rows);

  // Parse the Bounding Box constraints. The bounding box constraints which are
  // equalities are immediately added to A and b. The bounding box constraints
  // which are inequality constraints are stored in A_bb_ineq_triplets and
  // b_bb_ineq. They will be added to A and b after the linear constraints are
  // added.
  int num_bounding_box_inequality_constraint_rows{0};
  internal::ParseBoundingBoxConstraints(
      prog,
      // We can directly add the bounding box equality constraints
      // to A and b.
      &info->bounding_box_constraint_info.A_eq_triplets,
      &info->bounding_box_constraint_info.b_eq_std,
      &info->bounding_box_constraint_info.A_eq_row_count,
      &info->bounding_box_constraint_info.A_ineq_triplets,
      &info->bounding_box_constraint_info.b_ineq_std,
      &info->bounding_box_constraint_info.A_ineq_row_count,
      &info->bounding_box_constraint_info.bounding_box_constraint_dual_indices);

  // Parse Linear Constraints
  int num_linear_constraint_rows{0};
  internal::ParseLinearConstraints(
      prog, &info->linear_constraint_info.A_triplets,
      &info->linear_constraint_info.b_std,
      &info->linear_constraint_info.A_row_count,
      &info->linear_constraint_info.linear_constraint_dual_indices,
      &num_linear_constraint_rows);

  int scalar_psd_positive_cone_length{0};
  internal::ParseScalarPositiveSemidefiniteConstraints(
      prog, &info->linear_constraint_info.A_triplets,
      &info->linear_constraint_info.b_std,
      &info->linear_constraint_info.A_row_count,
      &scalar_psd_positive_cone_length,
      &info->linear_constraint_info.scalar_psd_dual_indices,
      &info->linear_constraint_info.scalar_lmi_dual_indices);

  // Parse Second-Order cone constraints
  internal::ParseSecondOrderConeConstraints(
      prog, &info->soc_constraint_info.A_triplets,
      &info->soc_constraint_info.b_std, &info->soc_constraint_info.A_row_count,
      &info->soc_constraint_info.second_order_cone_lengths,
      &info->soc_constraint_info.lorentz_cone_dual_variable_start_indices,
      &info->soc_constraint_info
           .rotated_lorentz_cone_dual_variable_start_indices);

  // Parse the 2x2 PSD constraint as second order cone constraints. Do this now
  // to be compatible with the strict ordering required by SCS.
  int num_twobytwo_psd_and_lmi_constraints{0};
  internal::Parse2x2PositiveSemidefiniteConstraints(
      prog, &info->soc_constraint_info.A_triplets,
      &info->soc_constraint_info.b_std, &info->soc_constraint_info.A_row_count,
      &num_twobytwo_psd_and_lmi_constraints,
      &info->soc_constraint_info.twobytwo_psd_dual_start_indices,
      &info->soc_constraint_info.twobytwo_lmi_dual_start_indices);
  for (int i = 0; i < num_twobytwo_psd_and_lmi_constraints; i++) {
    info->soc_constraint_info.second_order_cone_lengths.push_back(3);
  }

  // Parse PSD cone constraints
  internal::ParsePositiveSemidefiniteConstraints(
      prog, info->psd_constraint_info.parse_psd_using_upper_triangular,
      &info->psd_constraint_info.A_triplets, &info->psd_constraint_info.b_std,
      &info->psd_constraint_info.A_row_count,
      &info->psd_constraint_info.psd_cone_length,
      &info->psd_constraint_info.lmi_cone_length,
      &info->psd_constraint_info.psd_y_start_indices,
      &info->psd_constraint_info.lmi_y_start_indices);

  // Parse Exponential Cone Constraints
  internal::ParseExponentialConeConstraints(
      prog, &info->exponential_cone_info.A_triplets,
      &info->exponential_cone_info.b_std,
      &info->exponential_cone_info.A_row_count);
}

internal::ConicStandardFormInfo AggregateConicInformation(
    internal::ConvexAggregationInfo&& info) {
  internal::ConicStandardFormInfo returned_info;
  returned_info.A_triplets.reserve(
      info.equality_constraint_info.A_triplets.size() +
      info.bounding_box_constraint_info.A_eq_triplets.size() +
      info.bounding_box_constraint_info.A_ineq_triplets.size() +
      info.linear_constraint_info.A_triplets.size() +
      info.soc_constraint_info.A_triplets.size() +
      info.psd_constraint_info.A_triplets.size() +
      info.exponential_cone_info.A_triplets.size());

  // We cannot use move semantics to assemble all the triplets since the
  // Eigen::Triplets have const members.
  returned_info.A_row_count = 0;
  auto add_triplets = [&returned_info](const auto& triplets, int row_incr) {
    for (const auto& triplet : triplets) {
      returned_info.A_triplets.emplace_back(triplet.row() + row_incr,
                                            triplet.col(), triplet.value());
    }
  };

  add_triplets(info.equality_constraint_info.A_triplets,
               returned_info.A_row_count);
  returned_info.dual_info.linear_eq_dual_variable_start_indices = std::move(
      info.equality_constraint_info.linear_eq_dual_variable_start_indices);
  returned_info.A_row_count += info.equality_constraint_info.A_row_count;

  add_triplets(info.bounding_box_constraint_info.A_eq_triplets,
               returned_info.A_row_count);
  add_triplets(info.bounding_box_constraint_info.A_ineq_triplets,
               returned_info.A_row_count +
                   info.bounding_box_constraint_info.A_eq_row_count);
  returned_info.dual_info.bounding_box_constraint_dual_indices.reserve(
      info.bounding_box_constraint_info.bounding_box_constraint_dual_indices
          .size());
  std::transform(
      std::make_move_iterator(
          info.bounding_box_constraint_info.bounding_box_constraint_dual_indices
              .begin()),
      std::make_move_iterator(info.bounding_box_constraint_info
                                  .bounding_box_constraint_dual_indices.end()),
      std::back_inserter(
          returned_info.dual_info.bounding_box_constraint_dual_indices),
      [&returned_info, &info](auto&& bb_indices) {
        for (auto& [lb, ub] : bb_indices) {
          if (lb == ub && lb != -1) {
            lb += returned_info.A_row_count;
            ub += returned_info.A_row_count;
          } else {
            if (lb != -1) {
              lb += returned_info.A_row_count +
                    info.bounding_box_constraint_info.A_eq_row_count;
            }
            if (ub != -1) {
              ub += returned_info.A_row_count +
                    info.bounding_box_constraint_info.A_eq_row_count;
            }
          }
        }
        return std::move(bb_indices);
      });
  returned_info.A_row_count += info.bounding_box_constraint_info.A_eq_row_count;

  add_triplets(info.linear_constraint_info.A_triplets,
               returned_info.A_row_count);
  std::transform(
      std::make_move_iterator(
          info.linear_constraint_info.linear_constraint_dual_indices.begin()),
      std::make_move_iterator(
          info.linear_constraint_info.linear_constraint_dual_indices.end()),
      std::back_inserter(
          returned_info.dual_info.linear_constraint_dual_indices),
      [&returned_info, &info](auto&& indices) {
        for (auto& [lb, ub] : indices) {
          if (lb != -1) {
            lb += returned_info.A_row_count +
                  info.bounding_box_constraint_info.A_eq_row_count;
          }
          if (ub != -1) {
            ub += returned_info.A_row_count +
                  info.bounding_box_constraint_info.A_eq_row_count;
          }
        }
        return std::move(indices);
      });
  returned_info.A_row_count += info.linear_constraint_info.A_row_count;

  add_triplets(info.soc_constraint_info.A_triplets, returned_info.A_row_count);
  returned_info.A_row_count += info.soc_constraint_info.A_row_count;
  auto incremenent_dual_indices =
      [&returned_info, &info](std::vector<int>&& source_indices) {
        for (auto& ind : source_indices) {
          ind += returned_info.A_row_count;
        }
        return std::move(source_indices);
      };
  std::transform(
      std::make_move_iterator(
          info.soc_constraint_info.lorentz_cone_dual_variable_start_indices.begin()),
      std::make_move_iterator(
          info.soc_constraint_info.lorentz_cone_dual_variable_start_indices.end()),
      std::back_inserter(
          returned_info.dual_info.lorentz_cone_dual_variable_start_indices),
          incremenent_dual_indices
      );


//  add_triplets(info.psd_constraint_info.A_triplets);
//  returned_info.A_row_count += info.psd_constraint_info.A_row_count;
//  add_triplets(info.exponential_cone_info.A_triplets);
//  returned_info.A_row_count += info.exponential_cone_info.A_row_count;

  returned_info.b_std.reserve(
      info.equality_constraint_info.b_std.size() +
      info.bounding_box_constraint_info.b_eq_std.size() +
      info.bounding_box_constraint_info.b_ineq_std.size() +
      info.linear_constraint_info.b_std.size() +
      info.soc_constraint_info.b_std.size() +
      info.psd_constraint_info.b_std.size() +
      info.exponential_cone_info.b_std.size());
  returned_info.b_std.insert(
      returned_info.b_std.end(),
      std::make_move_iterator(info.equality_constraint_info.b_std.begin()),
      std::make_move_iterator(info.equality_constraint_info.b_std.end()));
  returned_info.b_std.insert(
      returned_info.b_std.end(),
      std::make_move_iterator(
          info.bounding_box_constraint_info.b_eq_std.begin()),
      std::make_move_iterator(
          info.bounding_box_constraint_info.b_eq_std.end()));
  returned_info.b_std.insert(
      returned_info.b_std.end(),
      std::make_move_iterator(
          info.bounding_box_constraint_info.b_ineq_std.begin()),
      std::make_move_iterator(
          info.bounding_box_constraint_info.b_ineq_std.end()));
  returned_info.b_std.insert(
      returned_info.b_std.end(),
      std::make_move_iterator(info.linear_constraint_info.b_std.begin()),
      std::make_move_iterator(info.linear_constraint_info.b_std.end()));
  returned_info.b_std.insert(
      returned_info.b_std.end(),
      std::make_move_iterator(info.soc_constraint_info.b_std.begin()),
      std::make_move_iterator(info.soc_constraint_info.b_std.end()));
  returned_info.b_std.insert(
      returned_info.b_std.end(),
      std::make_move_iterator(info.psd_constraint_info.b_std.begin()),
      std::make_move_iterator(info.psd_constraint_info.b_std.end()));
  returned_info.b_std.insert(
      returned_info.b_std.end(),
      std::make_move_iterator(info.exponential_cone_info.b_std.begin()),
      std::make_move_iterator(info.exponential_cone_info.b_std.end()));

  return returned_info;
  returned_info
};

}  // namespace

namespace internal {
void ParseConicStandardForm(const MathematicalProgram& prog,
                            const ConicStandardFormOptions& options) {
  ConvexAggregationInfo info{};
  ParseConvexCosts(prog, options, &info);
  ParseConvexConstraints(prog, options, &info);
  // Now we aggregate the intermediate information into the conic standard form
  // struct.

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
