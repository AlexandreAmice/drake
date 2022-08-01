#include "drake/multibody/rational_forward_kinematics/cspace_free_line.h"

#include <gtest/gtest.h>

#include "drake/multibody/parsing/parser.h"
#include "drake/solvers/get_program_type.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"


namespace drake {
namespace multibody {

// Helper method for testing CspaceFreeLine.
// CspaceFreeLine DoublePendulumCspaceFreeLine() {
// A simple double pendulum with link lengths `l1` and `l2` with a box at the
// tip with diagonal `2r` between two (fixed) walls at `w` from the origin.
// The true configuration space is - w  ≤ l₁s₁ + l₂s₁₊₂ ± r * sin(1+2±π/4) ≤ w
// . These regions are visualized as the white region at
// https://www.desmos.com/calculator/2ugpepabig.

class DoublePendulumTest : public ::testing::Test {
 public:
  DoublePendulumTest() {
    const double l1 = 2.0;
    const double l2 = 1.0;
    const double r = .6;
    const double w = 1.83;
    const std::string double_pendulum_urdf = fmt::format(
        R"(
    <robot name="double_pendulum">
      <link name="fixed">
        <collision name="right">
          <origin rpy="0 0 0" xyz="{w_plus_one_half} 0 0"/>
          <geometry><box size="1 1 10"/></geometry>
        </collision>
        <collision name="left">
          <origin rpy="0 0 0" xyz="-{w_plus_one_half} 0 0"/>
          <geometry><box size="1 1 10"/></geometry>
        </collision>
      </link>
      <joint name="fixed_link_weld" type="fixed">
        <parent link="world"/>
        <child link="fixed"/>
      </joint>
      <link name="link1"/>
      <joint name="joint1" type="revolute">
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57"/>
        <parent link="world"/>
        <child link="link1"/>
      </joint>
      <link name="link2">
        <collision name="box">
          <origin rpy="0 0 0" xyz="0 0 -{l2}"/>
          <geometry><box size="{r} {r} {r}"/></geometry>
        </collision>
      </link>
      <joint name="joint2" type="revolute">
        <origin rpy="0 0 0" xyz="0 0 -{l1}"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57"/>
        <parent link="link1"/>
        <child link="link2"/>
      </joint>
    </robot>
    )",
        fmt::arg("w_plus_one_half", w + .5), fmt::arg("l1", l1),
        fmt::arg("l2", l2), fmt::arg("r", r / sqrt(2)));

    systems::DiagramBuilder<double> builder;
    std::tie(plant_, scene_graph_) =
        multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);
    multibody::Parser(plant_).AddModelFromString(double_pendulum_urdf, "urdf");
    plant_->Finalize();
    diagram_ = builder.Build();
  }

 protected:
  std::unique_ptr<systems::Diagram<double>> diagram_;
  MultibodyPlant<double>* plant_;
  geometry::SceneGraph<double>* scene_graph_;
  const Eigen::Vector2d q_star_0_{0.0, 0.0};
  const Eigen::Vector2d q_star_1_{0.0, 1.0};

  std::vector<CspaceFreeRegion::CspacePolytopeTuple> alternation_tuples_;
  VectorX<symbolic::Variable> d_var_, lagrangian_gram_vars_,
      verified_gram_vars_, separating_plane_vars_;
  std::vector<std::vector<int>> separating_plane_to_tuples_;
};

TEST_F(DoublePendulumTest, TestCspaceFreeLineConstructor) {
  CspaceFreeLine dut(*diagram_, plant_, scene_graph_,
                     SeparatingPlaneOrder::kAffine, std::nullopt);
  EXPECT_TRUE(dut.get_s0().size() == 2);
  EXPECT_TRUE(dut.get_s1().size() == 2);
}

// Evaluates whether two polynomials with indeterminates that have the same name
// evaluate to about the same expression. We assume that the coefficients in the
// polynomial are not Expressions, but scalars.
bool AffinePolynomialCoefficientsAlmostEqualByName(symbolic::Polynomial p1,
                                                   symbolic::Polynomial p2,
                                                   double tol) {
  auto extract_var_name_to_coefficient_map =
      [](symbolic::Polynomial p,
         std::unordered_map<std::string, double>* var_name_to_map_ptr) {
        double c_val;
        for (const auto& [m, c] : p.monomial_to_coefficient_map()) {
          for (const auto& v : m.GetVariables()) {
            c_val = c.Evaluate();
            if (std::abs(c_val) > 1E-12) {
              (*var_name_to_map_ptr).insert({v.get_name(), c.Evaluate()});
            }
          }
        }
      };
  std::unordered_map<std::string, double> p1_var_name_to_coefficient_map;
  std::unordered_map<std::string, double> p2_var_name_to_coefficient_map;
  extract_var_name_to_coefficient_map(p1, &p1_var_name_to_coefficient_map);
  extract_var_name_to_coefficient_map(p2, &p2_var_name_to_coefficient_map);

  if (p1_var_name_to_coefficient_map.size() ==
      p2_var_name_to_coefficient_map.size()) {
    for (const auto& [name, c] : p1_var_name_to_coefficient_map) {
      if (p2_var_name_to_coefficient_map.find(name) ==
              p2_var_name_to_coefficient_map.end() ||
          std::abs(p2_var_name_to_coefficient_map[name] - c) > tol) {
        return false;
      }
    }
  }
  return true;
}

TEST_F(DoublePendulumTest, TestGenerateLinkOnOneSideOfPlaneRationals) {
  const CspaceFreeLine dut(*diagram_, plant_, scene_graph_,
                           SeparatingPlaneOrder::kAffine, std::nullopt);
  const CspaceFreeRegion dut2(*diagram_, plant_, scene_graph_,
                              SeparatingPlaneOrder::kAffine,
                              CspaceRegionType::kAxisAlignedBoundingBox);
  std::vector<LinkVertexOnPlaneSideRational> rationals_free_line_0 =
      dut.GenerateLinkOnOneSideOfPlaneRationals(q_star_0_, {});
  std::vector<LinkVertexOnPlaneSideRational> rationals_free_region_0 =
      dut2.GenerateLinkOnOneSideOfPlaneRationals(q_star_0_, {});

  std::vector<LinkVertexOnPlaneSideRational> rationals_free_line_1 =
      dut.GenerateLinkOnOneSideOfPlaneRationals(q_star_1_, {});
  std::vector<LinkVertexOnPlaneSideRational> rationals_free_region_1 =
      dut2.GenerateLinkOnOneSideOfPlaneRationals(q_star_1_, {});

  auto test_same_rationals =
      [&dut, &dut2](
          std::vector<LinkVertexOnPlaneSideRational>& rationals_free_line,
          std::vector<LinkVertexOnPlaneSideRational>& rationals_free_region) {
        EXPECT_EQ(rationals_free_region.size(), rationals_free_line.size());

        symbolic::Environment env_line = {{dut.get_s0()[0], 0.0},
                                          {dut.get_s0()[1], -0.5},
                                          {dut.get_s1()[0], -0.25},
                                          {dut.get_s1()[1], 0.5},
                                          {dut.get_mu(), 0.0}};
        Eigen::VectorXd mu_vals = Eigen::VectorXd::LinSpaced(10, 0, 1);

        // first place we evaluate the rationals is at s0
        symbolic::Environment env_region = {
            {dut2.rational_forward_kinematics().t()[0], 0.0},
            {dut2.rational_forward_kinematics().t()[0], -0.5}};

        for (unsigned int i = 0; i < rationals_free_line.size(); ++i) {
          EXPECT_EQ(rationals_free_line.at(i).link_polytope->body_index(),
                    rationals_free_region.at(i).link_polytope->body_index());
          EXPECT_EQ(rationals_free_line.at(i).expressed_body_index,
                    rationals_free_region.at(i).expressed_body_index);
          EXPECT_EQ(
              rationals_free_line.at(i).other_side_link_polytope->body_index(),
              rationals_free_region.at(i)
                  .other_side_link_polytope->body_index());

          for (unsigned int j = 0; j < rationals_free_line.at(i).a_A.size();
               ++j) {
            EXPECT_TRUE(AffinePolynomialCoefficientsAlmostEqualByName(
                symbolic::Polynomial(rationals_free_line.at(i).a_A(j)),
                symbolic::Polynomial(rationals_free_region.at(i).a_A(j)),
                1E-12));
          }
          EXPECT_TRUE(AffinePolynomialCoefficientsAlmostEqualByName(
              symbolic::Polynomial(rationals_free_line.at(i).b),
              symbolic::Polynomial(rationals_free_region.at(i).b), 1E-12));

          EXPECT_EQ(rationals_free_line.at(i).plane_side,
                    rationals_free_region.at(i).plane_side);

          EXPECT_EQ(rationals_free_line.at(i).plane_order,
                    rationals_free_region.at(i).plane_order);

          for (unsigned int j = 0; j < mu_vals.size(); j++) {
            double mu = mu_vals[j];
            env_line[dut.get_mu()] = mu;
            env_region[dut2.rational_forward_kinematics().t()[0]] =
                mu * env_line[dut.get_s0()[0]] +
                (1 - mu) * env_line[dut.get_s1()[0]];
            env_region[dut2.rational_forward_kinematics().t()[1]] =
                mu * env_line[dut.get_s0()[1]] +
                (1 - mu) * env_line[dut.get_s1()[1]];

            symbolic::Polynomial free_line_numerator =
                symbolic::Polynomial(rationals_free_line.at(i)
                                         .rational.numerator()
                                         .EvaluatePartial(env_line)
                                         .ToExpression()
                                         .Expand());
            symbolic::Polynomial free_line_denominator =
                symbolic::Polynomial(rationals_free_line.at(i)
                                         .rational.denominator()
                                         .EvaluatePartial(env_line)
                                         .ToExpression()
                                         .Expand());
            symbolic::Polynomial free_region_numerator =
                symbolic::Polynomial(rationals_free_region.at(i)
                                         .rational.numerator()
                                         .EvaluatePartial(env_region)
                                         .ToExpression()
                                         .Expand());
            symbolic::Polynomial free_region_denominator =
                symbolic::Polynomial(rationals_free_region.at(i)
                                         .rational.denominator()
                                         .EvaluatePartial(env_region)
                                         .ToExpression()
                                         .Expand());

            EXPECT_TRUE(AffinePolynomialCoefficientsAlmostEqualByName(
                free_region_numerator, free_line_numerator, 1E-12));
            EXPECT_TRUE(AffinePolynomialCoefficientsAlmostEqualByName(
                free_region_denominator, free_line_denominator, 1E-12));
          }
        }
      };
  test_same_rationals(rationals_free_line_0, rationals_free_region_0);
  test_same_rationals(rationals_free_line_1, rationals_free_region_1);
}

TEST_F(DoublePendulumTest, TestGenerateTuplesForCertification) {
  const CspaceFreeLine dut(*diagram_, plant_, scene_graph_,
                           SeparatingPlaneOrder::kAffine, std::nullopt);

  dut.GenerateTuplesForCertification(
      q_star_0_, {}, &alternation_tuples_, &lagrangian_gram_vars_,
      &verified_gram_vars_, &separating_plane_vars_,
      &separating_plane_to_tuples_);
  int rational_count = 0;
  for (const auto& separating_plane : dut.separating_planes()) {
    rational_count += separating_plane.positive_side_polytope->p_BV().cols() +
                      separating_plane.negative_side_polytope->p_BV().cols();
  }
  EXPECT_EQ(alternation_tuples_.size(), rational_count);
  // Now count the total number of lagrangian gram vars.
  int lagrangian_gram_vars_count = 0;
  int verified_gram_vars_count = 0;
  std::unordered_set<int> lagrangian_gram_vars_start;
  std::unordered_set<int> verified_gram_vars_start;
  for (const auto& tuple : alternation_tuples_) {
    const int gram_rows = tuple.monomial_basis.rows();
    const int gram_lower_size = gram_rows * (gram_rows + 1) / 2;
    lagrangian_gram_vars_count += gram_lower_size * 2;
    verified_gram_vars_count += gram_lower_size;
    std::copy(tuple.polytope_lagrangian_gram_lower_start.begin(),
              tuple.polytope_lagrangian_gram_lower_start.end(),
              std::inserter(lagrangian_gram_vars_start,
                            lagrangian_gram_vars_start.end()));
    std::copy(tuple.t_lower_lagrangian_gram_lower_start.begin(),
              tuple.t_lower_lagrangian_gram_lower_start.end(),
              std::inserter(lagrangian_gram_vars_start,
                            lagrangian_gram_vars_start.end()));
    std::copy(tuple.t_upper_lagrangian_gram_lower_start.begin(),
              tuple.t_upper_lagrangian_gram_lower_start.end(),
              std::inserter(lagrangian_gram_vars_start,
                            lagrangian_gram_vars_start.end()));
    verified_gram_vars_start.insert(tuple.verified_polynomial_gram_lower_start);
  }
  EXPECT_EQ(lagrangian_gram_vars_.rows(), lagrangian_gram_vars_count);
  EXPECT_EQ(verified_gram_vars_.rows(), verified_gram_vars_count);
  EXPECT_EQ(verified_gram_vars_start.size(), alternation_tuples_.size());
  EXPECT_EQ(lagrangian_gram_vars_start.size(), alternation_tuples_.size() * 2);

  int separating_plane_vars_count = 0;
  for (const auto& separating_plane : dut.separating_planes()) {
    separating_plane_vars_count += separating_plane.decision_variables.rows();
  }
  EXPECT_EQ(separating_plane_vars_.rows(), separating_plane_vars_count);
  const symbolic::Variables separating_plane_vars_set{separating_plane_vars_};
  EXPECT_EQ(separating_plane_vars_set.size(), separating_plane_vars_count);
  // Now check separating_plane_to_tuples
  EXPECT_EQ(separating_plane_to_tuples_.size(), dut.separating_planes().size());
  std::unordered_set<int> tuple_indices_set;
  for (const auto& tuple_indices : separating_plane_to_tuples_) {
    for (int index : tuple_indices) {
      EXPECT_EQ(tuple_indices_set.count(index), 0);
      tuple_indices_set.emplace(index);
      EXPECT_LT(index, rational_count);
      EXPECT_GE(index, 0);
    }
  }
  EXPECT_EQ(tuple_indices_set.size(), rational_count);
}

TEST_F(DoublePendulumTest, TestCertifyTangentConfigurationSpaceLine) {
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  CspaceFreeLine dut(*diagram_, plant_, scene_graph_,
                           SeparatingPlaneOrder::kAffine, q_star_0_, {},
                           {});


  Eigen::Vector2d q0{0, 0};
  Eigen::Vector2d q1_good{0, 0.1};
  Eigen::Vector2d q1_bad{0.2, 0.5};
  Eigen::Vector2d q1_out_of_limits{-2, -2};

  Eigen::VectorXd s0 = dut.rational_forward_kinematics().ComputeTValue(q0, q_star_0_);
  Eigen::VectorXd s1_good = dut.rational_forward_kinematics().ComputeTValue(q1_good, q_star_0_);
  Eigen::VectorXd s1_bad = dut.rational_forward_kinematics().ComputeTValue(q1_bad, q_star_0_);
  Eigen::VectorXd s1_out_of_limits= dut.rational_forward_kinematics().ComputeTValue(q1_out_of_limits, q_star_0_);

  EXPECT_TRUE(dut.CertifyTangentConfigurationSpaceLine(s0, s1_good));
  EXPECT_FALSE(dut.CertifyTangentConfigurationSpaceLine(s0, s1_bad));
  EXPECT_ANY_THROW(dut.CertifyTangentConfigurationSpaceLine(s0, s1_out_of_limits));
}

GTEST_TEST(AllocatedCertificationProgramConstructorTest, Test) {
  symbolic::Variable x{"x"};  // indeterminate
  symbolic::Variable a{"a"};  // decision variable
  symbolic::Variable b{"b"};  // decision variable
  symbolic::Variable p{"p"};  // parameter we will replace
  symbolic::Polynomial poly{2 * a * p * symbolic::pow(x, 2) +
                                3 * b * symbolic::pow(p, 2) * x + 1 + b + p,
                            symbolic::Variables{x}};
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  solvers::VectorXDecisionVariable decision_variables{2};
  decision_variables << a, b;
  prog->AddDecisionVariables(decision_variables);
  // some constraint
  prog->AddLinearConstraint(a <= 0);

  std::unordered_map<
      symbolic::Polynomial,
      std::unordered_map<symbolic::Monomial,
                         solvers::Binding<solvers::LinearEqualityConstraint>>,
      std::hash<symbolic::Polynomial>, internal::ComparePolynomials>
      polynomial_to_monomial_to_binding_map;
  symbolic::Expression coefficient_expression;
  std::unordered_map<symbolic::Monomial,
                     solvers::Binding<solvers::LinearEqualityConstraint>>
      monomial_to_equality_constraint;
  for (const auto& [monomial, coefficient] :
       poly.monomial_to_coefficient_map()) {
    // construct dummy expression to ensure that the binding has the
    // appropriate variables
    coefficient_expression = 0;
    for (const auto& v : coefficient.GetVariables()) {
      coefficient_expression += v;
    }
    monomial_to_equality_constraint.insert(
        {monomial, (prog->AddLinearEqualityConstraint(0, 0))});
    polynomial_to_monomial_to_binding_map.insert_or_assign(
        poly, monomial_to_equality_constraint);
  }

  EXPECT_NO_THROW(internal::AllocatedCertificationProgram(
      std::move(prog), polynomial_to_monomial_to_binding_map));
}

GTEST_TEST(EvaluatePolynomialsAndUpdateProgram, Test) {
  symbolic::Variable x{"x"};  // indeterminate
  symbolic::Variable a{"a"};  // decision variable
  symbolic::Variable b{"b"};  // decision variable
  symbolic::Variable p{"p"};  // parameter we will replace
  symbolic::Polynomial poly{2 * a * p * symbolic::pow(x, 2) +
                                3 * b * symbolic::pow(p, 2) * x + 1 + b + p,
                            symbolic::Variables{x}};
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  solvers::VectorXDecisionVariable decision_variables{2};
  decision_variables << a, b;
  prog->AddDecisionVariables(decision_variables);

  // some constraint
  prog->AddLinearConstraint(a <= 0);

  std::unordered_map<
      symbolic::Polynomial,
      std::unordered_map<symbolic::Monomial,
                         solvers::Binding<solvers::LinearEqualityConstraint>>,
      std::hash<symbolic::Polynomial>, internal::ComparePolynomials>
      polynomial_to_monomial_to_binding_map;
  symbolic::Expression coefficient_expression;
  std::unordered_map<symbolic::Monomial,
                     solvers::Binding<solvers::LinearEqualityConstraint>>
      monomial_to_equality_constraint;
  for (const auto& [monomial, coefficient] :
       poly.monomial_to_coefficient_map()) {
    // construct dummy expression to ensure that the binding has the
    // appropriate variables
    coefficient_expression = 0;
    for (const auto& v : coefficient.GetVariables()) {
      coefficient_expression += v;
    }
    monomial_to_equality_constraint.insert(
        {monomial, (prog->AddLinearEqualityConstraint(0, 0))});
    polynomial_to_monomial_to_binding_map.insert_or_assign(
        poly, monomial_to_equality_constraint);
  }

  internal::AllocatedCertificationProgram allocated_prog{
      std::move(prog), polynomial_to_monomial_to_binding_map};
  symbolic::Environment env{{p, 2.0}};

  allocated_prog.EvaluatePolynomialsAndUpdateProgram(env);

  auto prog_expected = std::make_unique<solvers::MathematicalProgram>();
  prog_expected->AddDecisionVariables(decision_variables);
  // some constraint
  prog_expected->AddLinearConstraint(a <= 0);
  prog_expected->AddLinearConstraint(2 * 2 * a == 0);  // 2 * a * p = 0
  prog_expected->AddLinearConstraint(3 * 4 * b ==
                                     0);  // 3 * b * symbolic::pow(p, 2) = 0
  prog_expected->AddLinearConstraint(3 + b == 0);  // 1 + b + p = 0

  std::cout << "testing decision variables" << std::endl;
  EXPECT_TRUE(allocated_prog.get_prog()->decision_variables() ==
              prog_expected->decision_variables());

  // there are only linear constraints in this program
  EXPECT_TRUE(solvers::GetProgramType(*allocated_prog.get_prog()) ==
              solvers::ProgramType::kLP);
  EXPECT_TRUE(solvers::GetProgramType(*prog_expected) ==
              solvers::ProgramType::kLP);
  EXPECT_TRUE(allocated_prog.get_prog()->linear_constraints().size() ==
              prog_expected->linear_constraints().size());
  double tol = 1E-12;
  for (const auto& binding : allocated_prog.get_prog()->linear_constraints()) {
    bool same_constraint_found = false;
    for (const auto& binding_expected : prog_expected->linear_constraints()) {
      same_constraint_found = CompareMatrices(binding.evaluator()->GetDenseA(),
                                  binding_expected.evaluator()->GetDenseA(), tol) &&
                              CompareMatrices(binding.evaluator()->lower_bound(),
                                  binding_expected.evaluator()->lower_bound(), tol) &&
                              CompareMatrices(binding.evaluator()->upper_bound(),
                                  binding_expected.evaluator()->upper_bound(), tol) &&
                              CompareMatrices(binding.evaluator()->lower_bound(),
                                  binding_expected.evaluator()->upper_bound(), tol);
      if (same_constraint_found) {
        break;
      }
    }
    EXPECT_TRUE(same_constraint_found);
  }
}

}  // namespace multibody
}  // namespace drake