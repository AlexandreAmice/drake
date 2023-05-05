#include "drake/geometry/optimization/dev/cspace_free_path.h"

#include <gtest/gtest.h>

#include "drake/geometry/optimization/cspace_free_internal.h"
#include "drake/geometry/optimization/dev/test/c_iris_path_test_utilities.h"
#include "drake/geometry/optimization/test/c_iris_test_utilities.h"
#include "drake/multibody/rational/rational_forward_kinematics_internal.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace geometry {
namespace optimization {

TEST_F(CIrisToyRobotTest, CspaceFreePathConstructor) {
  // Test CspaceFreePolytope constructor.
  const Eigen::Vector3d q_star(0, 0, 0);
  for (int maximum_path_degree = 1; maximum_path_degree < 3;
       ++maximum_path_degree) {
    for (int plane_order = 1; plane_order < 3; ++plane_order) {
      CspaceFreePathTester tester(plant_, scene_graph_, q_star,
                                  maximum_path_degree, plane_order);
      const CspaceFreePath& dut = tester.cspace_free_path();
      EXPECT_EQ(dut.max_degree(), maximum_path_degree);
      EXPECT_EQ(dut.plane_order(), plane_order);
      // check that the path map is properly instantiated
      VectorX<symbolic::Variable> s_vars{
          tester.get_rational_forward_kin()->s()};
      for (int i = 0; i < s_vars.size(); ++i) {
        EXPECT_GT(tester.get_path().count(s_vars(i)), 0);

        const symbolic::Polynomial& path_component{
            tester.get_path().at(s_vars(i))};
        EXPECT_EQ(path_component.indeterminates().size(), 1);
        EXPECT_EQ((*path_component.indeterminates().cbegin()), tester.get_mu());
        EXPECT_EQ(static_cast<unsigned int>(path_component.TotalDegree()),
                  maximum_path_degree);
        EXPECT_EQ(path_component.monomial_to_coefficient_map().size(),
                  maximum_path_degree + 1);
      }

      int num_planes_expected = 0;

      const auto link_geometries =
          internal::GetCollisionGeometries(*plant_, *scene_graph_);
      // Count the expected number of planes by hand.
      num_planes_expected +=
          link_geometries.at(plant_->world_body().index()).size() *
          // Don't include world_body to body0 as there is only a weld joint
          // between them.
          (link_geometries.at(body_indices_[1]).size() +
           link_geometries.at(body_indices_[2]).size() +
           link_geometries.at(body_indices_[3]).size());
      num_planes_expected += link_geometries.at(body_indices_[0]).size() *
                             link_geometries.at(body_indices_[2]).size();
      num_planes_expected += link_geometries.at(body_indices_[1]).size() *
                             link_geometries.at(body_indices_[3]).size();
      num_planes_expected += link_geometries.at(body_indices_[2]).size() *
                             link_geometries.at(body_indices_[3]).size();
      EXPECT_EQ(tester.get_separating_planes().size(), num_planes_expected);
      EXPECT_EQ(dut.plane_order(), plane_order);
      for (const auto& [geometry_pair, plane_index] :
           dut.map_geometries_to_separating_planes()) {
        // check plane
        const auto& plane = dut.separating_planes()[plane_index];
        EXPECT_EQ(plane.plane_degree, plane_order);
        if (plane.positive_side_geometry->id() <
            plane.negative_side_geometry->id()) {
          EXPECT_EQ(geometry_pair.first(), plane.positive_side_geometry->id());
          EXPECT_EQ(geometry_pair.second(), plane.negative_side_geometry->id());
        } else {
          EXPECT_EQ(geometry_pair.first(), plane.negative_side_geometry->id());
          EXPECT_EQ(geometry_pair.second(), plane.positive_side_geometry->id());
        }
        // Check the expressed body.
        EXPECT_EQ(plane.expressed_body,
                  multibody::internal::FindBodyInTheMiddleOfChain(
                      *plant_, plane.positive_side_geometry->body_index(),
                      plane.negative_side_geometry->body_index()));
        for (int i = 0; i < 3; ++i) {
          EXPECT_EQ(plane.a(i).TotalDegree(), plane_order);
          EXPECT_EQ(plane.a(i).indeterminates(),
                    symbolic::Variables({dut.mu()}));
        }
        EXPECT_EQ(plane.b.TotalDegree(), plane_order);
        EXPECT_EQ(plane.b.indeterminates(), symbolic::Variables({dut.mu()}));
      }
      EXPECT_EQ(dut.y_slack().size(), 3);
      // We will check the path rationals in a later test.
    }
  }
}

// This struct is used to check whether the rationals in cspace_free_path and
// cspace_free_polytope are properly related.
struct PathEvaluator {
  explicit PathEvaluator(const CspaceFreePathTester& tester)
      : mu_{tester.get_mu()} {
    std::default_random_engine generator;
    generator.seed(0);
    std::uniform_real_distribution<> coeff_distribution(-10, 10);

    for (const auto& [var, poly] : tester.get_path()) {
      for (const auto& coeff_var : poly.decision_variables()) {
        path_coefficient_evaluator_.insert(coeff_var,
                                           coeff_distribution(generator));
      }
      symbolic::Polynomial evaled_poly{poly};
      evaled_path_.insert(
          {var, evaled_poly.EvaluatePartial(path_coefficient_evaluator_)});
    }
  }

  symbolic::Environment MakeCspaceFreePolytopeAndPathEvaluator(double mu_val) {
    symbolic::Environment ret{path_coefficient_evaluator_};
    ret.insert(mu_, mu_val);
    for (const auto& [var, poly] : evaled_path_) {
      ret.insert(var, poly.Evaluate(ret));
    }
    return ret;
  }

 private:
  const symbolic::Variable mu_;
  std::unordered_map<symbolic::Variable, symbolic::Polynomial> evaled_path_;
  symbolic::Environment path_coefficient_evaluator_;
};

// TEST_F(CIrisToyRobotTest, CspaceFreePathGeneratePathRationalsTest) {
//  const Eigen::Vector3d q_star(0, 0, 0);
//  for (unsigned int maximum_path_degree = 1; maximum_path_degree < 3;
//       ++maximum_path_degree) {
//    CspaceFreePathTester tester(plant_, scene_graph_, q_star,
//                                maximum_path_degree, 1);;
//    PathEvaluator evaluator{tester};
//    const Eigen::Vector4d mu_test_values{0, 0.25, 0.77, 1};
//
//    // TODO(Alexandre.Amice) check that this is the right size.
////    EXPECT_EQ(tester.get_path_plane_geometries().size(),
////              tester.cspace_free_path().separating_planes().size());
//    const symbolic::Variables mu_indets{tester.get_mu()};
//    const symbolic::Variables y_slack{tester.cspace_free_path().y_slack()};
//
//
//    // TODO(Alexandre.Amice) rewrite this test.
////    for (const auto& free_path_plane_geometry :
////         tester.get_path_plane_geometries()) {
////      const auto& free_polytope_plane_geometry =
////          tester.get_mutable_plane_geometries().at(
////              free_path_plane_geometry.plane_index);
////
////      const auto& positive_side =
////          std::tie(free_path_plane_geometry.positive_side_conditions,
////                   free_polytope_plane_geometry.positive_side_rationals);
////      const auto& negative_side =
////          std::tie(free_path_plane_geometry.negative_side_conditions,
////                   free_polytope_plane_geometry.negative_side_rationals);
////
////      for (const auto& [path_conditions, polytope_rationals] :
////           {positive_side, negative_side}) {
////        EXPECT_EQ(path_conditions.size(), polytope_rationals.size());
////        int i = 0;
////        // Path rationals is a list while polytope_rationals is a vector so
/// we /        // iterate through the list and increment a counter for the
/// random /        // access to the vector. /        for (const auto&
/// path_rational : path_conditions) { /          const symbolic::Polynomial&
/// path_condition{path_rational.get_poly()}; /          const
/// symbolic::Polynomial& polytope_condition{ /
/// polytope_rationals.at(i).numerator()}; /          ++i;
////
////          // Gets the y_slacks needed to implement this matrix SOS
/// condition. /          const symbolic::Variables path_y_slack = /
/// intersect(y_slack, path_condition.indeterminates()); /          const
/// symbolic::Variables polytope_y_slack = /              intersect(y_slack,
/// polytope_condition.indeterminates()); /          // The same y_slacks should
/// be used to implement both the polytope and /          // path condition. /
/// EXPECT_EQ(path_y_slack, polytope_y_slack);
////
////          symbolic::Variables mu_and_y_indets{tester.get_mu()};
////          mu_and_y_indets.insert(path_y_slack.begin(), path_y_slack.end());
////
////          // The numerator is a function of the new path variable and of the
////          // matrix SOS variables.
////          EXPECT_EQ(path_condition.indeterminates(), mu_and_y_indets);
////
////          // The condition should be a quadratic in the y_slack.
////          for (const auto& y : path_y_slack) {
////            EXPECT_EQ(path_condition.Degree(y), 2);
////          }
////
////          auto compute_total_s_degree =
////              [&tester](const symbolic::Polynomial& poly) {
////                int degree{0};
////                for (const auto& [m, c] :
/// poly.monomial_to_coefficient_map()) { /                  int
/// cur_s_degree{0}; /                  VectorX<symbolic::Variable> s_vars{ /
///tester.get_rational_forward_kin()->s()}; /                  for (int k = 0; k
///< s_vars.size(); ++k) { /                    cur_s_degree +=
///m.degree(s_vars(k)); /                  } /                  degree =
///std::max(degree, cur_s_degree); /                } /                return
///degree; /              }; /          // The degree in terms of the path
///variable should be the total degree /          // of the original rational
/// with respect to the variables s_set_ times /          // the path degree. /
/// EXPECT_EQ( / static_cast<unsigned
/// int>(path_condition.Degree(tester.get_mu())), / maximum_path_degree *
/// compute_total_s_degree(polytope_condition));
////
////          // Now check that the path_conditions are actually properly
////          // substituted.
////          for (int j = 0; j < mu_test_values.rows(); ++j) {
////            const symbolic::Polynomial path_condition_eval{
////                path_condition.EvaluatePartial(
////                    evaluator.MakeCspaceFreePolytopeAndPathEvaluator(
////                        mu_test_values(j)))};
////            const symbolic::Polynomial polytope_condition_eval{
////                path_condition.EvaluatePartial(
////                    evaluator.MakeCspaceFreePolytopeAndPathEvaluator(
////                        mu_test_values(j)))};
////            EXPECT_TRUE(path_condition_eval.CoefficientsAlmostEqual(
////                polytope_condition_eval, 1e-10));
////          }
////        }
////      }
////    }
//  }
//}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
