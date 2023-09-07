#include <gtest/gtest.h>

#include "drake/common/polynomial.h"
#include "drake/common/trajectories/bezier_curve.h"
#include "drake/geometry/optimization/dev/cspace_free_path.h"
#include "drake/geometry/optimization/dev/test/c_iris_path_test_utilities.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/test/c_iris_test_utilities.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace geometry {
namespace optimization {
namespace {
[[nodiscard]] Eigen::VectorXd ProjectToPolytope(const Eigen::VectorXd& sample,
                                                const HPolyhedron& polyhedron) {
  DRAKE_DEMAND(sample.rows() == polyhedron.ambient_dimension());
  if (polyhedron.PointInSet(sample)) {
    return sample;
  } else {
    solvers::MathematicalProgram prog;
    auto s = prog.NewContinuousVariables(sample.rows());
    prog.AddLinearConstraint(
        polyhedron.A(),
        Eigen::VectorXd::Constant(polyhedron.b().rows(),
                                  -std::numeric_limits<double>::infinity()),
        polyhedron.b(), s);
    prog.AddQuadraticErrorCost(Eigen::MatrixXd::Identity(s.rows(), s.rows()),
                               sample, s);
    const auto result = solvers::Solve(prog);
    DRAKE_DEMAND(result.is_success());
    return result.GetSolution(s);
  }
}

VectorX<Polynomiald> MakeBezierCurvePolynomialPath(
    const Eigen::VectorXd& s0, const Eigen::VectorXd& s_end, int curve_order,
    std::optional<HPolyhedron> poly_to_check_containment = std::nullopt) {
  if (poly_to_check_containment.has_value()) {
    EXPECT_TRUE(poly_to_check_containment.value().PointInSet(s0));
    EXPECT_TRUE(poly_to_check_containment.value().PointInSet(s_end));
  }

  Eigen::MatrixXd control_points{s0.rows(), curve_order + 1};
  control_points.col(0) = s0;
  control_points.col(curve_order) = s_end;

  const int div = 1000;
  Eigen::VectorXd orth_offset{s0.rows()};
  for (int i = 0; i < s0.rows(); ++i) {
    orth_offset << std::pow(-1, i % 2) / div;
  }
  for (int i = 1; i < curve_order - 1; ++i) {
    // another control point slightly off the straight line path between s0
    // and s_end.
    Eigen::VectorXd si = i * (s0 + s_end) / curve_order + orth_offset;

    if (poly_to_check_containment.has_value()) {
      si = ProjectToPolytope(si, poly_to_check_containment.value());
    }
    control_points.col(i) = si;
  }
  if (poly_to_check_containment.has_value()) {
    poly_to_check_containment.value().PointInSet(control_points);
  }
  //
  //
  //  if (poly_to_check_containment.has_value() && curve_order > 1) {
  //    optimization::VPolytope control_polytope_v{control_points};
  //    HPolyhedron control_polytope{control_polytope_v};
  //    bool contained =
  //    control_polytope.ContainedIn(poly_to_check_containment.value());
  //    EXPECT_TRUE(contained);
  //  }

  const trajectories::BezierCurve<double> path{0, 1, control_points};
  const MatrixX<symbolic::Expression> bezier_path_expr =
      path.GetExpression(symbolic::Variable("t"));

  VectorX<Polynomiald> bezier_poly_path{bezier_path_expr.rows()};
  for (int r = 0; r < bezier_path_expr.rows(); ++r) {
    symbolic::Polynomial sym_poly{bezier_path_expr(r)};
    Eigen::VectorXd coefficients{sym_poly.TotalDegree() + 1};
    for (const auto& [monom, coeff] : sym_poly.monomial_to_coefficient_map()) {
      coefficients(monom.total_degree()) = coeff.Evaluate();
    }
    bezier_poly_path(r) = Polynomiald(coefficients);
  }
  return bezier_poly_path;
}

// @param a Maps the plane index to the separating plane parameter `a` in {x|
// aᵀx+b=0}
// @param b Maps the plane index to the separating plane parameter `b` in {x|
// aᵀx+b=0}
void CheckSeparationBySamples(
    const CspaceFreePathTester& tester, const systems::Diagram<double>& diagram,
    const Eigen::Ref<const Eigen::VectorXd>& mu_samples,
    const Eigen::Ref<const VectorX<Polynomiald>>& path,
    const std::unordered_map<int, Vector3<symbolic::Polynomial>>& a,
    const std::unordered_map<int, symbolic::Polynomial>& b,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreePolytope::IgnoredCollisionPairs& ignored_collision_pairs) {
  auto diagram_context = diagram.CreateDefaultContext();
  const auto& plant = tester.get_rational_forward_kin()->plant();
  const auto& scene_graph = tester.get_scene_graph();
  auto& plant_context =
      plant.GetMyMutableContextFromRoot(diagram_context.get());
  auto& scene_graph_context =
      scene_graph.GetMyMutableContextFromRoot(diagram_context.get());
  std::unique_ptr<AbstractValue> state_value =
      scene_graph.get_query_output_port().Allocate();

  for (int i = 0; i < mu_samples.rows(); ++i) {
    symbolic::Environment env;
    env.insert(tester.get_mu(), mu_samples(i));
    Eigen::VectorXd s{path.rows()};
    for (int r = 0; r < s.rows(); ++r) {
      s(r) = path(r).EvaluateUnivariate(mu_samples(i));
    }
    const Eigen::VectorXd q_val =
        tester.get_rational_forward_kin()->ComputeQValue(s, q_star);
    plant.SetPositions(&plant_context, q_val);
    const QueryObject<double>& query_object =
        state_value->get_value<QueryObject<double>>();
    scene_graph.get_query_output_port().Calc(scene_graph_context,
                                             state_value.get());
    EXPECT_FALSE(query_object.HasCollisions());
    for (int plane_index = 0;
         plane_index < static_cast<int>(tester.get_separating_planes().size());
         ++plane_index) {
      const auto& plane = tester.get_separating_planes()[plane_index];
      if (ignored_collision_pairs.count(SortedPair<geometry::GeometryId>(
              plane.positive_side_geometry->id(),
              plane.negative_side_geometry->id())) == 0 &&
          a.count(plane_index) > 0 && b.count(plane_index) > 0) {
        Eigen::Vector3d a_val;
        for (int j = 0; j < 3; ++j) {
          a_val(j) = a.at(plane_index)(j).Evaluate(env);
        }
        const double b_val = b.at(plane_index).Evaluate(env);
        EXPECT_GE(
            DistanceToHalfspace(*plane.positive_side_geometry, a_val, b_val,
                                plane.expressed_body, PlaneSide::kPositive,
                                plant, plant_context),
            0);
        EXPECT_GE(
            DistanceToHalfspace(*plane.negative_side_geometry, a_val, b_val,
                                plane.expressed_body, PlaneSide::kNegative,
                                plant, plant_context),
            0);
      }
    }
  }
  std::cout << std::endl;
}

void CheckForCollisionAlongPath(
    const CspaceFreePathTester& tester, const systems::Diagram<double>& diagram,
    const Eigen::Ref<const Eigen::VectorXd>& mu_samples,
    const Eigen::Ref<const VectorX<Polynomiald>>& path,
    const Eigen::Ref<const Eigen::VectorXd>& q_star) {
  auto diagram_context = diagram.CreateDefaultContext();
  const auto& plant = tester.get_rational_forward_kin()->plant();
  const auto& scene_graph = tester.get_scene_graph();
  auto& plant_context =
      plant.GetMyMutableContextFromRoot(diagram_context.get());
  auto& scene_graph_context =
      scene_graph.GetMyMutableContextFromRoot(diagram_context.get());
  std::unique_ptr<AbstractValue> state_value =
      scene_graph.get_query_output_port().Allocate();
  bool collision_found = false;
  for (int i = 0; i < mu_samples.rows(); ++i) {
    symbolic::Environment env;
    env.insert(tester.get_mu(), mu_samples(i));
    Eigen::VectorXd s{path.rows()};
    for (int r = 0; r < s.rows(); ++r) {
      s(r) = path(r).EvaluateUnivariate(mu_samples(i));
    }
    const Eigen::VectorXd q_val =
        tester.get_rational_forward_kin()->ComputeQValue(s, q_star);
    plant.SetPositions(&plant_context, q_val);
    const QueryObject<double>& query_object =
        state_value->get_value<QueryObject<double>>();
    scene_graph.get_query_output_port().Calc(scene_graph_context,
                                             state_value.get());
    collision_found = query_object.HasCollisions();
    if (collision_found) {
      break;
    }
  }
  EXPECT_TRUE(collision_found);
}
}  // namespace
TEST_F(CIrisToyRobotTest, MakeAndSolveIsGeometrySeparableOnPathProgram) {
  const Eigen::Vector3d q_star(0, 0, 0);
  Eigen::Matrix<double, 9, 3> C_good;
  // clang-format off
  C_good << 1, 1, 0,
       -1, -1, 0,
       -1, 0, 1,
       1, 0, -1,
       0, 1, 1,
       0, -1, -1,
       1, 0, 1,
       1, 1, -1,
       1, -1, 1;
  // clang-format on
  Eigen::Matrix<double, 9, 1> d_good;
  d_good << 0.1, 0.1, 0.1, 0.02, 0.02, 0.2, 0.1, 0.1, 0.2;
  // This polyhedron is fully collision free and can be certified as such using
  // CspaceFreePolytope.
  const HPolyhedron c_free_polyhedron{C_good, d_good};
  const SortedPair<geometry::GeometryId> geometry_pair{body0_box_,
                                                       body2_sphere_};

  CspaceFreePath::FindSeparationCertificateGivenPathOptions
      find_certificate_options;
  find_certificate_options.verbose = false;
  solvers::MosekSolver solver;
  find_certificate_options.solver_id = solver.id();

  const int num_samples{100};
  Eigen::VectorXd mu_samples{100};
  mu_samples = Eigen::VectorXd::LinSpaced(num_samples, 0, 1);

  const Eigen::Vector3d s0_safe{0.06, -0.02, 0.04};
  const Eigen::Vector3d s_end_safe{-0.17, 0.08, -0.19};

  const Eigen::Vector3d s0_unsafe{-1.74, -0.22, 0.24};
  const Eigen::Vector3d s_end_unsafe{0.84, 2.31, 1.47};

  const int plane_order = 1;
  //  const Eigen::Vector3d s0_unsafe{0.65*M_PI, 1.63, 2.9};
  //  const Eigen::Vector3d s_end_unsafe{-1.63, 0.612, -0.65};

  for (const int maximum_path_degree : {1, 4}) {
    CspaceFreePathTester tester(plant_, scene_graph_, q_star,
                                maximum_path_degree, plane_order);

    // Check that we can certify paths up to the maximum degree.
    for (int bezier_curve_order = 1; bezier_curve_order <= maximum_path_degree;
         ++bezier_curve_order) {
      // Construct a polynonomial of degree bezier_curve_order <=
      // maximum_path_degree. By constructing this with the control points
      // inside c_free_polyhedron, we guarantee that the trajectory inside
      // will be colllision free.
      VectorX<Polynomiald> bezier_poly_path_safe =
          MakeBezierCurvePolynomialPath(s0_safe, s_end_safe, bezier_curve_order,
                                        c_free_polyhedron);
      auto separation_certificate_program =
          tester.cspace_free_path().MakeIsGeometrySeparableOnPathProgram(
              geometry_pair, bezier_poly_path_safe);
      auto separation_certificate_result =
          tester.cspace_free_path().SolveSeparationCertificateProgram(
              separation_certificate_program, find_certificate_options);
      EXPECT_TRUE(separation_certificate_result.result.is_success());
      if (separation_certificate_result.result.is_success()) {
        CheckSeparationBySamples(tester, *diagram_, mu_samples,
                                 bezier_poly_path_safe,
                                 {{separation_certificate_result.plane_index,
                                   separation_certificate_result.a}},
                                 {{separation_certificate_result.plane_index,
                                   separation_certificate_result.b}},
                                 q_star, {});
      }

      VectorX<Polynomiald> bezier_poly_path_unsafe =
          MakeBezierCurvePolynomialPath(s0_unsafe, s_end_unsafe,
                                        bezier_curve_order);
      auto separation_certificate_program2 =
          tester.cspace_free_path().MakeIsGeometrySeparableOnPathProgram(
              geometry_pair, bezier_poly_path_unsafe);

      auto separation_certificate_result2 =
          tester.cspace_free_path().SolveSeparationCertificateProgram(
              separation_certificate_program2, find_certificate_options);
      EXPECT_FALSE(separation_certificate_result2.result.is_success());
      if (!separation_certificate_result2.result.is_success()) {
        CheckForCollisionAlongPath(tester, *diagram_, mu_samples,
                                   bezier_poly_path_unsafe, q_star);
      }
    }
  }
}

// TEST_F(CIrisToyRobotTest, FindSeparationCertificateGivenPathSuccess) {
//  const Eigen::Vector3d q_star(0, 0, 0);
//  Eigen::Matrix<double, 9, 3> C_good;
//  // clang-format off
//  C_good << 1, 1, 0,
//       -1, -1, 0,
//       -1, 0, 1,
//       1, 0, -1,
//       0, 1, 1,
//       0, -1, -1,
//       1, 0, 1,
//       1, 1, -1,
//       1, -1, 1;
//  // clang-format on
//  Eigen::Matrix<double, 9, 1> d_good;
//  d_good << 0.1, 0.1, 0.1, 0.02, 0.02, 0.2, 0.1, 0.1, 0.2;
//  // This polyhedron is fully collision free and can be certified as such
//  using
//  // CspaceFreePolytope.
//  const HPolyhedron c_free_polyhedron{C_good, d_good};
//  const SortedPair<geometry::GeometryId> geometry_pair{body0_box_,
//                                                       body2_sphere_};
//  const CspaceFreePolytope::IgnoredCollisionPairs ignored_collision_pairs{
//      SortedPair<geometry::GeometryId>(world_box_, body2_sphere_)};
//
//  CspaceFreePath::FindSeparationCertificateGivenPathOptions
//      find_certificate_options;
//  find_certificate_options.verbose = false;
//  find_certificate_options.num_threads = 1;
//  find_certificate_options.terminate_segment_certification_at_failure = false;
//  find_certificate_options.terminate_path_certification_at_failure = false;
//  solvers::MosekSolver solver;
//  find_certificate_options.solver_id = solver.id();
//
//  const int num_samples{100};
//  Eigen::VectorXd mu_samples{100};
//  mu_samples = Eigen::VectorXd::LinSpaced(num_samples, 0, 1);
//
//  const Eigen::Vector3d s0_safe{0.01, -0.02, 0.04};
//  const Eigen::Vector3d s_end_safe{-0.17, 0.08, -0.19};
//
//  const int num_trials = 15;
//
//  const int plane_order = 8;
//  for (const int maximum_path_degree : {1}) {
//    CspaceFreePathTester tester(plant_, scene_graph_, q_star,
//                                maximum_path_degree, plane_order);
//
//    // Check that we can certify paths up to the maximum degree.
//    for (int bezier_curve_order = 1; bezier_curve_order <=
//    maximum_path_degree;
//         ++bezier_curve_order) {
//      // Construct a polynonomial of degree bezier_curve_order <=
//      // maximum_path_degree. By constructing this with the control points
//      // inside c_free_polyhedron, we guarantee that the trajectory inside
//      // will be colllision free.
//      MatrixX<Polynomiald> bezier_poly_path_safe{s0_safe.rows(), num_trials};
//      for (int i = 0; i < num_trials; ++i) {
//        bezier_poly_path_safe.col(i) = MakeBezierCurvePolynomialPath(
//            s0_safe, s_end_safe, bezier_curve_order, c_free_polyhedron);
//      }
//      for (int i = 0; i < mu_samples.rows(); ++i) {
//        Eigen::VectorXd point{bezier_poly_path_safe.rows()};
//        for (int j = 0; j < bezier_poly_path_safe.rows(); j++) {
//          point(j) =
//              bezier_poly_path_safe(j, 0).EvaluateUnivariate(mu_samples(i));
//        }
//        EXPECT_TRUE(c_free_polyhedron.PointInSet(point));
//      }
//      std::unordered_map<SortedPair<geometry::GeometryId>,
//                         std::vector<std::optional<
//                             CspaceFreePath::SeparationCertificateResult>>>
//          certificates;
//      auto start = std::chrono::high_resolution_clock::now();
//      std::vector<std::optional<bool>> piece_is_safe =
//          tester.cspace_free_path().FindSeparationCertificateGivenPath(
//              bezier_poly_path_safe, ignored_collision_pairs,
//              find_certificate_options, &certificates);
//      auto end = std::chrono::high_resolution_clock::now();
//      auto duration =
//          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//
//      for (const auto& [pair, certs] : certificates) {
//        unused(pair);
//        std::cout
//            << std::all_of(
//                   certs.begin(), certs.end(),
//                   [](std::optional<CspaceFreePath::SeparationCertificateResult>
//                          cert) {
//                     return cert.has_value();
//                   })
//            << std::endl;
//      }
//
//      // To get the value of duration use the count()
//      // member function on the duration object
//      std::cout << "Time taken by function: " << duration.count() <<
//      std::endl; EXPECT_EQ(certificates.size(),
//                tester.cspace_free_path().separating_planes().size() -
//                    ignored_collision_pairs.size());
//      for (const auto& [pair, cert] : certificates) {
//        EXPECT_EQ(static_cast<int>(cert.size()), num_trials);
//        //        EXPECT_TRUE(std::all_of(cert.begin(), cert.end(),
//        // [](std::optional<CspaceFreePolytope::SeparationCertificateResult>
//        //                              flag) {
//        //                                return flag.has_value();
//        //                              }));
//      }
//
//      for (int i = 0; i < static_cast<int>(piece_is_safe.size()); ++i) {
//        //        if (!(piece_is_safe.at(i).has_value() &&
//        //        piece_is_safe.at(i).value())) {
//        bool piece_has_value = piece_is_safe.at(i).has_value();
//        std::cout << fmt::format("Piece is safe index {}, has value = {}", i,
//                                 piece_has_value)
//                  << std::endl;
//        if(piece_has_value) {
//          std::cout << fmt::format("Piece is safe index {}, value = {}", i,
//                                 piece_is_safe.at(i).value())
//                  << std::endl;
//        }
//        std::cout << std::endl;
//
//        //        }
//        //        std::cout << fmt::format("Piece is safe index {}, value =
//        {}",
//        //        i,
//        //                                 piece_is_safe.at(i).value_or("no
//        //                                 value"))
//        //                  << std::endl;
//      }
//      std::cout << std::endl;
//      EXPECT_TRUE(std::all_of(piece_is_safe.begin(), piece_is_safe.end(),
//                              [](std::optional<bool> flag) {
//                                return flag.has_value() && flag.value();
//                              }));
//    }
//  }
//  EXPECT_TRUE(false);
//}

TEST_F(CIrisRobotPolytopicGeometryTest,
       FindSeparationCertificateGivenPathSuccessPolytopic) {
  const Eigen::Vector4d q_star(0, 0, 0, 0);
  Eigen::Matrix<double, 12, 4> C_good;
  // clang-format off
  C_good << 1, 1, 0, 0,
       -1, -1, 0, 1,
       -1, 0, 1, -1,
       1, 0, -1, 0,
       0, 1, 1, 0,
       0, -1, -1, 1,
       1, 0, 1, 0,
       1, 1, -1, 1,
       1, -1, 1, -1,
       0, 1, 1, 1,
       1, 1, 1, -1,
       1, 0, 0, 1;
  // clang-format on
  Eigen::Matrix<double, 12, 1> d_good;
  d_good << 0.1, 0.1, 0.1, 0.02, 0.02, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.02;
  for (int i = 0; i < C_good.rows(); ++i) {
    const double C_row_norm = C_good.row(i).norm();
    C_good.row(i) = C_good.row(i) / C_row_norm;
    d_good(i) = d_good(i) / C_row_norm;
  }

  // This polyhedron is fully collision free and can be certified as such using
  // CspaceFreePolytope.
  const HPolyhedron c_free_polyhedron{C_good, d_good};
  CspaceFreePolytope::IgnoredCollisionPairs ignored_collision_pairs{};

  CspaceFreePath::FindSeparationCertificateGivenPathOptions
      find_certificate_options;
  find_certificate_options.verbose = false;
  find_certificate_options.num_threads = 1;
  find_certificate_options.terminate_segment_certification_at_failure = false;
  find_certificate_options.terminate_path_certification_at_failure = false;
  solvers::MosekSolver solver;
  find_certificate_options.solver_id = solver.id();

  const int num_samples{100};
  Eigen::VectorXd mu_samples{num_samples};
  mu_samples = Eigen::VectorXd::LinSpaced(num_samples, 0, 1);

  const Eigen::Vector4d s0_safe = {-0.016, -0.039, 0.0059, 0.018};
  const Eigen::Vector4d s_end_safe{-0.012, -0.044, 0.012, -0.015};

  const int num_trials = 15;

  const int plane_order = 8;
  for (const int maximum_path_degree : {1}) {
    CspaceFreePathTester tester(plant_, scene_graph_, q_star,
                                maximum_path_degree, plane_order);

    // Check that we can certify paths up to the maximum degree.
    for (int bezier_curve_order = 1; bezier_curve_order <= maximum_path_degree;
         ++bezier_curve_order) {
      // Construct a polynonomial of degree bezier_curve_order <=
      // maximum_path_degree. By constructing this with the control points
      // inside c_free_polyhedron, we guarantee that the trajectory inside
      // will be colllision free.
      MatrixX<Polynomiald> bezier_poly_path_safe{s0_safe.rows(), num_trials};
      for (int i = 0; i < num_trials; ++i) {
        bezier_poly_path_safe.col(i) = MakeBezierCurvePolynomialPath(
            s0_safe, s_end_safe, bezier_curve_order, c_free_polyhedron);
      }
      for (int i = 0; i < mu_samples.rows(); ++i) {
        Eigen::VectorXd point{bezier_poly_path_safe.rows()};
        for (int j = 0; j < bezier_poly_path_safe.rows(); j++) {
          point(j) =
              bezier_poly_path_safe(j, 0).EvaluateUnivariate(mu_samples(i));
        }
        EXPECT_TRUE(c_free_polyhedron.PointInSet(point));
      }
      std::vector<std::unordered_map<
          SortedPair<geometry::GeometryId>,
          std::optional<CspaceFreePath::SeparationCertificateResult>>>
          certificates;
      auto start = std::chrono::high_resolution_clock::now();
      //      std::vector<std::optional<bool>> piece_is_safe =
      //          tester.cspace_free_path().FindSeparationCertificateGivenPath(
      //              bezier_poly_path_safe, ignored_collision_pairs,
      //              find_certificate_options, &certificates);
      std::vector<CspaceFreePath::FindSeparationCertificateStatistics>
          certification_statistics =
              tester.cspace_free_path().FindSeparationCertificateGivenPath(
                  bezier_poly_path_safe, ignored_collision_pairs,
                  find_certificate_options, &certificates);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "Time taken by function: " << duration.count() << std::endl;

      EXPECT_EQ(static_cast<int>(certification_statistics.size()),
                static_cast<int>(bezier_poly_path_safe.cols()));

      // Check that all the pieces are safe
      EXPECT_TRUE(std::all_of(certification_statistics.begin(), certification_statistics.end(),
                              [](CspaceFreePath::FindSeparationCertificateStatistics stat) {
                                return stat.certified_safe();
                              }));

      // Check that all the certificates are populated
      EXPECT_EQ(certificates.size(), bezier_poly_path_safe.cols());
      for (int i = 0; i < static_cast<int>(certificates.size()); ++i) {
        std::unordered_map<
            SortedPair<geometry::GeometryId>,
            std::optional<CspaceFreePath::SeparationCertificateResult>>
            pair_to_certs_map = certificates[i];
        EXPECT_EQ(static_cast<int>(pair_to_certs_map.size()),
                  tester.cspace_free_path().separating_planes().size() -
                      ignored_collision_pairs.size());
        EXPECT_TRUE(std::all_of(
            pair_to_certs_map.begin(), pair_to_certs_map.end(),
            [](std::pair<
                SortedPair<geometry::GeometryId>,
                std::optional<CspaceFreePath::SeparationCertificateResult>>
                   flag) {
              return flag.second.has_value();
            }));
      }
    }
  }
}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
