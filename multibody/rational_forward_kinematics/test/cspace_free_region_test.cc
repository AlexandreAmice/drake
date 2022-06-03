#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"

#include <chrono>
#include <limits>

#include <gtest/gtest.h>

#include "drake/common/temp_directory.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/geometry/collision_filter_declaration.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/plant/coulomb_friction.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
using drake::Vector3;
using drake::VectorX;
using drake::math::RigidTransformd;
using drake::math::RotationMatrixd;
using drake::multibody::BodyIndex;

const double kInf = std::numeric_limits<double>::infinity();

class IiwaCspaceTest : public ::testing::Test {
 public:
  IiwaCspaceTest() {
    auto iiwa = ConstructIiwaPlant("iiwa14_no_collision.sdf", false);
    iiwa_link_.resize(iiwa->num_bodies());
    for (int i = 0; i < 8; ++i) {
      iiwa_link_[i] =
          iiwa->GetBodyByName("iiwa_link_" + std::to_string(i)).index();
    }
    systems::DiagramBuilder<double> builder;
    plant_ = builder.AddSystem<MultibodyPlant<double>>(std::move(iiwa));
    scene_graph_ = builder.AddSystem<geometry::SceneGraph<double>>();
    plant_->RegisterAsSourceForSceneGraph(scene_graph_);

    builder.Connect(scene_graph_->get_query_output_port(),
                    plant_->get_geometry_query_input_port());

    builder.Connect(
        plant_->get_geometry_poses_output_port(),
        scene_graph_->get_source_pose_port(plant_->get_source_id().value()));

    // Arbitrarily add some polytopes to links
    link7_polytopes_id_.push_back(plant_->RegisterCollisionGeometry(
        plant_->get_body(iiwa_link_[7]), {}, geometry::Box(0.1, 0.1, 0.2),
        "link7_box1", CoulombFriction<double>()));
    const RigidTransformd X_7P{RotationMatrixd(Eigen::AngleAxisd(
                                   0.2 * M_PI, Eigen::Vector3d::UnitX())),
                               {0.1, 0.2, -0.1}};
    link7_polytopes_id_.push_back(plant_->RegisterCollisionGeometry(
        plant_->get_body(iiwa_link_[7]), X_7P, geometry::Box(0.1, 0.2, 0.1),
        "link7_box2", CoulombFriction<double>()));

    const RigidTransformd X_5P{X_7P.rotation(), {-0.2, 0.1, 0}};
    link5_polytopes_id_.push_back(plant_->RegisterCollisionGeometry(
        plant_->get_body(iiwa_link_[5]), X_5P, geometry::Box(0.2, 0.1, 0.2),
        "link5_box1", CoulombFriction<double>()));

    RigidTransformd X_WP = X_5P * Eigen::Translation3d(0.15, -0.1, 0.05);
    obstacles_id_.push_back(plant_->RegisterCollisionGeometry(
        plant_->world_body(), X_WP, geometry::Box(0.1, 0.2, 0.15), "world_box1",
        CoulombFriction<double>()));
    X_WP = X_WP * RigidTransformd(RotationMatrixd(Eigen::AngleAxisd(
                      -0.1 * M_PI, Eigen::Vector3d::UnitY())));
    obstacles_id_.push_back(plant_->RegisterCollisionGeometry(
        plant_->world_body(), X_WP, geometry::Box(0.1, 0.25, 0.15),
        "world_box2", CoulombFriction<double>()));
    link1_polytopes_id_.push_back(plant_->RegisterCollisionGeometry(
        plant_->GetBodyByName("iiwa_link_1"), {},
        geometry::Box(0.01, 0.01, 0.005), "link1_polytope",
        CoulombFriction<double>()));

    plant_->Finalize();
    diagram_ = builder.Build();
  }

  // Only allow collision between link7_polytope_id_[0], obstacles_id_[0],
  // obstacles_id_[1].
  std::vector<geometry::FilterId> ApplyFilter() {
    const auto filter_id1 =
        scene_graph_->collision_filter_manager().ApplyTransient(
            geometry::CollisionFilterDeclaration().ExcludeWithin(
                geometry::GeometrySet({link7_polytopes_id_[1],
                                       link5_polytopes_id_[0], obstacles_id_[0],
                                       obstacles_id_[1]})));
    const auto filter_id2 =
        scene_graph_->collision_filter_manager().ApplyTransient(
            geometry::CollisionFilterDeclaration().ExcludeWithin(
                geometry::GeometrySet(
                    {link7_polytopes_id_[0], link5_polytopes_id_[0]})));
    return {filter_id1, filter_id2};
  }

 protected:
  std::unique_ptr<systems::Diagram<double>> diagram_;
  MultibodyPlant<double>* plant_;
  geometry::SceneGraph<double>* scene_graph_;
  std::vector<BodyIndex> iiwa_link_;
  std::vector<geometry::GeometryId> link7_polytopes_id_;
  std::vector<geometry::GeometryId> link5_polytopes_id_;
  std::vector<geometry::GeometryId> link1_polytopes_id_;
  std::vector<geometry::GeometryId> obstacles_id_;
};

// Checks if p is an affine polynomial of x, namely p = a * x + b.
void CheckIsAffinePolynomial(
    const symbolic::Polynomial& p,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const symbolic::Variables& decision_vars) {
  EXPECT_EQ(p.TotalDegree(), 1);
  EXPECT_EQ(p.monomial_to_coefficient_map().size(), x.rows() + 1);
  for (int i = 0; i < x.rows(); ++i) {
    EXPECT_EQ(p.Degree(x(i)), 1);
  }
  for (const auto& decision_var : p.decision_variables()) {
    EXPECT_TRUE(decision_vars.find(decision_var) != decision_vars.end());
  }
}

void TestCspaceFreeRegionConstructor(
    const systems::Diagram<double>& diagram,
    const multibody::MultibodyPlant<double>* plant,
    const geometry::SceneGraph<double>* scene_graph,
    SeparatingPlaneOrder plane_order, CspaceRegionType cspace_region_type) {
  const CspaceFreeRegion dut(diagram, plant, scene_graph, plane_order,
                             cspace_region_type);
  const auto& model_inspector = scene_graph->model_inspector();
  const auto collision_pairs = model_inspector.GetCollisionCandidates();
  EXPECT_EQ(dut.separating_planes().size(), collision_pairs.size());
  EXPECT_EQ(dut.map_polytopes_to_separating_planes().size(),
            collision_pairs.size());
  // Check that each pair of geometry show up in
  // map_polytopes_to_separating_planes()
  for (const auto& collision_pair : collision_pairs) {
    auto it = dut.map_polytopes_to_separating_planes().find(
        SortedPair<geometry::GeometryId>(collision_pair.first,
                                         collision_pair.second));
    EXPECT_NE(it, dut.map_polytopes_to_separating_planes().end());
    const SeparatingPlane<symbolic::Variable>& separating_plane =
        dut.separating_planes()[it->second];
    EXPECT_EQ(it->first,
              SortedPair<geometry::GeometryId>(
                  separating_plane.positive_side_polytope->get_id(),
                  separating_plane.negative_side_polytope->get_id()));
    const auto& a = separating_plane.a;
    const auto& b = separating_plane.b;
    const symbolic::Variables t_vars(dut.rational_forward_kinematics().t());
    if (plane_order == SeparatingPlaneOrder::kConstant) {
      for (int i = 0; i < 3; ++i) {
        const symbolic::Polynomial a_poly(a(i), t_vars);
        EXPECT_EQ(a_poly.TotalDegree(), 0);
      }
      EXPECT_EQ(symbolic::Polynomial(b, t_vars).TotalDegree(), 0);
    } else if (plane_order == SeparatingPlaneOrder::kAffine) {
      VectorX<symbolic::Variable> t_for_plane;
      if (cspace_region_type == CspaceRegionType::kGenericPolytope) {
        t_for_plane = dut.rational_forward_kinematics().t();
      } else {
        t_for_plane = dut.rational_forward_kinematics().FindTOnPath(
            separating_plane.positive_side_polytope->body_index(),
            separating_plane.negative_side_polytope->body_index());
      }
      // Check if a, b are affine function of t_for_plane.
      const symbolic::Variables decision_vars(
          separating_plane.decision_variables);
      EXPECT_EQ(decision_vars.size(), 4 * t_for_plane.rows() + 4);
      CheckIsAffinePolynomial(symbolic::Polynomial(b, t_vars), t_for_plane,
                              decision_vars);
      for (int i = 0; i < 3; ++i) {
        CheckIsAffinePolynomial(symbolic::Polynomial(a(i), t_vars), t_for_plane,
                                decision_vars);
      }
    }
  }
}

TEST_F(IiwaCspaceTest, TestConstructor) {
  TestCspaceFreeRegionConstructor(*diagram_, plant_, scene_graph_,
                                  SeparatingPlaneOrder::kConstant,
                                  CspaceRegionType::kGenericPolytope);
  // Add some collision filters.
  const auto filter_ids = ApplyFilter();
  TestCspaceFreeRegionConstructor(*diagram_, plant_, scene_graph_,
                                  SeparatingPlaneOrder::kConstant,
                                  CspaceRegionType::kGenericPolytope);
  for (const auto filter_id : filter_ids) {
    scene_graph_->collision_filter_manager().RemoveDeclaration(filter_id);
  }
  // Test with as axis-aligned bounding box
  TestCspaceFreeRegionConstructor(*diagram_, plant_, scene_graph_,
                                  SeparatingPlaneOrder::kConstant,
                                  CspaceRegionType::kAxisAlignedBoundingBox);
}

void TestGenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const SeparatingPlane<symbolic::Variable>& separating_plane,
    PlaneSide plane_side, const Eigen::Ref<const Eigen::VectorXd>& q_star) {
  const ConvexPolytope* link_polytope;
  const ConvexPolytope* other_side_polytope;
  if (plane_side == PlaneSide::kPositive) {
    link_polytope = separating_plane.positive_side_polytope;
    other_side_polytope = separating_plane.negative_side_polytope;
  } else {
    link_polytope = separating_plane.negative_side_polytope;
    other_side_polytope = separating_plane.positive_side_polytope;
  }
  const auto X_AB_multilinear =
      rational_forward_kinematics.CalcLinkPoseAsMultilinearPolynomial(
          q_star, link_polytope->body_index(), separating_plane.expressed_link);

  const auto rationals = GenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link_polytope, other_side_polytope,
      X_AB_multilinear, separating_plane.a, separating_plane.b, plane_side,
      separating_plane.order);
  EXPECT_EQ(rationals.size(), link_polytope->p_BV().cols());
  for (const auto& rational : rationals) {
    EXPECT_EQ(rational.link_polytope->get_id(), link_polytope->get_id());
    EXPECT_EQ(rational.other_side_link_polytope->get_id(),
              other_side_polytope->get_id());
  }
  // Now take many samples of q, evaluate a.dot(x) + b - 1 or -1 - a.dot(x) - b
  // for these sampled q.
  std::vector<Eigen::VectorXd> q_samples;
  q_samples.push_back(
      q_star +
      (Eigen::VectorXd(7) << 0.1, 0.2, -0.1, -0.3, 1.2, 0.5, 0.1).finished());
  q_samples.push_back(
      q_star +
      (Eigen::VectorXd(7) << 0.3, -0.4, -0.8, -0.3, 1.1, -0.5, 0.4).finished());
  q_samples.push_back(
      q_star + (Eigen::VectorXd(7) << -0.3, -0.7, -1.2, -0.9, 1.3, -0.7, 0.3)
                   .finished());
  symbolic::Environment env;
  // Set the plane decision variables to arbitrary values.
  const Eigen::VectorXd plane_decision_var_vals = Eigen::VectorXd::LinSpaced(
      separating_plane.decision_variables.rows(), -2, 3);
  env.insert(separating_plane.decision_variables, plane_decision_var_vals);
  const auto& plant = rational_forward_kinematics.plant();
  auto context = plant.CreateDefaultContext();
  for (const auto& q : q_samples) {
    plant.SetPositions(context.get(), q);
    Eigen::Matrix3Xd p_AV(3, link_polytope->p_BV().cols());
    plant.CalcPointsPositions(
        *context, plant.get_body(link_polytope->body_index()).body_frame(),
        link_polytope->p_BV(),
        plant.get_body(separating_plane.expressed_link).body_frame(), &p_AV);
    const Eigen::VectorXd t_val = ((q - q_star) / 2).array().tan();
    for (int i = 0; i < t_val.rows(); ++i) {
      auto it = env.find(rational_forward_kinematics.t()(i));
      if (it == env.end()) {
        env.insert(rational_forward_kinematics.t()(i), t_val(i));
      } else {
        it->second = t_val(i);
      }
    }

    for (int i = 0; i < static_cast<int>(rationals.size()); ++i) {
      const double rational_val = rationals[i].rational.Evaluate(env);
      // Now evaluate this rational function.
      Eigen::Vector3d a_val;
      for (int j = 0; j < 3; ++j) {
        a_val(j) = separating_plane.a(j).Evaluate(env);
      }
      const double b_val = separating_plane.b.Evaluate(env);
      const double rational_val_expected =
          plane_side == PlaneSide::kPositive
              ? a_val.dot(p_AV.col(i)) + b_val - 1
              : -1 - a_val.dot(p_AV.col(i)) - b_val;
      EXPECT_NEAR(rational_val, rational_val_expected, 1E-12);
    }
  }
}

TEST_F(IiwaCspaceTest, GenerateLinkOnOneSideOfPlaneRationalFunction1) {
  scene_graph_->collision_filter_manager().ApplyTransient(
      geometry::CollisionFilterDeclaration().AllowWithin(
          geometry::GeometrySet({link7_polytopes_id_[0], obstacles_id_[0]})));
  const CspaceFreeRegion dut(*diagram_, plant_, scene_graph_,
                             SeparatingPlaneOrder::kAffine,
                             CspaceRegionType::kGenericPolytope);
  const Eigen::VectorXd q_star1 = Eigen::VectorXd::Zero(7);
  const Eigen::VectorXd q_star2 =
      (Eigen::VectorXd(7) << 0.1, 0.2, -0.1, 0.3, 0.2, 0.4, 0.2).finished();

  const auto& separating_plane = dut.separating_planes()[0];
  for (const auto plane_side : {PlaneSide::kPositive, PlaneSide::kNegative}) {
    TestGenerateLinkOnOneSideOfPlaneRationalFunction(
        dut.rational_forward_kinematics(), separating_plane, plane_side,
        q_star1);
    TestGenerateLinkOnOneSideOfPlaneRationalFunction(
        dut.rational_forward_kinematics(), separating_plane, plane_side,
        q_star2);
  }
}

void TestGenerateLinkOnOneSideOfPlaneRationals(
    const CspaceFreeRegion& dut,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  const auto rationals = dut.GenerateLinkOnOneSideOfPlaneRationals(
      q_star, filtered_collision_pairs);
  // Check the size of rationals.
  int rationals_size = 0;
  for (const auto& [link_pair, separating_plane_index] :
       dut.map_polytopes_to_separating_planes()) {
    if (!IsGeometryPairCollisionIgnored(link_pair.first(), link_pair.second(),
                                        filtered_collision_pairs)) {
      rationals_size += dut.separating_planes()[separating_plane_index]
                            .positive_side_polytope->p_BV()
                            .cols() +
                        dut.separating_planes()[separating_plane_index]
                            .negative_side_polytope->p_BV()
                            .cols();
    }
  }
  EXPECT_EQ(rationals.size(), rationals_size);
}

TEST_F(IiwaCspaceTest, GenerateLinkOnOneSideOfPlaneRationals) {
  const geometry::FilterId filter_id =
      scene_graph_->collision_filter_manager().ApplyTransient(
          geometry::CollisionFilterDeclaration().ExcludeWithin(
              geometry::GeometrySet({link7_polytopes_id_[1],
                                     link5_polytopes_id_[0], obstacles_id_[0],
                                     obstacles_id_[1]})));
  const CspaceFreeRegion dut1(*diagram_, plant_, scene_graph_,
                              SeparatingPlaneOrder::kAffine,
                              CspaceRegionType::kGenericPolytope);
  const Eigen::VectorXd q_star = Eigen::VectorXd::Zero(7);
  TestGenerateLinkOnOneSideOfPlaneRationals(dut1, q_star, {});

  // Multiple pairs of polytopes.
  scene_graph_->collision_filter_manager().RemoveDeclaration(filter_id);
  const CspaceFreeRegion dut2(*diagram_, plant_, scene_graph_,
                              SeparatingPlaneOrder::kAffine,
                              CspaceRegionType::kGenericPolytope);
  TestGenerateLinkOnOneSideOfPlaneRationals(dut2, q_star, {});
  // Now test with filtered collision pairs.
  const CspaceFreeRegion::FilteredCollisionPairs filtered_collision_pairs{
      {{link7_polytopes_id_[0], obstacles_id_[0]}}};
  TestGenerateLinkOnOneSideOfPlaneRationals(dut2, q_star,
                                            filtered_collision_pairs);
}

// Check p has degree at most 2 for each variable in t.
void CheckPolynomialDegree2(const symbolic::Polynomial& p,
                            const symbolic::Variables& t) {
  for (const auto& var : t) {
    EXPECT_LE(p.Degree(var), 2);
  }
}

void ConstructInitialCspacePolytope(const CspaceFreeRegion& dut,
                                    Eigen::VectorXd* q_star, Eigen::MatrixXd* C,
                                    Eigen::VectorXd* d,
                                    Eigen::VectorXd* q_not_in_collision) {
  const auto& plant = dut.rational_forward_kinematics().plant();
  auto context = plant.CreateDefaultContext();
  *q_star = Eigen::VectorXd::Zero(7);

  // I will build a small C-space polytope C*t<=d around q_not_in_collision;
  *q_not_in_collision =
      (Eigen::VectorXd(7) << 0.5, 0.3, -0.3, 0.1, 0.4, 0.2, 0.1).finished();
  plant.SetPositions(context.get(), *q_not_in_collision);
  ASSERT_FALSE(dut.IsPostureInCollision(*context));

  // First generate a region C * t <= d.
  C->resize(24, 7);
  d->resize(24);
  // I create matrix C with arbitrary values, such that C * t is a small
  // polytope surrounding q_not_in_collision.
  // clang-format off
  *C << 1, 0, 0, 0, 2, 0, 0,
       -1, 0, 0, 0, 0, 1, 0,
       0, 1, 1, 0, 0, 0, 1,
       0, -1, -2, 0, 0, -1, 0,
       1, 1, 0, 2, 0, 0, 1,
       1, 0, 2, -1, 0, 1, 0,
       0, -1, 2, -2, 1, 3, 2,
       0, 1, -2, 1, 2, 4, 3,
       0, 3, -2, 2, 0, 1, -1,
       1, 0, 3, 2, 0, -1, 1,
       0, 1, -1, -2, 3, -2, 1,
       1, 0, -1, 1, 3, 2, 0,
       -1, -0.1, -0.2, 0, 0.3, 0.1, 0.1,
       -2, 0.1, 0.2, 0.2, -0.3, -0.1, 0.1,
       -1, 1, 1, 0, -1, 1, 0,
       0, 0.2, 0.1, 0, -1, 0.1, 0,
       0.1, 2, 0.2, 0.1, -0.1, -0.2, 0.1,
       -0.1, -2, 0.1, 0.2, -0.15, -0.1, -0.1,
       0.3, 0.5, 0.1, 0.7, -0.4, 1.2, 3.1,
       -0.5, 0.3, 0.2, -0.5, 1.2, 0.7, -0.5,
       0.4, 0.6, 1.2, -0.3, -0.5, 1.2, -0.1,
       1.5, -0.1, 0.6, 1.5, 0.4, 2.1, 0.3,
       0.5, 1.5, 0.3, 0.2, 1.5, -0.1, 0.5,
       0.5, 0.2, -0.1, 1.2, -0.3, 1.1, -0.4;
  // clang-format on

  // Now I normalize each row of C. Because later when we search for the
  // polytope we have the constraint that |C.row()|<=1, so it is better to start
  // with a C satisfying this constraint.
  for (int i = 0; i < C->rows(); ++i) {
    C->row(i).normalize();
  }
  // Now I take some samples of t slightly away from q_not_in_collision. C * t
  // <= d contains all these samples.
  Eigen::Matrix<double, 7, 6> t_samples;
  t_samples.col(0) = ((*q_not_in_collision - *q_star) / 2).array().tan();
  t_samples.col(1) =
      t_samples.col(0) +
      (Eigen::VectorXd(7) << 0.11, -0.02, 0.03, 0.01, 0, 0.02, 0.02).finished();
  t_samples.col(2) = t_samples.col(0) + (Eigen::VectorXd(7) << -0.005, 0.01,
                                         -0.02, 0.01, 0.005, 0.01, -0.02)
                                            .finished();
  t_samples.col(3) = t_samples.col(0) + (Eigen::VectorXd(7) << 0.02, -0.13,
                                         0.01, 0.02, -0.03, 0.01, 0.15)
                                            .finished();
  t_samples.col(4) = t_samples.col(0) + (Eigen::VectorXd(7) << 0.01, -0.04,
                                         0.003, 0.01, -0.01, -0.11, -0.08)
                                            .finished();
  t_samples.col(5) = t_samples.col(0) + (Eigen::VectorXd(7) << -0.01, -0.02,
                                         0.013, -0.02, 0.03, -0.03, -0.1)
                                            .finished();
  *d = ((*C) * t_samples).rowwise().maxCoeff();
}

TEST_F(IiwaCspaceTest, ConstructProgramForCspacePolytope) {
  scene_graph_->collision_filter_manager().Apply(
      geometry::CollisionFilterDeclaration().ExcludeWithin(
          geometry::GeometrySet({link7_polytopes_id_[1], link5_polytopes_id_[0],
                                 obstacles_id_[0], obstacles_id_[1]})));
  const CspaceFreeRegion dut(*diagram_, plant_, scene_graph_,

                             SeparatingPlaneOrder::kAffine,
                             CspaceRegionType::kGenericPolytope);
  const auto& plant = dut.rational_forward_kinematics().plant();
  auto context = plant.CreateDefaultContext();

  Eigen::VectorXd q_star;
  Eigen::MatrixXd C;
  Eigen::VectorXd d;
  Eigen::VectorXd q_not_in_collision;
  ConstructInitialCspacePolytope(dut, &q_star, &C, &d, &q_not_in_collision);

  CspaceFreeRegion::FilteredCollisionPairs filtered_collision_pairs{};
  auto clock_start = std::chrono::system_clock::now();
  const auto rationals = dut.GenerateLinkOnOneSideOfPlaneRationals(
      q_star, filtered_collision_pairs);
  auto ret = dut.ConstructProgramForCspacePolytope(q_star, rationals, C, d,
                                                   filtered_collision_pairs);
  auto clock_now = std::chrono::system_clock::now();
  std::cout << "Elapsed Time: "
            << static_cast<float>(
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       clock_now - clock_start)
                       .count()) /
                   1000
            << "s\n";
  // First make sure that the lagrangians and verified polynomial has the right
  // size.
  EXPECT_EQ(ret.polytope_lagrangians.size(), rationals.size());
  EXPECT_EQ(ret.t_lower_lagrangians.size(), rationals.size());
  EXPECT_EQ(ret.t_upper_lagrangians.size(), rationals.size());
  EXPECT_EQ(ret.verified_polynomials.size(), rationals.size());
  const auto& t = dut.rational_forward_kinematics().t();
  const symbolic::Variables t_variables{t};
  for (int i = 0; i < static_cast<int>(rationals.size()); ++i) {
    EXPECT_EQ(ret.polytope_lagrangians[i].rows(), C.rows());
    EXPECT_EQ(ret.t_lower_lagrangians[i].rows(), t.rows());
    EXPECT_EQ(ret.t_upper_lagrangians[i].rows(), t.rows());

    for (int j = 0; j < ret.polytope_lagrangians[i].rows(); ++j) {
      CheckPolynomialDegree2(ret.polytope_lagrangians[i](j), t_variables);
    }
    for (int j = 0; j < t.rows(); ++j) {
      CheckPolynomialDegree2(ret.t_lower_lagrangians[i](j), t_variables);
      CheckPolynomialDegree2(ret.t_upper_lagrangians[i](j), t_variables);
    }
  }
  // Make sure that each term in verified_polynomial has at most degree 3 for
  // each t, and at most one t has degree 3.
  for (const auto& verified_poly : ret.verified_polynomials) {
    for (const auto& [monomial, coeff] :
         verified_poly.monomial_to_coefficient_map()) {
      int degree_3_count = 0;
      for (int i = 0; i < dut.rational_forward_kinematics().t().rows(); ++i) {
        const int t_degree =
            monomial.degree(dut.rational_forward_kinematics().t()(i));
        EXPECT_LE(t_degree, 3);
        if (t_degree == 3) {
          degree_3_count++;
        }
      }
      EXPECT_LE(degree_3_count, 1);
    }
  }
  // TODO(hongkai.dai): test that t_lower_lagrangians and t_upper_lagrangians
  // are 0 since the bounds from the joint limits are redundant for this C * t
  // <= d.
  // Now check if ret.verified_polynomials is correct
  VectorX<symbolic::Polynomial> d_minus_Ct(d.rows());
  for (int i = 0; i < d_minus_Ct.rows(); ++i) {
    d_minus_Ct(i) = symbolic::Polynomial(
        d(i) - C.row(i).dot(dut.rational_forward_kinematics().t()),
        t_variables);
  }
  VectorX<symbolic::Polynomial> t_minus_t_lower(t.rows());
  VectorX<symbolic::Polynomial> t_upper_minus_t(t.rows());
  Eigen::VectorXd t_lower, t_upper;
  ComputeBoundsOnT(q_star, plant.GetPositionLowerLimits(),
                   plant.GetPositionUpperLimits(), &t_lower, &t_upper);
  for (int i = 0; i < t.rows(); ++i) {
    t_minus_t_lower(i) = symbolic::Polynomial(t(i) - t_lower(i), t_variables);
    t_upper_minus_t(i) = symbolic::Polynomial(t_upper(i) - t(i), t_variables);
  }
  for (int i = 0; i < static_cast<int>(rationals.size()); ++i) {
    symbolic::Polynomial eval_expected = rationals[i].rational.numerator();
    for (int j = 0; j < C.rows(); ++j) {
      eval_expected -= ret.polytope_lagrangians[i](j) * d_minus_Ct(j);
    }
    for (int j = 0; j < t.rows(); ++j) {
      eval_expected -= ret.t_lower_lagrangians[i](j) * t_minus_t_lower(j) +
                       ret.t_upper_lagrangians[i](j) * t_upper_minus_t(j);
    }
    const symbolic::Polynomial eval = ret.verified_polynomials[i];
    EXPECT_TRUE(eval.CoefficientsAlmostEqual(eval_expected, 1E-10));
  }
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(*(ret.prog), std::nullopt, solver_options);
  EXPECT_TRUE(result.is_success());
}

void CheckReadAndWriteCspacePolytope(const CspaceFreeRegion& dut,
                                     const CspaceFreeRegionSolution& solution) {
  const std::string file_name = temp_directory() + "/cspace_polytope.txt";
  WriteCspacePolytopeToFile(solution, dut.rational_forward_kinematics().plant(),
                            dut.scene_graph().model_inspector(), file_name, 10);
  Eigen::MatrixXd C;
  Eigen::VectorXd d;
  std::unordered_map<
      SortedPair<std::pair<BodyIndex, geometry::GeometryId>>,
      std::tuple<BodyIndex, Eigen::VectorXd, SeparatingPlaneOrder>>
      separating_planes;
  ReadCspacePolytopeFromFile(
      file_name, dut.rational_forward_kinematics().plant(),
      dut.scene_graph().model_inspector(), &C, &d, &separating_planes);
  const double tol = 1E-7;
  EXPECT_TRUE(CompareMatrices(solution.C, C, tol));
  EXPECT_TRUE(CompareMatrices(solution.d, d, tol));
  EXPECT_EQ(solution.separating_planes.size(), separating_planes.size());
  for (const auto& plane : solution.separating_planes) {
    auto it = separating_planes.find(
        SortedPair<std::pair<BodyIndex, geometry::GeometryId>>(
            std::pair(plane.positive_side_polytope->body_index(),
                      plane.positive_side_polytope->get_id()),
            std::pair(plane.negative_side_polytope->body_index(),
                      plane.negative_side_polytope->get_id())));
    EXPECT_NE(it, separating_planes.end());
    EXPECT_EQ(std::get<0>(it->second), plane.expressed_link);
    EXPECT_TRUE(CompareMatrices(std::get<1>(it->second),
                                plane.decision_variables, tol));
    EXPECT_TRUE(std::get<2>(it->second) == plane.order);
  }
}

void CheckReadAndWriteCspacePolytope2(
    const CspaceFreeRegion& dut, const CspaceFreeRegionSolution& solution) {
  const std::string file_name = temp_directory() + "/cspace_polytope2.txt";
  WriteCspacePolytopeToFile(solution, dut.rational_forward_kinematics().plant(),
                            dut.scene_graph().model_inspector(), file_name, 10);
  CspaceFreeRegionSolution solution_read_from_file;
  ReadCspacePolytopeFromFile(file_name, dut, &solution_read_from_file);
  const double tol = 1E-7;
  EXPECT_TRUE(CompareMatrices(solution.C, solution_read_from_file.C, tol));
  EXPECT_TRUE(CompareMatrices(solution.d, solution_read_from_file.d, tol));
  EXPECT_EQ(solution.separating_planes.size(),
            solution_read_from_file.separating_planes.size());

  auto check_separating_planes_equal = [tol](SeparatingPlane<double> p1,
                                             SeparatingPlane<double> p2) {
    bool same_plane{false};
    // check polytopes are the same
    same_plane = same_plane && (p1.positive_side_polytope->body_index() ==
                                p2.positive_side_polytope->body_index());
    same_plane = same_plane && (p1.positive_side_polytope->get_id() ==
                                p2.positive_side_polytope->get_id());
    same_plane =
        same_plane && (CompareMatrices(p1.positive_side_polytope->p_BV(),
                                       p2.positive_side_polytope->p_BV(), tol));
    same_plane = same_plane && (p1.negative_side_polytope->body_index() ==
                                p2.negative_side_polytope->body_index());
    same_plane = same_plane && (p1.negative_side_polytope->get_id() ==
                                p2.negative_side_polytope->get_id());
    same_plane =
        same_plane && (CompareMatrices(p1.negative_side_polytope->p_BV(),
                                       p2.negative_side_polytope->p_BV(), tol));
    // check plane orders are the same
    same_plane = same_plane && (p1.order == p2.order);
    // check planes are the same
    same_plane = same_plane && (CompareMatrices(p1.decision_variables,
                                                p2.decision_variables, tol));
    for (int i = 0; i < p1.a.rows(); ++i) {
      same_plane = same_plane && (p1.a(i).Expand().EqualTo(p2.a(i).Expand()));
    }
    same_plane = same_plane && (p1.b.Expand().EqualTo(p2.b.Expand()));
    return same_plane;
  };

  for (const auto& p1 : solution.separating_planes) {
    bool same_plane_found = false;
    for (const auto& p2 : solution_read_from_file.separating_planes) {
      bool same_plane = check_separating_planes_equal(p1, p2);
      if (same_plane) {
        same_plane_found = true;
        break;
      }
    }
    EXPECT_TRUE(same_plane_found);
  }
}

TEST_F(IiwaCspaceTest, GenerateTuplesForBilinearAlternation) {
  const CspaceFreeRegion dut(*diagram_, plant_, scene_graph_,
                             SeparatingPlaneOrder::kAffine,
                             CspaceRegionType::kGenericPolytope);
  const Eigen::VectorXd q_star = Eigen::VectorXd::Zero(7);
  const int C_rows = 5;
  std::vector<CspaceFreeRegion::CspacePolytopeTuple> alternation_tuples;
  VectorX<symbolic::Polynomial> d_minus_Ct;
  Eigen::VectorXd t_lower, t_upper;
  VectorX<symbolic::Polynomial> t_minus_t_lower, t_upper_minus_t;
  MatrixX<symbolic::Variable> C_var;
  VectorX<symbolic::Variable> d_var, lagrangian_gram_vars, verified_gram_vars,
      separating_plane_vars;
  std::vector<std::vector<int>> separating_plane_to_tuples;
  dut.GenerateTuplesForBilinearAlternation(
      q_star, {}, C_rows, &alternation_tuples, &d_minus_Ct, &t_lower, &t_upper,
      &t_minus_t_lower, &t_upper_minus_t, &C_var, &d_var, &lagrangian_gram_vars,
      &verified_gram_vars, &separating_plane_vars, &separating_plane_to_tuples);
  int rational_count = 0;
  for (const auto& separating_plane : dut.separating_planes()) {
    rational_count += separating_plane.positive_side_polytope->p_BV().cols() +
                      separating_plane.negative_side_polytope->p_BV().cols();
  }
  EXPECT_EQ(alternation_tuples.size(), rational_count);
  // Now count the total number of lagrangian gram vars.
  int lagrangian_gram_vars_count = 0;
  int verified_gram_vars_count = 0;
  std::unordered_set<int> lagrangian_gram_vars_start;
  std::unordered_set<int> verified_gram_vars_start;
  for (const auto& tuple : alternation_tuples) {
    const int gram_rows = tuple.monomial_basis.rows();
    const int gram_lower_size = gram_rows * (gram_rows + 1) / 2;
    lagrangian_gram_vars_count +=
        gram_lower_size *
        (C_rows + 2 * dut.rational_forward_kinematics().t().rows());
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
  Eigen::VectorXd t_lower_expected, t_upper_expected;
  const auto& plant = dut.rational_forward_kinematics().plant();
  ComputeBoundsOnT(q_star, plant.GetPositionLowerLimits(),
                   plant.GetPositionUpperLimits(), &t_lower_expected,
                   &t_upper_expected);
  EXPECT_TRUE(CompareMatrices(t_lower, t_lower_expected));
  EXPECT_TRUE(CompareMatrices(t_upper, t_upper_expected));
  const auto& t = dut.rational_forward_kinematics().t();
  for (int i = 0; i < t.rows(); ++i) {
    EXPECT_TRUE(
        t_minus_t_lower(i).EqualTo(symbolic::Polynomial(t(i) - t_lower(i))));
    EXPECT_TRUE(
        t_upper_minus_t(i).EqualTo(symbolic::Polynomial(t_upper(i) - t(i))));
  }
  EXPECT_EQ(lagrangian_gram_vars.rows(), lagrangian_gram_vars_count);
  EXPECT_EQ(verified_gram_vars.rows(), verified_gram_vars_count);
  EXPECT_EQ(verified_gram_vars_start.size(), alternation_tuples.size());
  EXPECT_EQ(lagrangian_gram_vars_start.size(),
            alternation_tuples.size() *
                (C_rows + 2 * dut.rational_forward_kinematics().t().rows()));
  int separating_plane_vars_count = 0;
  for (const auto& separating_plane : dut.separating_planes()) {
    separating_plane_vars_count += separating_plane.decision_variables.rows();
  }
  EXPECT_EQ(separating_plane_vars.rows(), separating_plane_vars_count);
  const symbolic::Variables separating_plane_vars_set{separating_plane_vars};
  EXPECT_EQ(separating_plane_vars_set.size(), separating_plane_vars_count);
  // Now check separating_plane_to_tuples
  EXPECT_EQ(separating_plane_to_tuples.size(), dut.separating_planes().size());
  std::unordered_set<int> tuple_indices_set;
  for (const auto& tuple_indices : separating_plane_to_tuples) {
    for (int index : tuple_indices) {
      EXPECT_EQ(tuple_indices_set.count(index), 0);
      tuple_indices_set.emplace(index);
      EXPECT_LT(index, rational_count);
      EXPECT_GE(index, 0);
    }
  }
  EXPECT_EQ(tuple_indices_set.size(), rational_count);
}

void CheckPsd(const Eigen::Ref<const Eigen::MatrixXd>& mat, double tol) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(mat);
  ASSERT_EQ(es.info(), Eigen::Success);
  EXPECT_TRUE((es.eigenvalues().array() > -tol).all());
}

void TestLagrangianResult(
    const CspaceFreeRegion& dut,
    const std::vector<CspaceFreeRegion::CspacePolytopeTuple>&
        alternation_tuples,
    const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
    const VectorX<symbolic::Variable>& separating_plane_vars,
    const VectorX<symbolic::Polynomial>& t_minus_t_lower,
    const VectorX<symbolic::Polynomial>& t_upper_minus_t,
    const Eigen::VectorXd& lagrangian_gram_var_vals,
    const Eigen::VectorXd& verified_gram_var_vals,
    const Eigen::VectorXd& separating_plane_var_vals, double tol) {
  // Check if the solution satisfies the PSD constraint, and the polynomials
  // match.
  symbolic::Environment env;
  env.insert(separating_plane_vars, separating_plane_var_vals);
  VectorX<symbolic::Polynomial> d_minus_Ct_poly(C.rows());
  const auto& t = dut.rational_forward_kinematics().t();
  for (int i = 0; i < C.rows(); ++i) {
    d_minus_Ct_poly(i) = symbolic::Polynomial(d(i) - C.row(i).dot(t));
  }

  // Now check if each Gram matrix is PSD.
  for (const auto& tuple : alternation_tuples) {
    symbolic::Polynomial verified_polynomial =
        tuple.rational_numerator.EvaluatePartial(env);
    const int gram_rows = tuple.monomial_basis.rows();
    const int gram_lower_size = gram_rows * (gram_rows + 1) / 2;
    Eigen::MatrixXd gram;
    SymmetricMatrixFromLower<double>(
        gram_rows,
        verified_gram_var_vals.segment(
            tuple.verified_polynomial_gram_lower_start, gram_lower_size),
        &gram);
    const double psd_tol = 1E-6;
    CheckPsd(gram, psd_tol);
    const symbolic::Polynomial verified_polynomial_expected =
        CalcPolynomialFromGram<double>(tuple.monomial_basis, gram);
    for (int i = 0; i < C.rows(); ++i) {
      SymmetricMatrixFromLower<double>(
          gram_rows,
          lagrangian_gram_var_vals.segment(
              tuple.polytope_lagrangian_gram_lower_start[i], gram_lower_size),
          &gram);
      CheckPsd(gram, psd_tol);
      verified_polynomial -=
          CalcPolynomialFromGram<double>(tuple.monomial_basis, gram) *
          d_minus_Ct_poly(i);
    }
    for (int i = 0; i < t.rows(); ++i) {
      SymmetricMatrixFromLower<double>(
          gram_rows,
          lagrangian_gram_var_vals.segment(
              tuple.t_lower_lagrangian_gram_lower_start[i], gram_lower_size),
          &gram);
      CheckPsd(gram, psd_tol);
      verified_polynomial -=
          CalcPolynomialFromGram<double>(tuple.monomial_basis, gram) *
          t_minus_t_lower(i);

      SymmetricMatrixFromLower<double>(
          gram_rows,
          lagrangian_gram_var_vals.segment(
              tuple.t_upper_lagrangian_gram_lower_start[i], gram_lower_size),
          &gram);
      CheckPsd(gram, psd_tol);
      verified_polynomial -=
          CalcPolynomialFromGram<double>(tuple.monomial_basis, gram) *
          t_upper_minus_t(i);
    }
    EXPECT_TRUE(verified_polynomial.CoefficientsAlmostEqual(
        verified_polynomial_expected, tol));
  }
}

TEST_F(IiwaCspaceTest, ConstructLagrangianAndPolytopeProgram) {
  // Test both ConstructLagrangianProgram and ConstructPolytopeProgram (the
  // latter needs the result from the former).
  ApplyFilter();
  const CspaceFreeRegion dut(*diagram_, plant_, scene_graph_,
                             SeparatingPlaneOrder::kAffine,
                             CspaceRegionType::kGenericPolytope);
  const auto& plant = dut.rational_forward_kinematics().plant();
  auto context = plant.CreateDefaultContext();

  Eigen::VectorXd q_star;
  Eigen::MatrixXd C;
  Eigen::VectorXd d;
  Eigen::VectorXd q_not_in_collision;
  ConstructInitialCspacePolytope(dut, &q_star, &C, &d, &q_not_in_collision);

  CspaceFreeRegion::FilteredCollisionPairs filtered_collision_pairs{};
  std::vector<CspaceFreeRegion::CspacePolytopeTuple> alternation_tuples;
  VectorX<symbolic::Polynomial> d_minus_Ct;
  Eigen::VectorXd t_lower, t_upper;
  VectorX<symbolic::Polynomial> t_minus_t_lower, t_upper_minus_t;
  MatrixX<symbolic::Variable> C_var;
  VectorX<symbolic::Variable> d_var, lagrangian_gram_vars, verified_gram_vars,
      separating_plane_vars;
  std::vector<std::vector<int>> separating_plane_to_tuples;
  dut.GenerateTuplesForBilinearAlternation(
      q_star, filtered_collision_pairs, C.rows(), &alternation_tuples,
      &d_minus_Ct, &t_lower, &t_upper, &t_minus_t_lower, &t_upper_minus_t,
      &C_var, &d_var, &lagrangian_gram_vars, &verified_gram_vars,
      &separating_plane_vars, &separating_plane_to_tuples);

  MatrixX<symbolic::Variable> P;
  VectorX<symbolic::Variable> q;
  auto clock_start = std::chrono::system_clock::now();
  double redundant_tighten = 0;
  auto prog = dut.ConstructLagrangianProgram(
      alternation_tuples, C, d, lagrangian_gram_vars, verified_gram_vars,
      separating_plane_vars, t_lower, t_upper, {}, redundant_tighten, &P, &q);
  auto clock_finish = std::chrono::system_clock::now();
  std::cout << "ConstructLagrangianProgram takes "
            << static_cast<float>(
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       clock_finish - clock_start)
                       .count()) /
                   1000
            << "s\n";
  prog->AddMaximizeLogDeterminantCost(P.cast<symbolic::Expression>());
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
  EXPECT_TRUE(result.is_success());

  // Now check the result of finding lagrangians.
  const double psd_tol = 1E-6;
  const auto P_sol = result.GetSolution(P);
  CheckPsd(P_sol, psd_tol);
  const auto q_sol = result.GetSolution(q);

  const Eigen::VectorXd lagrangian_gram_var_vals =
      result.GetSolution(lagrangian_gram_vars);
  Eigen::VectorXd verified_gram_var_vals =
      result.GetSolution(verified_gram_vars);
  const Eigen::VectorXd separating_plane_var_vals =
      result.GetSolution(separating_plane_vars);
  TestLagrangianResult(dut, alternation_tuples, C, d, separating_plane_vars,
                       t_minus_t_lower, t_upper_minus_t,
                       lagrangian_gram_var_vals, verified_gram_var_vals,
                       separating_plane_var_vals, 1E-5);
  const std::vector<bool> is_plane_active = internal::IsPlaneActive(
      dut.separating_planes(), filtered_collision_pairs);
  const std::vector<SeparatingPlane<double>> separating_planes_sol =
      internal::GetSeparatingPlanesSolution(dut, is_plane_active, result);
  CspaceFreeRegionSolution cspace_free_region_solution(C, d, P_sol, q_sol,
                                                       separating_planes_sol);
  CheckReadAndWriteCspacePolytope(dut, cspace_free_region_solution);
  CheckReadAndWriteCspacePolytope2(dut, cspace_free_region_solution);

  // Now test ConstructPolytopeProgram using the lagrangian result.
  VectorX<symbolic::Variable> margin;
  clock_start = std::chrono::system_clock::now();
  auto prog_polytope = dut.ConstructPolytopeProgram(
      alternation_tuples, C_var, d_var, d_minus_Ct, lagrangian_gram_var_vals,
      verified_gram_vars, separating_plane_vars, t_minus_t_lower,
      t_upper_minus_t, {});
  margin = prog_polytope->NewContinuousVariables(C_var.rows(), "margin");
  AddOuterPolytope(prog_polytope.get(), P_sol, q_sol, C_var, d_var, margin);
  prog_polytope->AddBoundingBoxConstraint(0, kInf, margin);
  clock_finish = std::chrono::system_clock::now();
  std::cout << "ConstructPolytopeProgram takes "
            << static_cast<float>(
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       clock_finish - clock_start)
                       .count()) /
                   1000
            << "s\n";
  // Number of PSD constraint is the number of SOS constraint, equal to the
  // number of rational numerators.
  EXPECT_EQ(prog_polytope->positive_semidefinite_constraints().size() +
                prog_polytope->linear_matrix_inequality_constraints().size(),
            alternation_tuples.size());
  // Maximize the summation of margin.
  prog_polytope->AddLinearCost(-Eigen::VectorXd::Ones(margin.rows()), 0.,
                               margin);
  const auto result_polytope =
      solvers::Solve(*prog_polytope, std::nullopt, solver_options);
  EXPECT_TRUE(result_polytope.is_success());
  // Test the result.
  symbolic::Environment env_polytope;
  env_polytope.insert(separating_plane_vars,
                      result_polytope.GetSolution(separating_plane_vars));
  const auto C_sol = result_polytope.GetSolution(C_var);
  const auto d_sol = result_polytope.GetSolution(d_var);
  VectorX<symbolic::Polynomial> d_minus_Ct_sol(C.rows());
  const auto& t = dut.rational_forward_kinematics().t();
  for (int i = 0; i < C.rows(); ++i) {
    d_minus_Ct_sol(i) = symbolic::Polynomial(d_sol(i) - C_sol.row(i).dot(t));
  }
  verified_gram_var_vals = result_polytope.GetSolution(verified_gram_vars);
  for (const auto& tuple : alternation_tuples) {
    symbolic::Polynomial verified_polynomial =
        tuple.rational_numerator.EvaluatePartial(env_polytope);
    const int gram_rows = tuple.monomial_basis.rows();
    const int gram_lower_size = gram_rows * (gram_rows + 1) / 2;
    for (int i = 0; i < C.rows(); ++i) {
      verified_polynomial -=
          CalcPolynomialFromGramLower<double>(
              tuple.monomial_basis,
              lagrangian_gram_var_vals.segment(
                  tuple.polytope_lagrangian_gram_lower_start[i],
                  gram_lower_size)) *
          d_minus_Ct_sol(i);
    }
    for (int i = 0; i < t.rows(); ++i) {
      verified_polynomial -=
          CalcPolynomialFromGramLower<double>(
              tuple.monomial_basis,
              lagrangian_gram_var_vals.segment(
                  tuple.t_lower_lagrangian_gram_lower_start[i],
                  gram_lower_size)) *
          t_minus_t_lower(i);
      verified_polynomial -=
          CalcPolynomialFromGramLower<double>(
              tuple.monomial_basis,
              lagrangian_gram_var_vals.segment(
                  tuple.t_upper_lagrangian_gram_lower_start[i],
                  gram_lower_size)) *
          t_upper_minus_t(i);
    }
    Eigen::MatrixXd verified_gram;
    SymmetricMatrixFromLower<double>(
        gram_rows,
        verified_gram_var_vals.segment(
            tuple.verified_polynomial_gram_lower_start, gram_lower_size),
        &verified_gram);
    CheckPsd(verified_gram, psd_tol);
    const symbolic::Polynomial verified_polynomial_expected =
        CalcPolynomialFromGram<double>(tuple.monomial_basis, verified_gram);
    EXPECT_TRUE(verified_polynomial.CoefficientsAlmostEqual(
        verified_polynomial_expected, 4E-4));
  }
  // Make sure that the polytope C * t <= d contains the ellipsoid.
  const auto margin_sol = result_polytope.GetSolution(margin);
  EXPECT_TRUE((margin_sol.array() >= -1E-6).all());
  for (int i = 0; i < C.rows(); ++i) {
    EXPECT_LE(
        (C_sol.row(i) * P_sol).norm() + C_sol.row(i).dot(q_sol) + margin_sol(i),
        d_sol(i) + 1E-6);
  }
}

TEST_F(IiwaCspaceTest, CspacePolytopeBilinearAlternation) {
  ApplyFilter();
  const CspaceFreeRegion dut(*diagram_, plant_, scene_graph_,
                             SeparatingPlaneOrder::kAffine,
                             CspaceRegionType::kGenericPolytope);
  const auto& plant = dut.rational_forward_kinematics().plant();
  auto context = plant.CreateDefaultContext();

  Eigen::VectorXd q_star;
  Eigen::MatrixXd C;
  Eigen::VectorXd d;
  Eigen::VectorXd q_not_in_collision;
  ConstructInitialCspacePolytope(dut, &q_star, &C, &d, &q_not_in_collision);

  CspaceFreeRegion::FilteredCollisionPairs filtered_collision_pairs{};
  // Intentially multiplies a factor to make the rows of C unnormalized.
  C.row(0) = 2 * C.row(0);
  d(0) = 2 * d(0);
  C.row(1) = 3 * C.row(1);
  d(1) = 3 * d(1);

  Eigen::MatrixXd C_final;
  Eigen::VectorXd d_final;
  Eigen::MatrixXd P_final;
  Eigen::VectorXd q_final;
  const CspaceFreeRegion::BilinearAlternationOption
      bilinear_alternation_options{.max_iters = 3,
                                   .convergence_tol = 0.001,
                                   .lagrangian_backoff_scale = 0.05,
                                   .polytope_backoff_scale = 0.05,
                                   .verbose = true};
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, true);
  const Eigen::VectorXd t_inner_pts =
      dut.rational_forward_kinematics().ComputeTValue(q_not_in_collision,
                                                      q_star);
  //  std::vector<SeparatingPlane> separating_planes_sol;
  std::vector<double> polytope_volumes, ellipsoid_determinants;
  CspaceFreeRegionSolution cspace_free_region_solution;
  dut.CspacePolytopeBilinearAlternation(
      q_star, filtered_collision_pairs, C, d, bilinear_alternation_options,
      solver_options, t_inner_pts, std::nullopt, &cspace_free_region_solution,
      &polytope_volumes, &ellipsoid_determinants);
  EXPECT_EQ(cspace_free_region_solution.separating_planes.size(),
            dut.separating_planes().size());
  const symbolic::Variables t_vars(dut.rational_forward_kinematics().t());
  for (const auto& separating_plane_sol :
       cspace_free_region_solution.separating_planes) {
    // Make sure a and b only contain t as variables.
    for (int i = 0; i < 3; ++i) {
      EXPECT_TRUE(separating_plane_sol.a(i).GetVariables().IsSubsetOf(t_vars));
    }
    EXPECT_TRUE(separating_plane_sol.b.GetVariables().IsSubsetOf(t_vars));
  }
  EXPECT_TRUE(((C_final * t_inner_pts).array() <= d_final.array()).all());
}

TEST_F(IiwaCspaceTest, CspacePolytopeBinarySearch) {
  ApplyFilter();
  const CspaceFreeRegion dut(*diagram_, plant_, scene_graph_,
                             SeparatingPlaneOrder::kAffine,
                             CspaceRegionType::kGenericPolytope);
  const auto& plant = dut.rational_forward_kinematics().plant();
  auto context = plant.CreateDefaultContext();

  Eigen::VectorXd q_star;
  Eigen::MatrixXd C;
  Eigen::VectorXd d;
  Eigen::VectorXd q_not_in_collision;
  ConstructInitialCspacePolytope(dut, &q_star, &C, &d, &q_not_in_collision);

  CspaceFreeRegion::FilteredCollisionPairs filtered_collision_pairs{};
  // Intentially multiplies a factor to make the rows of C unnormalized.
  C.row(0) = 2 * C.row(0);
  d(0) = 2 * d(0);
  C.row(1) = 3 * C.row(1);
  d(1) = 3 * d(1);

  CspaceFreeRegion::BinarySearchOption binary_search_option{
      .epsilon_max = 1, .epsilon_min = 0.1, .max_iters = 4, .search_d = false};
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, true);
  Eigen::VectorXd d_final;
  CspaceFreeRegionSolution cspace_free_region_solution;
  const Eigen::VectorXd t_inner_pts =
      dut.rational_forward_kinematics().ComputeTValue(q_not_in_collision,
                                                      q_star);
  dut.CspacePolytopeBinarySearch(
      q_star, filtered_collision_pairs, C, d, binary_search_option,
      solver_options, t_inner_pts, std::nullopt, &cspace_free_region_solution);
  EXPECT_EQ(cspace_free_region_solution.separating_planes.size(),
            dut.separating_planes().size());

  // Now do binary search but also look for d.
  binary_search_option.search_d = true;
  binary_search_option.max_iters = 2;
  Eigen::VectorXd d_final_search_d;
  dut.CspacePolytopeBinarySearch(
      q_star, filtered_collision_pairs, C, d, binary_search_option,
      solver_options, t_inner_pts, std::nullopt, &cspace_free_region_solution);
  EXPECT_EQ(cspace_free_region_solution.separating_planes.size(),
            dut.separating_planes().size());
}

void CheckSeparatingPlanesSol(
    const CspaceFreeRegion& dut,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs,
    const std::vector<SeparatingPlane<double>>& separating_planes_sol) {
  const std::vector<bool> is_plane_active = internal::IsPlaneActive(
      dut.separating_planes(), filtered_collision_pairs);
  int active_plane_count = 0;
  const symbolic::Variables t_vars(dut.rational_forward_kinematics().t());
  for (int plane_index = 0;
       plane_index < static_cast<int>(dut.separating_planes().size());
       ++plane_index) {
    if (is_plane_active[plane_index]) {
      EXPECT_EQ(separating_planes_sol[active_plane_count]
                    .positive_side_polytope->get_id(),
                dut.separating_planes()[plane_index]
                    .positive_side_polytope->get_id());
      EXPECT_EQ(separating_planes_sol[active_plane_count]
                    .negative_side_polytope->get_id(),
                dut.separating_planes()[plane_index]
                    .negative_side_polytope->get_id());
      for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(separating_planes_sol[active_plane_count]
                        .a(i)
                        .GetVariables()
                        .IsSubsetOf(t_vars));
      }
      EXPECT_TRUE(
          separating_planes_sol[active_plane_count].b.GetVariables().IsSubsetOf(
              t_vars));
      active_plane_count++;
    }
  }
  EXPECT_EQ(separating_planes_sol.size(), active_plane_count);
}

TEST_F(IiwaCspaceTest, FindLagrangianAndSeparatingPlanes) {
  // Test both the single-thread version and the multiple-thread version. Make
  // sure the result are correct.
  ApplyFilter();
  const CspaceFreeRegion dut(*diagram_, plant_, scene_graph_,
                             SeparatingPlaneOrder::kAffine,
                             CspaceRegionType::kGenericPolytope);
  const auto& plant = dut.rational_forward_kinematics().plant();
  auto context = plant.CreateDefaultContext();

  Eigen::VectorXd q_star;
  Eigen::MatrixXd C;
  Eigen::VectorXd d;
  Eigen::VectorXd q_not_in_collision;
  ConstructInitialCspacePolytope(dut, &q_star, &C, &d, &q_not_in_collision);

  CspaceFreeRegion::FilteredCollisionPairs filtered_collision_pairs{
      {SortedPair<geometry::GeometryId>(
          dut.separating_planes()[0].positive_side_polytope->get_id(),
          dut.separating_planes()[0].negative_side_polytope->get_id())}};
  std::vector<CspaceFreeRegion::CspacePolytopeTuple> alternation_tuples;
  VectorX<symbolic::Polynomial> d_minus_Ct;
  Eigen::VectorXd t_lower, t_upper;
  VectorX<symbolic::Polynomial> t_minus_t_lower, t_upper_minus_t;
  MatrixX<symbolic::Variable> C_var;
  VectorX<symbolic::Variable> d_var, lagrangian_gram_vars, verified_gram_vars,
      separating_plane_vars;
  std::vector<std::vector<int>> separating_plane_to_tuples;
  dut.GenerateTuplesForBilinearAlternation(
      q_star, filtered_collision_pairs, C.rows(), &alternation_tuples,
      &d_minus_Ct, &t_lower, &t_upper, &t_minus_t_lower, &t_upper_minus_t,
      &C_var, &d_var, &lagrangian_gram_vars, &verified_gram_vars,
      &separating_plane_vars, &separating_plane_to_tuples);
  const std::vector<bool> is_plane_active = internal::IsPlaneActive(
      dut.separating_planes(), filtered_collision_pairs);
  EXPECT_FALSE(is_plane_active[0]);
  EXPECT_EQ(is_plane_active.size(), dut.separating_planes().size());
  for (int i = 1; i < static_cast<int>(is_plane_active.size()); ++i) {
    EXPECT_TRUE(is_plane_active[i]);
  }
  const int num_active_planes = is_plane_active.size() - 1;
  for (const std::optional<int> num_threads :
       {std::optional<int>(std::nullopt), std::optional<int>(-1),
        std::optional<int>(5)}) {
    Eigen::VectorXd lagrangian_gram_var_vals, verified_gram_var_vals,
        separating_plane_var_vals;
    CspaceFreeRegionSolution cspace_free_region_solution;
    const VerificationOption verification_option{};
    const double redundant_tighten = 0;
    solvers::SolverOptions solver_options{};
    const bool verbose{true};
    bool is_success = internal::FindLagrangianAndSeparatingPlanes(
        dut, alternation_tuples, C, d, lagrangian_gram_vars, verified_gram_vars,
        separating_plane_vars, t_lower, t_upper, verification_option,
        redundant_tighten, solver_options, verbose, num_threads,
        separating_plane_to_tuples, &lagrangian_gram_var_vals,
        &verified_gram_var_vals, &separating_plane_var_vals,
        &cspace_free_region_solution);
    EXPECT_TRUE(is_success);
    EXPECT_EQ(cspace_free_region_solution.separating_planes.size(),
              num_active_planes);
    TestLagrangianResult(dut, alternation_tuples, C, d, separating_plane_vars,
                         t_minus_t_lower, t_upper_minus_t,
                         lagrangian_gram_var_vals, verified_gram_var_vals,
                         separating_plane_var_vals, 1E-5);
    CheckSeparatingPlanesSol(dut, filtered_collision_pairs,
                             cspace_free_region_solution.separating_planes);

    // Now increase d a lot. The SOS problem should be infeasible.
    const Eigen::VectorXd d_infeasible = (d.array() + 1E5).matrix();
    is_success = internal::FindLagrangianAndSeparatingPlanes(
        dut, alternation_tuples, C, d_infeasible, lagrangian_gram_vars,
        verified_gram_vars, separating_plane_vars, t_lower, t_upper,
        verification_option, redundant_tighten, solver_options, verbose,
        num_threads, separating_plane_to_tuples, &lagrangian_gram_var_vals,
        &verified_gram_var_vals, &separating_plane_var_vals,
        &cspace_free_region_solution);
    EXPECT_FALSE(is_success);
  }
}

GTEST_TEST(CalcPolynomialFromGram, Test1) {
  const symbolic::Variable x("x");
  // monomial_basis = [x, x², 1]
  const Vector3<symbolic::Monomial> monomial_basis(
      symbolic::Monomial(x, 1), symbolic::Monomial(x, 2), symbolic::Monomial());
  Eigen::Matrix3d Q;
  // clang-format off
  Q << 1, 2, 3,
       4, 2, 5,
       4, 1, 3;
  // clang-format on
  Vector6<double> Q_lower;
  Q_lower << 1, 3, 3.5, 2, 3, 3;
  const auto ret1 = CalcPolynomialFromGram<double>(monomial_basis, Q);
  // ret should be 6x³ + 7x + 2x⁴ + 7x²+3
  const symbolic::Polynomial ret_expected{{{symbolic::Monomial(x, 3), 6},
                                           {symbolic::Monomial(x, 1), 7},
                                           {symbolic::Monomial(x, 4), 2},
                                           {symbolic::Monomial(x, 2), 7},
                                           {symbolic::Monomial(), 3}}};
  EXPECT_TRUE(ret1.EqualToAfterExpansion(ret_expected));

  const auto ret2 =
      CalcPolynomialFromGramLower<double>(monomial_basis, Q_lower);
  EXPECT_TRUE(ret2.EqualToAfterExpansion(ret_expected));
}

GTEST_TEST(CalcPolynomialFromGram, Test2) {
  // Test the overloaded function with MathematicalProgramResult as an input.
  const symbolic::Variable x("x");
  // monomial_basis = [x, x², 1]
  const Vector3<symbolic::Monomial> monomial_basis(
      symbolic::Monomial(x, 1), symbolic::Monomial(x, 2), symbolic::Monomial());
  Eigen::Matrix3d Q;
  // clang-format off
  Q << 1, 3, 3.5,
       3, 2, 4,
       3.5, 4, 3;
  // clang-format on
  Matrix3<symbolic::Variable> Q_var;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Q_var(i, j) = symbolic::Variable(fmt::format("Q({}, {})", i, j));
    }
  }
  Vector6<symbolic::Variable> Q_lower_var;
  Q_lower_var << Q_var(0, 0), Q_var(1, 0), Q_var(2, 0), Q_var(1, 1),
      Q_var(2, 1), Q_var(2, 2);

  solvers::MathematicalProgramResult result;
  // set result to store Q1.
  std::unordered_map<symbolic::Variable::Id, int> decision_variable_index;
  int variable_count = 0;
  Eigen::Matrix<double, 9, 1> Q_val_flat;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      decision_variable_index.emplace(Q_var(i, j).get_id(), variable_count);
      Q_val_flat(variable_count) = Q(i, j);
      variable_count++;
    }
  }
  result.set_decision_variable_index(decision_variable_index);
  result.set_x_val(Q_val_flat);

  const auto ret1 = CalcPolynomialFromGram(monomial_basis, Q_var, result);
  // ret should be 6x³ + 7x + 2x⁴ + 9x²+3
  const symbolic::Polynomial ret_expected{{{symbolic::Monomial(x, 3), 6},
                                           {symbolic::Monomial(x, 1), 7},
                                           {symbolic::Monomial(x, 4), 2},
                                           {symbolic::Monomial(x, 2), 9},
                                           {symbolic::Monomial(), 3}}};
  EXPECT_TRUE(ret1.EqualToAfterExpansion(ret_expected));
  const auto ret2 =
      CalcPolynomialFromGramLower(monomial_basis, Q_lower_var, result);
  EXPECT_TRUE(ret2.EqualToAfterExpansion(ret_expected));
}

GTEST_TEST(SymmetricMatrixFromLower, Test) {
  Eigen::MatrixXd mat1;
  SymmetricMatrixFromLower<double>(2, Eigen::Vector3d(1, 2, 3), &mat1);
  Eigen::Matrix2d mat1_expected;
  // clang-format off
  mat1_expected << 1, 2,
                   2, 3;
  // clang-format on
  EXPECT_TRUE(CompareMatrices(mat1, mat1_expected));

  Vector6<double> lower2;
  lower2 << 1, 2, 3, 4, 5, 6;
  Eigen::Matrix3d mat2_expected;
  // clang-format off
  mat2_expected << 1, 2, 3,
                   2, 4, 5,
                   3, 5, 6;
  // clang-format on
  Eigen::MatrixXd mat2;
  SymmetricMatrixFromLower<double>(3, lower2, &mat2);
  EXPECT_TRUE(CompareMatrices(mat2, mat2_expected));
}

GTEST_TEST(AddInscribedEllipsoid, Test1) {
  // Test an ellipsoid inside the box with four corners (-1, 0), (1, 0), (-1,
  // 2), (1, 2). Find the largest inscribed ellipsoid.
  solvers::MathematicalProgram prog;
  const auto P = prog.NewSymmetricContinuousVariables<2>();
  const auto q = prog.NewContinuousVariables<2>();

  const Eigen::Vector2d t_lower(-1, 0);
  const Eigen::Vector2d t_upper(1, 2);
  AddInscribedEllipsoid(&prog, Eigen::MatrixXd::Zero(0, 2), Eigen::VectorXd(0),
                        t_lower, t_upper, P, q);
  prog.AddMaximizeLogDeterminantCost(P.cast<symbolic::Expression>());
  const auto result = solvers::Solve(prog);
  const double tol = 1E-7;
  EXPECT_TRUE(
      CompareMatrices(result.GetSolution(q), Eigen::Vector2d(0, 1), tol));
  const auto P_sol = result.GetSolution(P);
  EXPECT_TRUE(CompareMatrices(P_sol * P_sol.transpose(),
                              Eigen::Matrix2d::Identity(), tol));
}

GTEST_TEST(AddInscribedEllipsoid, Test2) {
  // Test an ellipsoid inside the box with four corners (0, 0), (1, 1), (-1,
  // 1), (2, 0). Find the largest inscribed ellipsoid.
  solvers::MathematicalProgram prog;
  const auto P = prog.NewSymmetricContinuousVariables<2>();
  const auto q = prog.NewContinuousVariables<2>();

  const Eigen::Vector2d t_lower(-1, 0);
  const Eigen::Vector2d t_upper(1, 2);
  Eigen::Matrix<double, 4, 2> C;
  // clang-format off
  C << 1, 1,
       -1, 1,
       1, -1,
       -1, -1;
  // clang-format on
  const Eigen::Vector4d d(2, 2, 0, 0);
  AddInscribedEllipsoid(&prog, C, d, t_lower, t_upper, P, q);
  prog.AddMaximizeLogDeterminantCost(P.cast<symbolic::Expression>());
  const auto result = solvers::Solve(prog);
  const double tol = 1E-7;
  EXPECT_TRUE(
      CompareMatrices(result.GetSolution(q), Eigen::Vector2d(0, 1), tol));
  const auto P_sol = result.GetSolution(P);
  EXPECT_TRUE(CompareMatrices(P_sol * P_sol.transpose(),
                              0.5 * Eigen::Matrix2d::Identity(), tol));
}

GTEST_TEST(AddOuterPolytope, Test) {
  solvers::MathematicalProgram prog;
  Eigen::Matrix2d P;
  P << 1, 2, 2, 5;
  const Eigen::Vector2d q(3, 4);
  constexpr int C_rows = 6;
  const auto C = prog.NewContinuousVariables<C_rows, 2>();
  const auto d = prog.NewContinuousVariables<C_rows>();
  const auto margin = prog.NewContinuousVariables<C_rows>();
  AddOuterPolytope(&prog, P, q, C, d, margin);
  // Add the constraint that the margin is at least 0.5
  Eigen::Matrix<double, C_rows, 1> min_margin;
  min_margin << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  prog.AddBoundingBoxConstraint(min_margin, min_margin, margin);
  // There is a trivial solution to set C = 0 and d>=0, and the polytope C* t <=
  // d is just the entire space. To avoid this trivial solution, we add the
  // constraint that the C.row(i).sum() >= 0.001 or <= -0.001;
  for (int i = 0; i < C_rows / 2; ++i) {
    prog.AddLinearConstraint(Eigen::Vector2d::Ones(), 0.001, kInf, C.row(i));
  }
  for (int i = C_rows / 2; i < C_rows; ++i) {
    prog.AddLinearConstraint(Eigen::Vector2d::Ones(), -kInf, -0.001, C.row(i));
  }
  const auto result = solvers::Solve(prog);
  EXPECT_TRUE(result.is_success());
  const auto C_val = result.GetSolution(C);
  const auto d_val = result.GetSolution(d);
  // Now solve a program
  // min |Py+q-x|
  // s.t C_val.row(i) * x >= d_val(i)
  //     |y|₂ <= 1
  // Namely we find the minimal distance between the ellipsoid and the outside
  // hafplane of C_val.row(i) * x >= d_val(i). This distance should be at least
  // min_margin.
  for (int i = 0; i < C.rows(); ++i) {
    solvers::MathematicalProgram prog_check;
    auto x = prog_check.NewContinuousVariables<2>();
    auto y = prog_check.NewContinuousVariables<2>();
    // Add the constraint that [1, y] is in the lorentz cone.
    const Vector3<symbolic::Expression> lorentz_cone_expr1(1, y(0), y(1));
    prog_check.AddLorentzConeConstraint(lorentz_cone_expr1);
    prog_check.AddLinearConstraint(C_val.row(i), d_val(i), kInf, x);
    // Now add the slack variable s with the constraint [s, Py+q-x] is in the
    // Lorentz cone.
    const auto s = prog_check.NewContinuousVariables<1>()(0);
    Vector3<symbolic::Expression> lorentz_cone_expr2;
    lorentz_cone_expr2(0) = s;
    lorentz_cone_expr2.tail<2>() = P * y + q - x;
    prog_check.AddLorentzConeConstraint(lorentz_cone_expr2);
    prog_check.AddLinearCost(s);
    const auto result_check = solvers::Solve(prog_check);
    EXPECT_TRUE(result_check.is_success());
    EXPECT_GE(result_check.get_optimal_cost(), min_margin(i) - 1E-6);
  }
}

GTEST_TEST(GetConvexPolytopes, Test) {
  systems::DiagramBuilder<double> builder;
  auto iiwa = builder.AddSystem<MultibodyPlant<double>>(
      ConstructIiwaPlant("iiwa14_no_collision.sdf", false));

  auto sg = builder.AddSystem<geometry::SceneGraph<double>>();
  iiwa->RegisterAsSourceForSceneGraph(sg);
  builder.Connect(sg->get_query_output_port(),
                  iiwa->get_geometry_query_input_port());
  builder.Connect(iiwa->get_geometry_poses_output_port(),
                  sg->get_source_pose_port(iiwa->get_source_id().value()));
  // Now add the collision geometries.
  const auto link7_box1_id = iiwa->RegisterCollisionGeometry(
      iiwa->GetBodyByName("iiwa_link_7"), {}, geometry::Box(0.1, 0.2, 0.3),
      "link7_box1", CoulombFriction<double>());
  const math::RigidTransform<double> X_7P2(
      math::RotationMatrixd(
          Eigen::AngleAxisd(0.2, Eigen::Vector3d(0.1, 0.3, 0.5).normalized())),
      Eigen::Vector3d(0.1, 0.5, -0.2));
  const Eigen::Vector3d box2_size(0.2, 0.3, 0.1);
  const auto link7_box2_id = iiwa->RegisterCollisionGeometry(
      iiwa->GetBodyByName("iiwa_link_7"), X_7P2,
      geometry::Box(box2_size(0), box2_size(1), box2_size(2)), "link7_box2",
      CoulombFriction<double>());

  const auto world_box_id = iiwa->RegisterCollisionGeometry(
      iiwa->world_body(), {}, geometry::Box(0.2, 0.1, 0.3), "world_box",
      CoulombFriction<double>());
  iiwa->Finalize();
  auto diagram = builder.Build();

  const auto polytope_geometries = GetConvexPolytopes(*diagram, iiwa, sg);
  EXPECT_EQ(polytope_geometries.size(), 2u);
  const auto& link7_polytopes =
      polytope_geometries.at(iiwa->GetBodyByName("iiwa_link_7").index());
  EXPECT_EQ(link7_polytopes.size(), 2u);
  const auto& obstacles = polytope_geometries.at(iiwa->world_body().index());
  EXPECT_EQ(obstacles[0].body_index(), iiwa->world_body().index());
  EXPECT_EQ(obstacles[0].get_id(), world_box_id);

  std::unordered_map<ConvexGeometry::Id, const ConvexPolytope*>
      link7_polytope_map;
  for (const auto& link7_polytope : link7_polytopes) {
    link7_polytope_map.emplace(link7_polytope.get_id(), &link7_polytope);
  }
  EXPECT_EQ(link7_polytope_map.size(), 2u);
  const ConvexPolytope* link7_box1 = link7_polytope_map.at(link7_box1_id);
  const ConvexPolytope* link7_box2 = link7_polytope_map.at(link7_box2_id);
  EXPECT_EQ(link7_box1->body_index(),
            iiwa->GetBodyByName("iiwa_link_7").index());
  EXPECT_EQ(link7_box2->body_index(),
            iiwa->GetBodyByName("iiwa_link_7").index());
  // Now compute the geometry vertices manually and check with
  // link7_box1->p_BV().
  const Eigen::Matrix<double, 3, 8> link7_box2_vertices =
      GenerateBoxVertices(box2_size, X_7P2);
  EXPECT_EQ(link7_box2->p_BV().cols(), 8);
  for (int i = 0; i < 8; ++i) {
    bool found_match = false;
    for (int j = 0; j < 8; ++j) {
      if ((link7_box2->p_BV().col(i) - link7_box2_vertices.col(j)).norm() <
          1E-8) {
        found_match = true;
      }
    }
    EXPECT_TRUE(found_match);
  }
}

GTEST_TEST(FindRedundantInequalities, Test) {
  Eigen::Matrix<double, 4, 2> C;
  C << 1, 1, -1, 1, 1, -1, -1, -1;
  Eigen::Vector4d d(2, 2, 2, 2);
  Eigen::Vector2d t_lower(-2.5, -2.5);
  Eigen::Vector2d t_upper(2.5, 2.5);
  std::unordered_set<int> C_redundant_indices, t_lower_redundant_indices,
      t_upper_redundant_indices;
  double tighten = 0;
  FindRedundantInequalities(C, d, t_lower, t_upper, tighten,
                            &C_redundant_indices, &t_lower_redundant_indices,
                            &t_upper_redundant_indices);
  EXPECT_TRUE(C_redundant_indices.empty());
  EXPECT_EQ(t_lower_redundant_indices, std::unordered_set<int>({0, 1}));
  EXPECT_EQ(t_upper_redundant_indices, std::unordered_set<int>({0, 1}));
  // Set tighten = 0.6, now the bound t_lower <= t <= t_upper is not redundant.
  tighten = 0.6;
  FindRedundantInequalities(C, d, t_lower, t_upper, tighten,
                            &C_redundant_indices, &t_lower_redundant_indices,
                            &t_upper_redundant_indices);
  EXPECT_TRUE(C_redundant_indices.empty());
  EXPECT_TRUE(t_lower_redundant_indices.empty());
  EXPECT_TRUE(t_upper_redundant_indices.empty());
  // Set tighten = -3.1, now C*t<=d is redundant.
  tighten = -3.1;
  FindRedundantInequalities(C, d, t_lower, t_upper, tighten,
                            &C_redundant_indices, &t_lower_redundant_indices,
                            &t_upper_redundant_indices);
  EXPECT_EQ(C_redundant_indices, std::unordered_set<int>({0, 1, 2, 3}));
  EXPECT_EQ(t_lower_redundant_indices, std::unordered_set<int>({0, 1}));
  EXPECT_EQ(t_upper_redundant_indices, std::unordered_set<int>({0, 1}));
}

GTEST_TEST(FindEpsilonLower, Test) {
  const Eigen::Vector2d t_lower(-1, -1);
  const Eigen::Vector2d t_upper(1, 1);
  // C*t<=d is |x| + |y| <= 3
  Eigen::Matrix<double, 4, 2> C;
  // clang-format off
  C << 1, 1,
       1, -1,
       -1, 1,
       -1, -1;
  // clang-format on
  Eigen::Vector4d d(3, 3, 3, 3);
  const double tol{1E-6};
  EXPECT_NEAR(
      FindEpsilonLower(C, d, t_lower, t_upper, std::nullopt, std::nullopt), -3,
      tol);

  // C*t<=d is |x-2| + |y-2|<=3
  d << 7, 3, 3, -1;
  EXPECT_NEAR(
      FindEpsilonLower(C, d, t_lower, t_upper, std::nullopt, std::nullopt), -1,
      tol);

  // C*t<=d is |x| + |y| <= 2, with inner_pts being (0.1, 0.5), (-0.3, -.4), (1,
  // 0.2)
  d << 2, 2, 2, 2;
  Eigen::Matrix<double, 2, 3> inner_pts;
  // clang-format off
  inner_pts << 0.1, -0.3, 1,
               0.5, -0.4, 0.2;
  // clang-format on
  EXPECT_NEAR(FindEpsilonLower(C, d, t_lower, t_upper, inner_pts, std::nullopt),
              -0.8, tol);

  // C*t<=d is |x| + |y| <= 2, with inner_polytope being
  // x+y >= 0.2, x<= 0.5, y <= 0.6
  d << 2, 2, 2, 2;
  Eigen::MatrixXd C_inner(3, 2);
  // clang-format off
  C_inner << -1, -1,
              1,  0,
              0,  1;
  // clang-format on
  Eigen::VectorXd d_inner = Eigen::Vector3d(-0.2, 0.5, 0.6);
  EXPECT_NEAR(FindEpsilonLower(C, d, t_lower, t_upper, std::nullopt,
                               std::make_pair(C_inner, d_inner)),
              -0.9, tol);
}

GTEST_TEST(GetCspacePolytope, Test) {
  Eigen::Matrix<double, 3, 2> C;
  // Use arbitrary value of C, d, t_lower, t_upper.
  // clang-format off
  C << 1, 3,
       2, 4,
       -1, 2;
  // clang-format on
  Eigen::Vector3d d(1, 3, 5);

  Eigen::Vector2d t_lower(2, -4);
  Eigen::Vector2d t_upper(4, 8);
  Eigen::MatrixXd C_bar;
  Eigen::VectorXd d_bar;
  GetCspacePolytope(C, d, t_lower, t_upper, &C_bar, &d_bar);
  Eigen::MatrixXd C_bar_expected(7, 2);
  C_bar_expected.topRows<3>() = C;
  C_bar_expected.middleRows<2>(3) = Eigen::Matrix2d::Identity();
  C_bar_expected.bottomRows<2>() = -Eigen::Matrix2d::Identity();
  Eigen::VectorXd d_bar_expected(7);
  d_bar_expected << d, t_upper, -t_lower;
  EXPECT_TRUE(CompareMatrices(C_bar, C_bar_expected));
  EXPECT_TRUE(CompareMatrices(d_bar, d_bar_expected));
}

GTEST_TEST(AddCspacePolytopeContainment, Test1) {
  solvers::MathematicalProgram prog;
  // Contain the polytope |x| + |y| <= 1 and |x|<= 0.8, |y|<= 0.9.
  Eigen::Matrix<double, 4, 2> C_inner;
  // clang-format off
  C_inner << 1, 1,
             1, -1,
             -1, 1,
             -1, -1;
  // clang-format on
  Eigen::Vector4d d_inner(1, 1, 1, 1);
  const Eigen::Vector2d t_lower(-0.8, -0.9);
  const Eigen::Vector2d t_upper(0.8, 0.9);
  // Find a triangle that contains the polytope.
  auto C = prog.NewContinuousVariables<3, 2>();
  auto d = prog.NewContinuousVariables<3>();
  AddCspacePolytopeContainment(&prog, C, d, C_inner, d_inner, t_lower, t_upper);
  // Also constraint d >= 1 to avoid the trivial solution C=0 and d=0
  prog.AddBoundingBoxConstraint(1, kInf, d);
  auto result = solvers::Solve(prog);
  EXPECT_TRUE(result.is_success());
  // The vertices of the contained region
  Eigen::Matrix<double, 8, 2> vertices;
  // clang-format off
  vertices << -0.1, 0.9,
               0.1, 0.9,
              -0.8, 0.2,
               0.8, 0.2,
              -0.8, -0.2,
               0.8, 0.2,
              -0.1, -0.9,
               0.1, -0.9;
  // clang-format on
  auto C_sol = result.GetSolution(C);
  auto d_sol = result.GetSolution(d);
  const double tol{1E-6};
  for (int i = 0; i < vertices.rows(); ++i) {
    EXPECT_TRUE(
        ((C_sol * vertices.row(i).transpose()).array() <= d_sol.array() + tol)
            .all());
  }
  // Now I fix C and only minimize d
  prog.AddBoundingBoxConstraint(Eigen::Vector2d(1, 2), Eigen::Vector2d(1, 2),
                                C.row(0).transpose());
  prog.AddLinearCost(d(0));
  result = solvers::Solve(prog);
  // The line x + 2y touches the inner polytope at (0.1, 0.9), namely d(0) = 0.1
  // + 2 * 0.9=1.9.
  EXPECT_NEAR(result.GetSolution(d(0)), 1.9, tol);
}

GTEST_TEST(AddCspacePolytopeContainment, Test2) {
  solvers::MathematicalProgram prog;
  Eigen::Matrix<double, 2, 3> inner_pts;
  // clang-format off
  inner_pts << 1, 2, -4,
               3, -1, -1;
  // clang-format on
  auto C = prog.NewContinuousVariables<4, 2>();
  auto d = prog.NewContinuousVariables<4>();
  AddCspacePolytopeContainment(&prog, C, d, inner_pts);
  // Avoid the trivial solution C = d = 0.
  prog.AddBoundingBoxConstraint(1, kInf, d);
  const auto result = solvers::Solve(prog);
  EXPECT_TRUE(result.is_success());
  const auto C_sol = result.GetSolution(C);
  const auto d_sol = result.GetSolution(d);
  for (int i = 0; i < inner_pts.cols(); ++i) {
    EXPECT_TRUE(
        ((C_sol * inner_pts.col(i)).array() <= d_sol.array() + 1E-6).all());
  }
}

}  // namespace multibody
}  // namespace drake

int main(int argc, char** argv) {
  // Ensure that we have the MOSEK license for the entire duration of this test,
  // so that we do not have to release and re-acquire the license for every
  // test.
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
