#include "drake/planning/iris/fast_clique_inflation.h"

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/geometry/test_utilities/meshcat_environment.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/planning/robot_diagram_builder.h"
#include "drake/planning/scene_graph_collision_checker.h"

namespace drake {
namespace planning {
namespace {

using common::MaybePauseForUser;
using Eigen::Vector2d;
using geometry::Meshcat;
using geometry::Rgba;
using geometry::Sphere;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;
using geometry::optimization::VPolytope;
using symbolic::Variable;

const double kInf = std::numeric_limits<double>::infinity();

// Helper method for testing FastCliqueInflation from a urdf string.
HPolyhedron FastCliqueInflationFromUrdf(
    const std::string urdf, const Eigen::MatrixXd& clique,
    const FastCliqueInflationOptions& options) {
  CollisionCheckerParams params;
  RobotDiagramBuilder<double> builder(0.0);

  builder.parser().package_map().AddPackageXml(FindResourceOrThrow(
      "drake/multibody/parsing/test/box_package/package.xml"));
  params.robot_model_instances =
      builder.parser().AddModelsFromString(urdf, "urdf");

  auto plant_ptr = &(builder.plant());
  plant_ptr->Finalize();

  params.model = builder.Build();
  params.edge_step_size = 0.01;

  HPolyhedron domain = HPolyhedron::MakeBox(
      plant_ptr->GetPositionLowerLimits(), plant_ptr->GetPositionUpperLimits());

  planning::SceneGraphCollisionChecker checker(std::move(params));
  // plant.SetPositions(&plant.GetMyMutableContextFromRoot(context.get()),
  // sample);
  return FastCliqueInflation(checker, clique, domain, options);
}

// One prismatic link with joint limits.  Iris should return the joint limits.
GTEST_TEST(FastCliqueInflationTest, JointLimits) {
  const std::string limits_urdf = R"(
<robot name="limits">
  <link name="movable">
    <collision>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
  </link>
  <joint name="movable" type="prismatic">
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="world"/>
    <child link="movable"/>
  </joint>
</robot>
)";
  const Vector1d sample = Vector1d::Zero();
  Eigen::MatrixXd clique = sample;
  FastCliqueInflationOptions options;
  options.verbose = true;
  std::cout << "calling fci urdf\n";
  HPolyhedron region =
      FastCliqueInflationFromUrdf(limits_urdf, clique, options);

  EXPECT_EQ(region.ambient_dimension(), 1);

  const double kTol = 1e-5;
  const double qmin = -2.0, qmax = 2.0;
  EXPECT_TRUE(region.PointInSet(Vector1d{qmin + kTol}));
  EXPECT_TRUE(region.PointInSet(Vector1d{qmax - kTol}));
  EXPECT_FALSE(region.PointInSet(Vector1d{qmin - kTol}));
  EXPECT_FALSE(region.PointInSet(Vector1d{qmax + kTol}));
}
// Taken from IrisInConfigurationSpace unit tests.
// A simple double pendulum with link lengths `l1` and `l2` with a sphere at the
// tip of radius `r` between two (fixed) walls at `w` from the origin.  The
// true configuration space is - w + r ≤ l₁s₁ + l₂s₁₊₂ ≤ w - r.  These regions
// are visualized at https://www.desmos.com/calculator/ff0hbnkqhm.
GTEST_TEST(FastCliqueInflationTest, DoublePendulum) {
  const double l1 = 2.0;
  const double l2 = 1.0;
  const double r = .5;
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
    <collision name="ball">
      <origin rpy="0 0 0" xyz="0 0 -{l2}"/>
      <geometry><sphere radius="{r}"/></geometry>
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
      fmt::arg("l2", l2), fmt::arg("r", r));

  const Vector2d sample = Vector2d::Zero();
  std::shared_ptr<Meshcat> meshcat = geometry::GetTestEnvironmentMeshcat();
  meshcat->Delete("face_pt");
  meshcat->Delete("start_pt");
  meshcat->Delete("bisection");
  FastCliqueInflationOptions options;
  options.verbose = true;
  options.meshcat = meshcat;
  Eigen::MatrixXd clique(2, 1);
  clique.col(0) = sample;
  HPolyhedron region =
      FastCliqueInflationFromUrdf(double_pendulum_urdf, clique, options);
  EXPECT_EQ(region.ambient_dimension(), 2);
  // Confirm that we've found a substantial region.
  EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 2.0);

  EXPECT_TRUE(region.PointInSet(Vector2d{.4, 0.0}));
  EXPECT_FALSE(region.PointInSet(Vector2d{.5, 0.0}));
  EXPECT_TRUE(region.PointInSet(Vector2d{.3, .3}));
  EXPECT_FALSE(region.PointInSet(Vector2d{.4, .3}));
  EXPECT_TRUE(region.PointInSet(Vector2d{-.4, 0.0}));
  EXPECT_FALSE(region.PointInSet(Vector2d{-.5, 0.0}));
  EXPECT_TRUE(region.PointInSet(Vector2d{-.3, -.3}));
  EXPECT_FALSE(region.PointInSet(Vector2d{-.4, -.3}));

  {
    meshcat->Set2dRenderMode(math::RigidTransformd(Eigen::Vector3d{0, 0, 1}),
                             -3.25, 3.25, -3.25, 3.25);
    meshcat->SetProperty("/Grid", "visible", true);
    Eigen::RowVectorXd theta2s =
        Eigen::RowVectorXd::LinSpaced(100, -1.57, 1.57);
    Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 2 * theta2s.size() + 1);
    const double c = -w + r;
    for (int i = 0; i < theta2s.size(); ++i) {
      const double a = l1 + l2 * std::cos(theta2s[i]),
                   b = l2 * std::sin(theta2s[i]);
      // wolfram solve a*sin(q) + b*cos(q) = c for q
      points(0, i) =
          2 * std::atan((std::sqrt(a * a + b * b - c * c) + a) / (b + c)) +
          M_PI;
      points(1, i) = theta2s[i];
      points(0, points.cols() - i - 2) =
          2 * std::atan((std::sqrt(a * a + b * b - c * c) + a) / (b - c)) -
          M_PI;
      points(1, points.cols() - i - 2) = theta2s[i];
    }
    points.col(points.cols() - 1) = points.col(0);
    meshcat->SetLine("True C_free", points, 2.0, Rgba(0, 0, 1));
    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("Inflated Clique", points, 2.0, Rgba(0, 1, 0));

    MaybePauseForUser();
  }
}

const char block_urdf[] = R"(
<robot name="block">
  <link name="fixed">
    <collision name="ground">
      <origin rpy="0 0 0" xyz="0 0 -1"/>
      <geometry><box size="10 10 2"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="link1"/>
  <joint name="joint1" type="prismatic">
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="3.0"/>
    <parent link="world"/>
    <child link="link1"/>
  </joint>
  <link name="link2">
    <collision name="block">
      <geometry><box size="2 1 1"/></geometry>
    </collision>
  </link>
  <joint name="joint2" type="revolute">
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159"/>
    <parent link="link1"/>
    <child link="link2"/>
  </joint>
</robot>
)";
// Taken from IrisInConfigurationSpace unit tests.
// A block on a vertical track, free to rotate (in the plane) with width `w` of
// 2 and height `h` of 1, plus a ground plane at z=0.  The true configuration
// space is min(q₀ ± .5w sin(q₁) ± .5h cos(q₁)) ≥ 0, where the min is over the
// ±. This region is also visualized at
// https://www.desmos.com/calculator/ok5ckpa1kp.
GTEST_TEST(FastCliqueInflation, BlockOnGround) {
  const Vector2d sample{1.0, 0.0};
  std::shared_ptr<Meshcat> meshcat = geometry::GetTestEnvironmentMeshcat();
  meshcat->Delete("face_pt");
  meshcat->Delete("start_pt");
  meshcat->Delete("bisection");
  meshcat->Delete("/obstacles");
  meshcat->Delete("/drake/Domain");
  FastCliqueInflationOptions options;
  options.verbose = true;
  options.meshcat = meshcat;
  Eigen::MatrixXd clique(3, 3);
  // clang-format off
        clique << 2,  2, 2.2,
                  2, -2, 0.,
                  0,  0,  0;
  // clang-format on
  HPolyhedron region =
      FastCliqueInflationFromUrdf(block_urdf, clique.topRows(2), options);
  {
    meshcat->Set2dRenderMode(math::RigidTransformd(Eigen::Vector3d{0, 0, 1}), 0,
                             3.25, -3.25, 3.25);
    meshcat->SetProperty("/Grid", "visible", true);
    Eigen::RowVectorXd thetas = Eigen::RowVectorXd::LinSpaced(100, -M_PI, M_PI);
    const double w = 2, h = 1;
    Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 2 * thetas.size() + 1);
    for (int i = 0; i < thetas.size(); ++i) {
      const double a = 0.5 *
                       (-w * std::sin(thetas[i]) - h * std::cos(thetas[i])),
                   b = 0.5 *
                       (-w * std::sin(thetas[i]) + h * std::cos(thetas[i])),
                   c = 0.5 *
                       (+w * std::sin(thetas[i]) - h * std::cos(thetas[i])),
                   d = 0.5 *
                       (+w * std::sin(thetas[i]) + h * std::cos(thetas[i]));
      points(0, i) = std::max({a, b, c, d});
      points(1, i) = thetas[i];
      points(0, points.cols() - i - 2) = 3.0;
      points(1, points.cols() - i - 2) = thetas[i];
    }
    for (int pt_to_draw = 0; pt_to_draw < clique.cols(); ++pt_to_draw) {
      Eigen::Vector3d point_to_draw = Eigen::Vector3d::Zero();
      std::string path = fmt::format("clique_pt/{}", pt_to_draw);
      options.meshcat->SetObject(path, Sphere(0.04),
                                 geometry::Rgba(1, 0, 0.0, 1.0));
      point_to_draw.head(2) = clique.col(pt_to_draw);
      options.meshcat->SetTransform(
          path, math::RigidTransform<double>(point_to_draw));
      EXPECT_TRUE(region.PointInSet(point_to_draw.head(2)));
    }

    EXPECT_EQ(region.ambient_dimension(), 2);
    // Confirm that we've found a substantial region.
    EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 2.0);
    points.col(points.cols() - 1) = points.col(0);
    meshcat->SetLine("True C_free", points, 2.0, Rgba(0, 0, 1));
    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("Inflated Clique", points, 2.0, Rgba(0, 1, 0));

    MaybePauseForUser();
  }
}
// Taken from IrisInConfigurationSpace unit tests.
GTEST_TEST(FastCliqueInflationTest, ConvexConfigurationSpace) {
  const double l = 1.5;
  const double r = 0.1;

  std::shared_ptr<Meshcat> meshcat = geometry::GetTestEnvironmentMeshcat();
  meshcat->Delete("face_pt");
  meshcat->Delete("start_pt");
  meshcat->Delete("bisection");
  meshcat->Set2dRenderMode(math::RigidTransformd(Eigen::Vector3d{0, 0, 1}),
                           -3.25, 3.25, -3.25, 3.25);
  meshcat->SetProperty("/Grid", "visible", true);
  Eigen::RowVectorXd theta1s = Eigen::RowVectorXd::LinSpaced(100, -1.5, 1.5);
  Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 2 * theta1s.size());
  for (int i = 0; i < theta1s.size(); ++i) {
    points(0, i) = r - l * cos(theta1s[i]);
    points(1, i) = theta1s[i];
    points(0, points.cols() - i - 1) = 0;
    points(1, points.cols() - i - 1) = theta1s[i];
  }
  meshcat->SetLine("True C_free", points, 2.0, Rgba(0, 0, 1));

  const std::string convex_urdf = fmt::format(
      R"(
<robot name="pendulum_on_vertical_track">
  <link name="fixed">
    <collision name="ground">
      <origin rpy="0 0 0" xyz="0 0 -1"/>
      <geometry><box size="10 10 2"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="cart">
  </link>
  <joint name="track" type="prismatic">
    <axis xyz="0 0 1"/>
    <limit lower="-{l}" upper="0"/>
    <parent link="world"/>
    <child link="cart"/>
  </joint>
  <link name="pendulum">
    <collision name="ball">
      <origin rpy="0 0 0" xyz="0 0 {l}"/>
      <geometry><sphere radius="{r}"/></geometry>
    </collision>
  </link>
  <joint name="pendulum" type="revolute">
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
    <parent link="cart"/>
    <child link="pendulum"/>
  </joint>
</robot>
)",
      fmt::arg("l", l), fmt::arg("r", r));

  const Vector2d sample{-0.5, 0.0};
  FastCliqueInflationOptions options;

  // This point should be outside of the configuration space (in collision).
  // The particular value was found by visual inspection using meshcat.
  const double z_test = 0, theta_test = -1.55;
  // Confirm that the pendulum is colliding with the wall with true kinematics:
  EXPECT_LE(z_test + l * std::cos(theta_test), r);

  // Turn on meshcat for addition debugging visualizations.
  // This example is truly adversarial for IRIS. After one iteration, the
  // maximum-volume inscribed ellipse is approximately centered in C-free. So
  // finding a counter-example in the bottom corner (near the test point) is
  // not only difficult because we need to sample in a corner of the polytope,
  // but because the objective is actually pulling the counter-example search
  // away from that corner. Open the meshcat visualization to step through the
  // details!
  options.meshcat = meshcat;
  options.verbose = true;
  Eigen::MatrixXd clique(2, 1);
  clique.col(0) = sample;
  // std::this_thread::sleep_for(std::chrono::milliseconds(100));
  HPolyhedron region =
      FastCliqueInflationFromUrdf(convex_urdf, clique, options);
  // TODO(russt): Expecting the test point to be outside the verified region is
  // too strong of a requirement right now. If we can improve the algorithm then
  // we should make this EXPECT_FALSE.
  if (!region.PointInSet(Vector2d{z_test, theta_test})) {
    log()->info("Our test point is not in the set");
  }

  EXPECT_EQ(region.ambient_dimension(), 2);
  // Confirm that we've found a substantial region.
  EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 0.5);

  {
    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("Inflated Clique", points, 2.0, Rgba(0, 1, 0));

    meshcat->SetObject("Test point", Sphere(0.03), Rgba(1, 0, 0));
    meshcat->SetTransform("Test point", math::RigidTransform(Eigen::Vector3d(
                                            z_test, theta_test, 0)));

    MaybePauseForUser();
  }
}
/* A movable sphere with fixed boxes in all corners.
┌───────────────┐
│┌────┐   ┌────┐│
││    │   │    ││
│└────┘   └────┘│
│       o       │
│┌────┐   ┌────┐│
││    │   │    ││
│└────┘   └────┘│
└───────────────┘ */
const char boxes_in_corners_urdf[] = R"(
<robot name="boxes">
  <link name="fixed">
    <collision name="top_left">
      <origin rpy="0 0 0" xyz="-1 1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
    <collision name="top_right">
      <origin rpy="0 0 0" xyz="1 1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
    <collision name="bottom_left">
      <origin rpy="0 0 0" xyz="-1 -1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
    <collision name="bottom_right">
      <origin rpy="0 0 0" xyz="1 -1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="movable">
    <collision name="sphere">
      <geometry><sphere radius="0.01"/></geometry>
    </collision>
  </link>
  <link name="for_joint"/>
  <joint name="x" type="prismatic">
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="world"/>
    <child link="for_joint"/>
  </joint>
  <joint name="y" type="prismatic">
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="for_joint"/>
    <child link="movable"/>
  </joint>
</robot>
)";
class FastCliqueInflationTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    meshcat = geometry::GetTestEnvironmentMeshcat();
    meshcat->Delete("face_pt");
    meshcat->Delete("start_pt");
    meshcat->Delete("bisection");
    meshcat->Delete("clique_pt");
    meshcat->Delete("True C_free");
    meshcat->Delete("Test point");
    meshcat->Delete("Inflated Clique");
    meshcat->Set2dRenderMode(math::RigidTransformd(Eigen::Vector3d{0, 0, 1}),
                             -3.25, 3.25, -3.25, 3.25);
    meshcat->SetProperty("/Grid", "visible", true);
    // Draw the true cspace.
    Eigen::Matrix3Xd env_points(3, 5);
    // clang-format off
            env_points << -2, 2,  2, -2, -2,
                            2, 2, -2, -2,  2,
                            0, 0,  0,  0,  0;
    // clang-format on
    meshcat->SetLine("Domain", env_points, 8.0, Rgba(0, 0, 0));
    Eigen::Matrix3Xd centers(3, 4);
    double c = 1.0;
    // clang-format off
            centers << -c, c,  c, -c,
                        c, c, -c, -c,
                        0, 0,  0,  0;
    // clang-format on
    Eigen::Matrix3Xd obs_points(3, 5);
    // approximating offset due to sphere radius with fixed offset
    double s = 0.7 + 0.01;
    // clang-format off
            obs_points << -s, s,  s, -s, -s,
                            s, s, -s, -s, s,
                            s, 0,  0,  0,  0;
    // clang-format on
    for (int obstacle_idx = 0; obstacle_idx < 4; ++obstacle_idx) {
      Eigen::Matrix3Xd obstacle = obs_points;
      obstacle.colwise() += centers.col(obstacle_idx);
      meshcat->SetLine(fmt::format("/obstacles/obs_{}", obstacle_idx), obstacle,
                       8.0, Rgba(0, 0, 0));
    }

    // setup fci options
    options.verbose = true;
    options.meshcat = meshcat;
    options.configuration_space_margin = 0.04;
  }

  std::shared_ptr<Meshcat> meshcat;
  FastCliqueInflationOptions options;
};
// Four point clique that is hard to contain with iris but must lie in the
// region resulting from FCI by construction.
TEST_F(FastCliqueInflationTestFixture, FourPointClique) {
  Eigen::Matrix3Xd clique(3, 4);
  double xw, yw;
  xw = 0.4;
  yw = 0.28;
  // clang-format off
    clique << -xw, xw,  xw, -xw,
               yw, yw, -yw, -yw,
                0,  0,  0,  0;
  // clang-format on
  HPolyhedron region = FastCliqueInflationFromUrdf(boxes_in_corners_urdf,
                                                   clique.topRows(2), options);
  EXPECT_EQ(region.ambient_dimension(), 2);
  // Confirm that we've found a substantial region.
  EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 0.5);
  {
    for (int pt_to_draw = 0; pt_to_draw < clique.cols(); ++pt_to_draw) {
      Eigen::Vector3d point_to_draw = Eigen::Vector3d::Zero();
      std::string path = fmt::format("clique_pt/{}", pt_to_draw);
      options.meshcat->SetObject(path, Sphere(0.04),
                                 geometry::Rgba(1, 0, 0.0, 1.0));
      point_to_draw.head(2) = clique.col(pt_to_draw);
      options.meshcat->SetTransform(
          path, math::RigidTransform<double>(point_to_draw));
      EXPECT_TRUE(region.PointInSet(point_to_draw.head(2)));
    }
    Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 20);

    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("Inflated Clique", points, 2.0, Rgba(0, 1, 0));

    MaybePauseForUser();
  }
}
// Three point clique that barely a valid clique. None the less, because it is
// valid all points should always be contained in the region.
TEST_F(FastCliqueInflationTestFixture, ThreePointClique) {
  Eigen::Matrix3Xd clique(3, 3);
  double xw;
  xw = 0.4;
  // clang-format off
    clique << -xw, 0,  0,
                0, 0, xw,
                0, 0,  0;
  // clang-format on
  HPolyhedron region = FastCliqueInflationFromUrdf(boxes_in_corners_urdf,
                                                   clique.topRows(2), options);
  EXPECT_EQ(region.ambient_dimension(), 2);
  // Confirm that we've found a substantial region.
  EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 0.3);
  {
    for (int pt_to_draw = 0; pt_to_draw < clique.cols(); ++pt_to_draw) {
      Eigen::Vector3d point_to_draw = Eigen::Vector3d::Zero();
      std::string path = fmt::format("clique_pt/{}", pt_to_draw);
      options.meshcat->SetObject(path, Sphere(0.04),
                                 geometry::Rgba(1, 0, 0.0, 1.0));
      point_to_draw.head(2) = clique.col(pt_to_draw);
      options.meshcat->SetTransform(
          path, math::RigidTransform<double>(point_to_draw));
      EXPECT_TRUE(region.PointInSet(point_to_draw.head(2)));
    }
    Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 20);

    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("Inflated Clique", points, 2.0, Rgba(0, 1, 0));

    MaybePauseForUser();
  }
}
// Three points that dont form a clique. We expect a region that matches the
// shape of the clique but excludes two of the points.
TEST_F(FastCliqueInflationTestFixture, ThreePointInvalidClique) {
  Eigen::Matrix3Xd clique(3, 3);
  double xw;
  xw = 0.6;
  // clang-format off
    clique << -xw, 0,  0,
                0, 0, xw,
                0, 0,  0;
  // clang-format on
  HPolyhedron region = FastCliqueInflationFromUrdf(boxes_in_corners_urdf,
                                                   clique.topRows(2), options);
  EXPECT_EQ(region.ambient_dimension(), 2);
  // Confirm that we've found a substantial region.
  EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 0.3);
  {
    for (int pt_to_draw = 0; pt_to_draw < clique.cols(); ++pt_to_draw) {
      Eigen::Vector3d point_to_draw = Eigen::Vector3d::Zero();
      std::string path = fmt::format("clique_pt/{}", pt_to_draw);
      options.meshcat->SetObject(path, Sphere(0.04),
                                 geometry::Rgba(1, 0, 0.0, 1.0));
      point_to_draw.head(2) = clique.col(pt_to_draw);
      options.meshcat->SetTransform(
          path, math::RigidTransform<double>(point_to_draw));
    }

    EXPECT_FALSE(region.PointInSet(clique.col(0).head(2)));
    EXPECT_TRUE(region.PointInSet(clique.col(1).head(2)));
    EXPECT_FALSE(region.PointInSet(clique.col(2).head(2)));
    Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 20);

    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("Inflated Clique", points, 2.0, Rgba(0, 1, 0));

    MaybePauseForUser();
  }
}
// Inflating a three point clique that form a line, checking that handling lower
// dimensional cliques works as expected in the case of a valid clique.
TEST_F(FastCliqueInflationTestFixture, ThreePointLineClique) {
  Eigen::Matrix3Xd clique(3, 3);
  double xw;
  xw = 0.8;
  // clang-format off
    clique << 0, 0,  0,
            -xw, 0, xw,
              0, 0,  0;
  // clang-format on
  HPolyhedron region = FastCliqueInflationFromUrdf(boxes_in_corners_urdf,
                                                   clique.topRows(2), options);
  EXPECT_EQ(region.ambient_dimension(), 2);
  // Confirm that we've found a substantial region.
  EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 0.5);
  {
    for (int pt_to_draw = 0; pt_to_draw < clique.cols(); ++pt_to_draw) {
      Eigen::Vector3d point_to_draw = Eigen::Vector3d::Zero();
      std::string path = fmt::format("clique_pt/{}", pt_to_draw);
      options.meshcat->SetObject(path, Sphere(0.04),
                                 geometry::Rgba(1, 0, 0.0, 1.0));
      point_to_draw.head(2) = clique.col(pt_to_draw);
      options.meshcat->SetTransform(
          path, math::RigidTransform<double>(point_to_draw));
      EXPECT_TRUE(region.PointInSet(point_to_draw.head(2)));
    }
    Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 20);

    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("Inflated Clique", points, 2.0, Rgba(0, 1, 0));

    MaybePauseForUser();
  }
}
// Attempting to inflate a linesegment that is in collision. This is a hard case
// where the convex hull of the points lies in a lower dimensional space and is
// in collision. We expect to get a nonempty region that contains one of the
// points. Furthermore, the face that separates the found collision in the
// convex hull is expected to be perpendicular to the linesegment.
TEST_F(FastCliqueInflationTestFixture, TwoPointCliqueCornerCase) {
  Eigen::Matrix3Xd clique(3, 2);
  double xw;
  xw = 0.9;
  // clang-format off
    clique << -xw,  0,
                0, xw,
                0,  0;
  // clang-format on
  for (int pt_to_draw = 0; pt_to_draw < clique.cols(); ++pt_to_draw) {
    Eigen::Vector3d point_to_draw = Eigen::Vector3d::Zero();
    std::string path = fmt::format("clique_pt/{}", pt_to_draw);
    options.meshcat->SetObject(path, Sphere(0.04),
                               geometry::Rgba(1, 0, 0.0, 1.0));
    point_to_draw.head(2) = clique.col(pt_to_draw);
    options.meshcat->SetTransform(path,
                                  math::RigidTransform<double>(point_to_draw));
  }
  HPolyhedron region = FastCliqueInflationFromUrdf(boxes_in_corners_urdf,
                                                   clique.topRows(2), options);
  EXPECT_EQ(region.ambient_dimension(), 2);
  // Confirm that we've found a "substantial" region.
  EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 0.15);
  Eigen::VectorXd dir(2);
  dir << 1 / sqrt(2), 1 / sqrt(2);
  Eigen::MatrixXd face_normals = region.A();
  double max_dotprod = (face_normals * dir).cwiseAbs().maxCoeff();
  // Check that the collision in the convex hull indeed produces a face in the
  // orthogonal complement to the affine space of the edge.
  EXPECT_GE(max_dotprod, 0.99);
  {
    EXPECT_TRUE(region.PointInSet(clique.col(0).head(2)));
    EXPECT_FALSE(region.PointInSet(clique.col(1).head(2)));
    Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 20);

    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("nflated Clique", points, 2.0, Rgba(0, 1, 0));

    MaybePauseForUser();
  }
}
// Verify that FastCliqueInflation throws if one of the clique points is in
// collision.
TEST_F(FastCliqueInflationTestFixture, CliqueInCollision) {
  Eigen::Matrix3Xd clique(3, 3);
  double xw;
  xw = 0.4;
  // clang-format off
    clique << 1, 0,  0,
              1, 0, xw,
              0,  0,  0;
  // clang-format on
  for (int pt_to_draw = 0; pt_to_draw < clique.cols(); ++pt_to_draw) {
    Eigen::Vector3d point_to_draw = Eigen::Vector3d::Zero();
    std::string path = fmt::format("clique_pt/{}", pt_to_draw);
    options.meshcat->SetObject(path, Sphere(0.04),
                               geometry::Rgba(1, 0, 0.0, 1.0));
    point_to_draw.head(2) = clique.col(pt_to_draw);
    options.meshcat->SetTransform(path,
                                  math::RigidTransform<double>(point_to_draw));
  }
  EXPECT_THROW(FastCliqueInflationFromUrdf(boxes_in_corners_urdf,
                                           clique.topRows(2), options),
               std::runtime_error);
}
}  // namespace
}  // namespace planning
}  // namespace drake