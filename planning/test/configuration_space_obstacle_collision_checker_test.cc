#include "drake/planning/configuration_space_obstacle_collision_checker.h"

#include <gtest/gtest.h>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/planning/linear_distance_and_interpolation_provider.h"
#include "drake/planning/robot_diagram_builder.h"
#include "drake/planning/test/planning_test_helpers.h"
#include "drake/planning/test_utilities/collision_checker_abstract_test_suite.h"
#include "drake/planning/scene_graph_collision_checker.h"

namespace drake {
namespace planning {
namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using drake::geometry::optimization::ConvexSet;
using drake::geometry::optimization::ConvexSets;
using drake::geometry::optimization::VPolytope;

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
const char boxes_in_corners[] = R"""(
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
)""";
GTEST_TEST(ConfigurationSpaceObstacleCollisionChecker, CheckConfigCollisionFree) {
  CollisionCheckerParams params;

  RobotDiagramBuilder<double> builder(0.0);
  params.robot_model_instances =
      builder.parser().AddModelsFromString(boxes_in_corners, "urdf");
  params.edge_step_size = 0.01;

  params.model = builder.Build();
  copyable_unique_ptr<CollisionChecker> checker_{
    std::make_unique<SceneGraphCollisionChecker>(std::move(params))
  };

  // Square obstacle in top right
  Eigen::MatrixXd obs1_pts(2, 4);
  obs1_pts << 1.7, 2.0, 1.7, 2.0,
              1.7, 1.7, 2.0, 2.0;
  VPolytope obs1(obs1_pts);

  // Square obstacle in bottom left
  Eigen::MatrixXd obs2_pts(2, 4);
  obs2_pts << -1.7, -2.0, -1.7, -2.0,
              -1,7, -1.7, -2.0, -2.0;
  VPolytope obs2(obs2_pts);

  ConvexSets cspace_obstacles;
  cspace_obstacles.emplace_back(obs1.Clone());
  cspace_obstacles.emplace_back(obs2.Clone());

  ConfigurationSpaceObstacleCollisionChecker checker(checker_, 
      cspace_obstacles);

  EXPECT_TRUE(checker.CheckConfigCollisionFree(Eigen::Vector2d{0, 0}));
  EXPECT_TRUE(checker.CheckConfigCollisionFree(Eigen::Vector2d{-1.9, 1.9}));
  EXPECT_TRUE(checker.CheckConfigCollisionFree(Eigen::Vector2d{1.9, -1.9}));

  // Test collisions with context obstacles
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{-1, 1}));
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{1, -1}));
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{-1, -1}));
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{1, 1}));

  // Test collisions with configuration obstacles
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{-1.9, -1.9}));
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{1.9, 1.9}));

  // Remove obstacles
  checker.SetConfigurationSpaceObstacles(ConvexSets{});

  // Context obstacles should still be in collisoni
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{-1, 1}));
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{1, -1}));
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{-1, -1}));
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{1, 1}));

  // Collisions with configuration obstacles should be gone
  EXPECT_TRUE(checker.CheckConfigCollisionFree(Eigen::Vector2d{-1.9, -1.9}));
  EXPECT_TRUE(checker.CheckConfigCollisionFree(Eigen::Vector2d{1.9, 1.9}));

  // Add obstacles back
  checker.SetConfigurationSpaceObstacles(cspace_obstacles);
  
  // Test collisions with context obstacles
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{-1, 1}));
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{1, -1}));
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{-1, -1}));
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{1, 1}));

  // Test collisions with configuration obstacles
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{-1.9, -1.9}));
  EXPECT_FALSE(checker.CheckConfigCollisionFree(Eigen::Vector2d{1.9, 1.9}));
}

}
}
}