#include <string>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace manipulation {
namespace {
GTEST_TEST(Ur3ePrimitiveCollision, Test) {
  multibody::MultibodyPlant<double> plant(0.0);
  const std::string model_file(
      FindResourceOrThrow("drake/manipulation/models/ur3e_description/urdf/"
                          "ur3e_robot_primitive_collision.urdf"));
  multibody::Parser parser(&plant);
  parser.AddModelFromFile(model_file);
  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("ur_base_link").body_frame());
  plant.Finalize();
  EXPECT_EQ(plant.num_actuated_dofs(), 6);
  EXPECT_EQ(plant.num_positions(), 6);
}

GTEST_TEST(TestDiagram, Test) {
  systems::DiagramBuilder<double> builder;
  auto [plant, sg] = multibody::AddMultibodyPlantSceneGraph(&builder, 0.);
  multibody::Parser parser(&plant);
  const std::string model_file(
      FindResourceOrThrow("drake/manipulation/models/ur3e_description/urdf/"
                          "ur3e_robot_primitive_collision.urdf"));
  parser.AddModelFromFile(model_file, "ur3e");
  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("ur_base_link").body_frame());
  plant.Finalize();
  geometry::MeshcatVisualizerParams meshcat_params{};
  meshcat_params.role = geometry::Role::kProximity;
  auto meshcat = std::make_shared<geometry::Meshcat>();
  geometry::MeshcatVisualizer<double>::AddToBuilder(&builder, sg, meshcat,
                                                    meshcat_params);
  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();
  diagram->Publish(*diagram_context);
}

}  // namespace
}  // namespace manipulation
}  // namespace drake
