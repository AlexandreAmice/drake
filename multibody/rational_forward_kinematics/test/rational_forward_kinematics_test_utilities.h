#pragma once

#include <array>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/rational_forward_kinematics/convex_geometry.h"

namespace drake {
namespace multibody {

std::unique_ptr<drake::multibody::MultibodyPlant<double>> ConstructIiwaPlant(
    const std::string& iiwa_sdf_name, bool finalize);

Eigen::Matrix<double, 3, 8> GenerateBoxVertices(
    const Eigen::Vector3d& size, const drake::math::RigidTransformd& pose);

std::unique_ptr<drake::multibody::MultibodyPlant<double>>
ConstructDualArmIiwaPlant(
    const std::string& iiwa_sdf_name, const drake::math::RigidTransformd& X_WL,
    const drake::math::RigidTransformd& X_WR,
    drake::multibody::ModelInstanceIndex* left_iiwa_instance,
    drake::multibody::ModelInstanceIndex* right_iiwa_instance);

class IiwaTest : public ::testing::Test {
 public:
  IiwaTest();

  void AddBox(const math::RigidTransform<double>& X_BG,
              const Eigen::Vector3d& box_size, BodyIndex body_index,
              const std::string& name,
              std::vector<std::unique_ptr<const ConvexPolytope>>* geometries);

 protected:
  std::unique_ptr<drake::multibody::MultibodyPlant<double>> iiwa_;
  std::unique_ptr<drake::geometry::SceneGraph<double>> scene_graph_;
  const drake::multibody::internal::MultibodyTree<double>& iiwa_tree_;
  const drake::multibody::BodyIndex world_;
  std::array<drake::multibody::BodyIndex, 8> iiwa_link_;
  std::array<drake::multibody::internal::MobilizerIndex, 8> iiwa_joint_;
};

/**
 * @param X_7S The transformation from schunk frame to iiwa link 7.
 * @note the plant is not finalized.
 */
void AddIiwaWithSchunk(const drake::math::RigidTransformd& X_7S,
                       drake::multibody::MultibodyPlant<double>* plant);

/**
 * @param X_WL the pose of the left IIWA base in the world frame.
 * @param X_WR the pose of the right IIWA base in the world frame.
 */
void AddDualArmIiwa(const drake::math::RigidTransformd& X_WL,
                    const drake::math::RigidTransformd& X_WR,
                    const drake::math::RigidTransformd& X_7S,
                    drake::multibody::MultibodyPlant<double>* plant,
                    drake::multibody::ModelInstanceIndex* left_iiwa_instance,
                    drake::multibody::ModelInstanceIndex* right_iiwa_instance);

}  // namespace multibody
}  // namespace drake
