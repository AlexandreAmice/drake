#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <unordered_map>
#include <utility>

#include "drake/multibody/tree/revolute_mobilizer.h"

namespace drake {
namespace multibody {
namespace internal {
using drake::multibody::BodyIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::internal::BodyTopology;
using drake::multibody::internal::GetInternalTree;
using drake::multibody::internal::Mobilizer;
using drake::multibody::internal::MobilizerIndex;
using drake::multibody::internal::MultibodyTree;
using drake::multibody::internal::RevoluteMobilizer;
using drake::multibody::internal::WeldMobilizer;

void AddChildrenToReshuffledBody(const MultibodyPlant<double>& plant,
                                 ReshuffledBody* body,
                                 std::unordered_set<BodyIndex>* visited) {
  const MultibodyTree<double>& tree = GetInternalTree(plant);
  const BodyTopology& body_topology =
      tree.get_topology().get_body(body->body_index);

  if (body_topology.parent_body.is_valid()) {
    auto it = visited->find(body_topology.parent_body);
    if (it == visited->end()) {
      auto child = std::make_unique<ReshuffledBody>(
          body_topology.parent_body, body,
          &(tree.get_mobilizer(body_topology.inboard_mobilizer)));
      body->children.push_back(std::move(child));
      visited->emplace_hint(it, body_topology.parent_body);
    }
  }
  for (BodyIndex body_index(0); body_index < plant.num_bodies(); ++body_index) {
    const BodyTopology& body_topology_i =
        tree.get_topology().get_body(body_index);
    if (body_topology_i.parent_body.is_valid() &&
        body_topology_i.parent_body == body->body_index) {
      auto it = visited->find(body_index);
      if (it == visited->end()) {
        body->children.emplace_back(new ReshuffledBody(
            body_index, body,
            &(tree.get_mobilizer(body_topology_i.inboard_mobilizer))));
        visited->emplace_hint(it, body_index);
      }
    }
  }
}

void ReshuffleKinematicsTree(const MultibodyPlant<double>& plant,
                             ReshuffledBody* root) {
  DRAKE_DEMAND(root->parent == nullptr);
  DRAKE_DEMAND(root->mobilizer == nullptr);
  root->children.clear();
  std::unordered_set<BodyIndex> visited;
  visited.emplace(root->body_index);
  std::queue<ReshuffledBody*> queue_bodies;
  queue_bodies.push(root);
  while (!queue_bodies.empty()) {
    ReshuffledBody* reshuffled_body = queue_bodies.front();
    queue_bodies.pop();
    AddChildrenToReshuffledBody(plant, reshuffled_body, &visited);
    if (!reshuffled_body->children.empty()) {
      for (int i = 0; i < static_cast<int>(reshuffled_body->children.size());
           ++i) {
        queue_bodies.push(reshuffled_body->children[i].get());
      }
    }
  }
}

struct BodyOnPath {
  BodyOnPath(BodyIndex m_index, int m_distance_to_start, BodyOnPath* m_parent)
      : index(m_index),
        distance_to_start(m_distance_to_start),
        parent(m_parent) {}
  BodyIndex index;
  int distance_to_start;
  BodyOnPath* parent;
};

std::vector<BodyIndex> FindShortestPath(const MultibodyPlant<double>& plant,
                                        BodyIndex start, BodyIndex end) {
  DRAKE_ASSERT(start.is_valid() && end.is_valid());
  const MultibodyTree<double>& tree = internal::GetInternalTree(plant);
  // Do a breadth first search in the tree.
  std::unordered_map<BodyIndex, std::unique_ptr<BodyOnPath>> visited_bodies;
  BodyIndex start_copy = start;
  visited_bodies.emplace(std::make_pair(
      std::move(start_copy), std::make_unique<BodyOnPath>(start, 0, nullptr)));
  std::queue<BodyOnPath*> queue_bodies;
  queue_bodies.push(visited_bodies[start].get());
  BodyOnPath* queue_front = queue_bodies.front();
  while (queue_front->index != end) {
    queue_bodies.pop();
    const BodyTopology& body_node =
        tree.get_topology().get_body(queue_front->index);
    if (body_node.parent_body.is_valid()) {
      BodyIndex parent = body_node.parent_body;
      visited_bodies.emplace(std::make_pair(
          std::move(parent),
          std::make_unique<BodyOnPath>(body_node.parent_body,
                                       queue_front->distance_to_start + 1,
                                       queue_front)));
      queue_bodies.emplace(visited_bodies[body_node.parent_body].get());
    }
    for (BodyIndex child : body_node.child_bodies) {
      if (child.is_valid()) {
        BodyIndex child_copy = child;
        visited_bodies.emplace(std::make_pair(
            std::move(child_copy),
            std::make_unique<BodyOnPath>(
                child, queue_front->distance_to_start + 1, queue_front)));
        queue_bodies.emplace(visited_bodies[child].get());
      }
    }
    queue_front = queue_bodies.front();
  }
  // Backup the path
  std::vector<BodyIndex> path;
  BodyOnPath* path_body = queue_front;
  while (path_body->index != start) {
    path.push_back(path_body->index);
    path_body = path_body->parent;
  }
  path.push_back(start);
  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<MobilizerIndex> FindMobilizersOnShortestPath(
    const MultibodyPlant<double>& plant, BodyIndex start, BodyIndex end) {
  const std::vector<BodyIndex> path = FindShortestPath(plant, start, end);
  std::vector<MobilizerIndex> mobilizers_on_path;
  mobilizers_on_path.reserve(path.size() - 1);
  const MultibodyTree<double>& tree = GetInternalTree(plant);
  for (int i = 0; i < static_cast<int>(path.size()) - 1; ++i) {
    const BodyTopology& body_topology = tree.get_topology().get_body(path[i]);
    if (body_topology.parent_body.is_valid() &&
        body_topology.parent_body == path[i + 1]) {
      mobilizers_on_path.push_back(body_topology.inboard_mobilizer);
    } else {
      mobilizers_on_path.push_back(
          tree.get_topology().get_body(path[i + 1]).inboard_mobilizer);
    }
  }
  return mobilizers_on_path;
}

BodyIndex FindBodyInTheMiddleOfChain(const MultibodyPlant<double>& plant,
                                     BodyIndex start, BodyIndex end) {
  const std::vector<BodyIndex> path = FindShortestPath(plant, start, end);

  // path_only_revolute goes from start to end. If path[i] and path[i-1] is
  // connected through a prismatic joint, then path[i] is not included in
  // path_only_revolute.
  std::vector<BodyIndex> path_only_revolute;
  path_only_revolute.reserve(path.size());
  path_only_revolute.push_back(start);
  const MultibodyTree<double>& tree = GetInternalTree(plant);
  for (int i = 0; i < static_cast<int>(path.size()) - 1; ++i) {
    const BodyTopology& body_topology = tree.get_topology().get_body(path[i]);
    MobilizerIndex mobilizer_index;
    if (body_topology.parent_body.is_valid() &&
        body_topology.parent_body == path[i + 1]) {
      mobilizer_index = body_topology.inboard_mobilizer;
    } else {
      mobilizer_index =
          tree.get_topology().get_body(path[i + 1]).inboard_mobilizer;
    }
    const Mobilizer<double>& mobilizer = tree.get_mobilizer(mobilizer_index);
    if (dynamic_cast<const RevoluteMobilizer<double>*>(&mobilizer) != nullptr) {
      path_only_revolute.push_back(path[i + 1]);
    } else if (dynamic_cast<const WeldMobilizer<double>*>(&mobilizer) !=
               nullptr) {
      continue;
    } else {
      throw std::invalid_argument(
          "FindBodyInTheMiddleOfChain: only allow revolute or weld mobilizer "
          "along the path.");
    }
  }

  return path_only_revolute[(path_only_revolute.size() / 2)];
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake