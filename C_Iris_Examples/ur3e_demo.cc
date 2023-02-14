#include <iostream>
#include <limits>

#include "drake/common/find_resource.h"
#include "drake/geometry/collision_filter_declaration.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/meshcat_visualizer_params.h"
#include "drake/geometry/optimization/dev/cspace_free_polytope.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/multibody_tree_indexes.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace geometry {
namespace optimization {
const double kInf = std::numeric_limits<double>::infinity();

class UrDiagram {
 public:
  UrDiagram(int num_ur, bool weld_wrist, bool add_shelf, bool add_gripper)
      : meshcat_{std::make_shared<geometry::Meshcat>()} {
    systems::DiagramBuilder<double> builder;
    auto [plant, sg] = multibody::AddMultibodyPlantSceneGraph(&builder, 0.);
    plant_ = &plant;
    scene_graph_ = &sg;

    multibody::Parser parser(plant_);
    const std::string ur_file_name =
        fmt::format("drake/manipulation/models/ur3e/{}",
                    (weld_wrist ? "ur3e_cylinder_weld_wrist.urdf"
                                : "ur3e_cylinder_revolute_wrist.urdf"));
    const std::string ur_file_path = FindResourceOrThrow(ur_file_name);
    for (int ur_count = 0; ur_count < num_ur; ++ur_count) {
      const auto ur_instance =
          parser.AddModelFromFile(ur_file_path, fmt::format("ur{}", ur_count));
      plant_->WeldFrames(
          plant_->world_frame(),
          plant_->GetFrameByName("ur_base_link", ur_instance),
          math::RigidTransformd(Eigen::Vector3d(0, ur_count * 0.6, 0)));
      ur_instances_.push_back(ur_instance);
      if (add_gripper) {
        const std::string gripper_file_path = FindResourceOrThrow(
            "drake/manipulation/models/wsg_50_description/sdf/"
            "schunk_wsg_50_welded_fingers.sdf");
        const auto gripper_instance = parser.AddModelFromFile(
            gripper_file_path, fmt::format("schunk{}", ur_count));
        plant_->WeldFrames(
            plant_->GetBodyByName("ur_ee_link", ur_instance).body_frame(),
            plant_->GetBodyByName("body", gripper_instance).body_frame(),
            math::RigidTransform(math::RollPitchYawd(0, 0, -M_PI / 2),
                                 Eigen::Vector3d(0.06, 0, 0)));
      }
    }
    if (add_shelf) {
      const std::string shelf_file_path = FindResourceOrThrow(
          "drake/geometry/optimization/dev/models/shelves.sdf");
      const auto shelf_instance =
          parser.AddModelFromFile(shelf_file_path, "shelves");
      const auto& shelf_frame =
          plant_->GetFrameByName("shelves_body", shelf_instance);
      const math::RigidTransformd X_WShelf(Eigen::Vector3d(0.5, 0, 0.4));
      plant_->WeldFrames(plant_->world_frame(), shelf_frame, X_WShelf);
    }
    plant_->Finalize();

    const auto& inspector = scene_graph_->model_inspector();
    for (const auto& ur_instance : ur_instances_) {
      geometry::GeometrySet ur_geometries;
      for (const auto& body_index : plant_->GetBodyIndices(ur_instance)) {
        const auto body_geometries =
            inspector.GetGeometries(plant_->GetBodyFrameIdOrThrow(body_index));
        for (const auto& body_geometry : body_geometries) {
          ur_geometries.Add(body_geometry);
        }
      }
      scene_graph_->collision_filter_manager().Apply(
          geometry::CollisionFilterDeclaration().ExcludeWithin(ur_geometries));
    }

    geometry::MeshcatVisualizerParams meshcat_params{};
    meshcat_params.role = geometry::Role::kProximity;
    visualizer_ = &geometry::MeshcatVisualizer<double>::AddToBuilder(
        &builder, *scene_graph_, meshcat_, meshcat_params);
    std::cout << meshcat_->web_url() << "\n";
    diagram_ = builder.Build();
  }

  const systems::Diagram<double>& diagram() const { return *diagram_; }

  const multibody::MultibodyPlant<double>& plant() const { return *plant_; }

  const geometry::SceneGraph<double>& scene_graph() const {
    return *scene_graph_;
  }

  const std::vector<multibody::ModelInstanceIndex>& ur_instances() const {
    return ur_instances_;
  }

 private:
  std::unique_ptr<systems::Diagram<double>> diagram_;
  multibody::MultibodyPlant<double>* plant_;
  geometry::SceneGraph<double>* scene_graph_;
  std::shared_ptr<geometry::Meshcat> meshcat_;
  geometry::MeshcatVisualizer<double>* visualizer_;
  std::vector<multibody::ModelInstanceIndex> ur_instances_;
  std::vector<multibody::ModelInstanceIndex> gripper_instances_;
};

void SetupDualArmCspacePolytope(bool weld_wrist, Eigen::MatrixXd* C,
                                Eigen::VectorXd* d) {
  if (weld_wrist) {
    C->resize(22, 10);
    d->resize(22);
    // clang-format off
  *C << 1.5, 0.1, 0.1, -0.1, 0.2, 0.2, 0.3, 0, 0, 0,
        -2., -0.2, 0.5, 0.1, 0.3, 0., 0., 0., 0., 0.1,
        -0.2, 3.4, 0.0, 0.3, -0.2, 0.5, -0.3, 0.0, 0.0, 0.1,
        0.3, -3.2, -0.4, 0.4, -0.2, -0.3, 0.2, -0.1, 0.5, 0.9,
        1.2, -0.1, 3.3, -0.5, 0.4, 0.2, 1.4, -0.2, 0.5, -1.5,
        0.4, -2.1, -4.5, 0.2, -0.3, -0.5, 0.3, 0.2, 0.9, 1.4,
        0.5, -0.1, 2.1, 4.3, -0.2, 0.3, 0.1, -0.5, 0.5, 0.2,
        -1.5, -0.1, 0.8, -3.2, -2.1, -0.3, 1.4, 0.1, -1.5, 0.1,
        0.4, 0.1, -0.2, 0.4, 4.5, 0.6, 0.8, 1.2, -0.1, 2.3,
        0.1, 0.3, 0.6, -0.2, -4.4, 0.5, 0.6, -0.8, 0.3, 1.5,
        -0.3, 1.8, 1.2, -0.6, 0.4, 3.4, 0.8, 1.5, -0.2, 0.7,
        0.4, 2.1, 0.5, -0.2, 1.2, -3.8, 1.7, -0.2, 0.3, 1.5,
        -0.5, -0.2, 0.3, 0.5, -0.1, 0.8, 3.5, -0.8, 0.4, 0.2,
        -0.4, -0.2, -0.8, -0.3, -0.5, 0.1, -4.2, -0.4, 0.5, 0.1,
        -0.5, -0.3, 0.3, 0.2, 0.7, 0.6, 0.4, 3.1, -0.2, 0.5,
        -0.8, 0.1, 0.4, 0.5, 0.1, -0.1, 1.0, -3.2, -0.3, 0.5,
        3.2, 0.4, 1.5, 0.1, 0.2, 0.6, 1.5, 0.2, 0.7, 0.8,
        1.5, 0.3, 0.7, -0.2, -2.1, -2.5, 0.5, 0.1, -1.5, 0.7,
        2.6, 0.7, 1.5, 0.2, 1.9, 0.4, 0.5, 1.6, -0.2, -2.5,
        -0.4, -0.2, 1.9, 2.1, 0.4, -1.5, 0.5, 0.8, 1.5, 0.7,
        2.1, 2.5, 0.2, 1.4, -0.2, 1.8, -0.7, 1.2, 3.1, 0.4,
        -1.2, -0.5, -0.9, -1.5, 0.4, 0.3, -2.1, 0.6, 0.7, 1.5;
    // clang-format on
    *d << 0.5, 0.1, 0.3, 0.8, 0.2, 1.3, 0.4, 1.5, 0.1, 0.2, 0.3, 0.2, 0.1, 0.4,
        0.2, 0.2, 0.1, 0.1, 0.2, 0.3, 0.1, 0.2;
  } else {
    C->resize(13, 12);
    d->resize(13);
    // Generate n+1 vectors in an n-dimensional space, such that the degrees
    // between any pair of these vectors are the same, and cone of these vectors
    // cover the entire n-dimensional space.
    const Eigen::MatrixXd S = Eigen::MatrixXd::Identity(13, 13) -
                              Eigen::MatrixXd::Ones(13, 13) / 13.0;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    C->topRows<13>() = es.eigenvectors().rightCols<12>();

    *d << 0.5, 0.1, 0.3, 0.8, 0.2, 1.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.1;
    C->conservativeResize(14, 12);
    d->conservativeResize(14);
    C->bottomRows<1>() << 0.5, 0.2, -0.3, 0.1, 0.6, -0.5, -0.4, -0.9, -1.1, 1.2, 0.3, -0.2;
    (*d)(13) = 0.7;
  }
  HPolyhedron polyhedron(*C, *d);
  DRAKE_DEMAND(polyhedron.IsBounded());
}

Eigen::VectorXd FindUrShelfPosture(
    const multibody::MultibodyPlant<double>& plant,
    systems::Context<double>* plant_context) {
  multibody::InverseKinematics ik(plant, plant_context);
  const auto& ee_frame = plant.GetFrameByName("ur_ee_link");
  const auto& shelf = plant.GetFrameByName("shelves_body");
  ik.AddPositionConstraint(ee_frame, Eigen::Vector3d::Zero(), shelf,
                           Eigen::Vector3d(-0.25, -0.2, -0.2),
                           Eigen::Vector3d(0., 0.2, 0.2));
  ik.AddMinimumDistanceConstraint(0.01);

  Eigen::Matrix<double, 5, 1> q_init;
  q_init << 0, 0.1, 0, 0, 0;
  ik.get_mutable_prog()->SetInitialGuess(ik.q(), q_init);
  const auto result = solvers::Solve(ik.prog());
  if (!result.is_success()) {
    drake::log()->warn("Cannot find the posture\n");
  }
  return result.GetSolution(ik.q());
}

void SetupUrShelfCspacePolytope(const Eigen::Matrix<double, 5, 1>& s_init,
                                Eigen::MatrixXd* C, Eigen::VectorXd* d) {
  C->resize(12, 5);
  d->resize(12);
  // clang-format off
  *C << 1.5, 0.1, 0.2, -0.1, -0.2,
        -2.1, 0.2, -0.1, 0.4, -0.3,
        0.1, 2.5, 0.3, -0.3, 0.2,
        -0.3, -2.1, 0.2, 0.3, -0.4,
        0.1, 0.2, 3.2, -0.1, 0.3,
        0.2, -0.1, -2.5, 0.2, 0.3,
        0.2, 0.1, 1.2, 3.2, 0.2,
        -0.2, 0.3, -0.4, -4.1, 0.4,
        0.4, 0.2, 0.5, -0.3, 3.2,
        -0.1, -0.5, 0.2, -0.5, -2.9,
        0.1, 1.2, 0.4, 1.5, -0.4,
        0.2, -0.3, -1.5, 0.2, 2.1;
  // clang-format on
  *d << 0.1, 0.05, 0.1, 0.2, 0.05, 0.15, 0.2, 0.1, 0.4, 0.1, 0.2, 0.2;
  *d += (*C) * s_init;

  HPolyhedron hpolyhedron(*C, *d);
  DRAKE_DEMAND(hpolyhedron.IsBounded());
}

void SearchDualArmCspacePolytope(bool weld_wrist, bool with_gripper) {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  const UrDiagram ur_diagram{2, weld_wrist, false /* add_shelf */,
                             with_gripper};
  auto diagram_context = ur_diagram.diagram().CreateDefaultContext();
  auto& plant_context =
      ur_diagram.plant().GetMyMutableContextFromRoot(diagram_context.get());
  ur_diagram.plant().SetPositions(
      &plant_context,
      Eigen::VectorXd::Zero(ur_diagram.plant().num_positions()));
  ur_diagram.diagram().ForcedPublish(*diagram_context);
  // std::cout << "Type to continue";
  // std::string command;
  // std::cin >> command;
  // return;
  Eigen::MatrixXd C_init;
  Eigen::VectorXd d_init;
  SetupDualArmCspacePolytope(weld_wrist, &C_init, &d_init);

  const Eigen::VectorXd q_star =
      Eigen::VectorXd::Zero(ur_diagram.plant().num_positions());
  CspaceFreePolytope::Options cspace_free_polytope_options;
  cspace_free_polytope_options.with_cross_y = false;
  const CspaceFreePolytope cspace_free_polytope(
      &(ur_diagram.plant()), &(ur_diagram.scene_graph()),
      SeparatingPlaneOrder::kAffine, q_star, cspace_free_polytope_options);

  CspaceFreePolytope::BinarySearchOptions binary_search_options;
  binary_search_options.scale_max = 0.5;
  binary_search_options.scale_min = 0.01;
  binary_search_options.max_iter = 6;
  binary_search_options.find_lagrangian_options.verbose = true;
  binary_search_options.find_lagrangian_options.num_threads = 1;
  binary_search_options.find_lagrangian_options.solver_options =
      solvers::SolverOptions();
  binary_search_options.find_lagrangian_options.solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 0);

  const Eigen::VectorXd s_star =
      cspace_free_polytope.rational_forward_kin().ComputeSValue(q_star, q_star);

  const CspaceFreePolytope::IgnoredCollisionPairs ignored_collision_pairs{};
  const auto binary_search_result = cspace_free_polytope.BinarySearch(
      ignored_collision_pairs, C_init, d_init, s_star, binary_search_options);
}

void SearchUrShelfCspacePolytope(bool weld_wrist, bool add_gripper) {
  const UrDiagram ur_diagram{1, weld_wrist, true /* add_shelf */, add_gripper};
  auto diagram_context = ur_diagram.diagram().CreateDefaultContext();
  auto& plant_context =
      ur_diagram.plant().GetMyMutableContextFromRoot(diagram_context.get());
  const Eigen::Matrix<double, 5, 1> q_init =
      FindUrShelfPosture(ur_diagram.plant(), &plant_context);
  ur_diagram.plant().SetPositions(&plant_context, q_init);
  ur_diagram.diagram().ForcedPublish(*diagram_context);
  // std::cout << "Type to continue";
  // std::string command;
  // std::cin >> command;
  const Eigen::VectorXd q_star = Eigen::VectorXd::Zero(5);
  CspaceFreePolytope::Options cspace_free_polytope_options;
  cspace_free_polytope_options.with_cross_y = false;
  const CspaceFreePolytope cspace_free_polytope(
      &(ur_diagram.plant()), &(ur_diagram.scene_graph()),
      SeparatingPlaneOrder::kAffine, q_star, cspace_free_polytope_options);

  const Eigen::VectorXd s_init =
      cspace_free_polytope.rational_forward_kin().ComputeSValue(q_init, q_star);

  Eigen::MatrixXd C_init;
  Eigen::VectorXd d_init;
  SetupUrShelfCspacePolytope(s_init, &C_init, &d_init);

  CspaceFreePolytope::BinarySearchOptions binary_search_options;
  binary_search_options.scale_max = 0.5;
  binary_search_options.scale_min = 0.3;
  binary_search_options.max_iter = 5;
  binary_search_options.find_lagrangian_options.verbose = false;
  binary_search_options.find_lagrangian_options.num_threads = -1;

  const CspaceFreePolytope::IgnoredCollisionPairs ignored_collision_pairs{};
  const auto binary_search_result = cspace_free_polytope.BinarySearch(
      ignored_collision_pairs, C_init, d_init, s_init, binary_search_options);

  CspaceFreePolytope::BilinearAlternationOptions bilinear_alternation_options;
  bilinear_alternation_options.find_lagrangian_options.num_threads = -1;
  bilinear_alternation_options.convergence_tol = 1E-12;
  bilinear_alternation_options.find_polytope_options.s_inner_pts.emplace(
      s_init);
  const auto bilinear_alternation_result =
      cspace_free_polytope.SearchWithBilinearAlternation(
          ignored_collision_pairs, binary_search_result->C,
          binary_search_result->d, bilinear_alternation_options);
}

int DoMain() {
  const bool with_gripper = true;
  const bool weld_wrist = false;
  SearchDualArmCspacePolytope(weld_wrist, with_gripper);
  // SearchUrShelfCspacePolytope(weld_wrist, with_gripper);
  return 0;
}
}  // namespace optimization
}  // namespace geometry
}  // namespace drake

int main() { return drake::geometry::optimization::DoMain(); }
