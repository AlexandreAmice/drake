#include <iostream>
#include <limits>

#include "drake/common/find_resource.h"
#include "drake/geometry/collision_filter_declaration.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/meshcat_visualizer_params.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/polytope_cover.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
const double kInf = std::numeric_limits<double>::infinity();

class IiwaDiagram {
 public:
  IiwaDiagram() : meshcat_{std::make_shared<geometry::Meshcat>()} {
    systems::DiagramBuilder<double> builder;
    auto [plant, sg] = AddMultibodyPlantSceneGraph(&builder, 0.);
    plant_ = &plant;
    scene_graph_ = &sg;

    multibody::Parser parser(plant_);
    const std::string iiwa_file_path = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/sdf/"
        "iiwa14_coarse_collision.sdf");
    const auto iiwa_instance = parser.AddModelFromFile(iiwa_file_path, "iiwa");
    plant_->WeldFrames(plant_->world_frame(),
                       plant_->GetFrameByName("iiwa_link_0"));

    const std::string schunk_file_path = FindResourceOrThrow(
        "drake/manipulation/models/wsg_50_description/sdf/"
        "schunk_wsg_50_welded_fingers.sdf");

    const Frame<double>& link7 =
        plant_->GetFrameByName("iiwa_link_7", iiwa_instance);
    const math::RigidTransformd X_7G(
        math::RollPitchYaw<double>(M_PI_2, 0, M_PI_2),
        Eigen::Vector3d(0, 0, 0.114));
    const auto wsg_instance =
        parser.AddModelFromFile(schunk_file_path, "gripper");
    const auto& schunk_frame = plant_->GetFrameByName("body", wsg_instance);
    plant_->WeldFrames(link7, schunk_frame, X_7G);
    // SceneGraph should ignore the collision between any geometries on the
    // gripper, and between the gripper and link 6
    geometry::GeometrySet gripper_link6_geometries;
    auto add_gripper_geometries =
        [this, wsg_instance,
         &gripper_link6_geometries](const std::string& body_name) {
          const geometry::FrameId frame_id = plant_->GetBodyFrameIdOrThrow(
              plant_->GetBodyByName(body_name, wsg_instance).index());
          gripper_link6_geometries.Add(frame_id);
        };
    add_gripper_geometries("body");
    add_gripper_geometries("left_finger");
    add_gripper_geometries("right_finger");

    const geometry::FrameId link_6_frame_id = plant_->GetBodyFrameIdOrThrow(
        plant_->GetBodyByName("iiwa_link_6", iiwa_instance).index());
    const auto& inspector = scene_graph_->model_inspector();
    const std::vector<geometry::GeometryId> link_6_geometries =
        inspector.GetGeometries(link_6_frame_id, geometry::Role::kProximity);
    for (const auto geometry : link_6_geometries) {
      gripper_link6_geometries.Add(geometry);
    }

    scene_graph_->collision_filter_manager().Apply(
        geometry::CollisionFilterDeclaration().ExcludeWithin(
            gripper_link6_geometries));

    // SceneGraph should ignore collision on the IIWA.
    std::vector<geometry::GeometryId> iiwa_geometry_ids;
    for (const auto& body_index : plant_->GetBodyIndices(iiwa_instance)) {
      const auto body_geometry_ids =
          plant_->GetCollisionGeometriesForBody(plant_->get_body(body_index));
      iiwa_geometry_ids.insert(iiwa_geometry_ids.end(),
                               body_geometry_ids.begin(),
                               body_geometry_ids.end());
    }
    scene_graph_->collision_filter_manager().Apply(
        geometry::CollisionFilterDeclaration().ExcludeWithin(
            geometry::GeometrySet(iiwa_geometry_ids)));

    const std::string shelf_file_path =
        FindResourceOrThrow("drake/sos_iris_certifier/shelves.sdf");
    const auto shelf_instance =
        parser.AddModelFromFile(shelf_file_path, "shelves");
    const auto& shelf_frame =
        plant_->GetFrameByName("shelves_body", shelf_instance);
    const math::RigidTransformd X_WShelf(Eigen::Vector3d(0.8, 0, 0.4));
    plant_->WeldFrames(plant_->world_frame(), shelf_frame, X_WShelf);

    // Add a sphere as an obstacle.
    math::RigidTransformd X_WSphere = X_WShelf;
    X_WSphere.set_translation(X_WShelf.translation() +
                              Eigen::Vector3d(-0.2, 0.3, 0.1));
    plant_->RegisterCollisionGeometry(plant_->world_body(), X_WSphere,
                                      geometry::Sphere(0.1), "world_sphere",
                                      CoulombFriction<double>());

    // Add a capsule as an obstacle.
    math::RigidTransformd X_WCapsule(Eigen::Vector3d(0.5, -0.3, 0.3));
    plant_->RegisterCollisionGeometry(
        plant_->world_body(), X_WCapsule, geometry::Capsule(0.03, 0.4),
        "world_capsule", CoulombFriction<double>());

    plant_->Finalize();

    geometry::MeshcatVisualizerParams meshcat_params{};
    meshcat_params.role = geometry::Role::kProximity;
    visualizer_ = &geometry::MeshcatVisualizer<double>::AddToBuilder(
        &builder, *scene_graph_, meshcat_, meshcat_params);
    diagram_ = builder.Build();
  }

  const systems::Diagram<double>& diagram() const { return *diagram_; }

  const multibody::MultibodyPlant<double>& plant() const { return *plant_; }

  const geometry::SceneGraph<double>& scene_graph() const {
    return *scene_graph_;
  }

 private:
  std::unique_ptr<systems::Diagram<double>> diagram_;
  multibody::MultibodyPlant<double>* plant_;
  geometry::SceneGraph<double>* scene_graph_;
  std::shared_ptr<geometry::Meshcat> meshcat_;
  geometry::MeshcatVisualizer<double>* visualizer_;
};

Eigen::VectorXd FindInitialPosture(const MultibodyPlant<double>& plant,
                                   systems::Context<double>* plant_context) {
  InverseKinematics ik(plant, plant_context);
  const auto& link7 = plant.GetFrameByName("iiwa_link_7");
  const auto& shelf = plant.GetFrameByName("shelves_body");
  ik.AddPositionConstraint(link7, Eigen::Vector3d::Zero(), shelf,
                           Eigen::Vector3d(-0.2, -0.2, -0.2),
                           Eigen::Vector3d(-.0, 0.2, 0.2));
  ik.AddMinimumDistanceConstraint(0.035);

  Eigen::Matrix<double, 7, 1> q_init;
  q_init << 0.1, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2;
  ik.get_mutable_prog()->SetInitialGuess(ik.q(), q_init);
  const auto result = solvers::Solve(ik.prog());
  if (!result.is_success()) {
    drake::log()->warn("Cannot find the posture\n");
  }
  return result.GetSolution(ik.q());
}

void BuildCandidateCspacePolytope(const Eigen::VectorXd q_free,
                                  Eigen::MatrixXd* C, Eigen::VectorXd* d) {
  const int C_rows = 23;
  C->resize(C_rows, 7);
  // Create arbitrary polytope normals.
  // clang-format off
  (*C) << 0.5, 0.3, 0.2, -0.1, -1, 0, 0.5,
          -0.1, 0.4, 0.2, 0.1, 0.5, -0.2, 0.3,
          0.4, 1.2, -0.3, 0.2, 0.1, 0.4, 0.5,
          -0.5, -2, -1.5, 0.3, 0.6, 0.1, -0.2,
          0.2, 0.1, -0.5, 0.3, 0.4, 1.4, 0.5,
          0.1, -0.5, 0.4, 1.5, -0.3, 0.2, 0.1,
          0.2, 0.3, 1.3, 0.2, -0.3, -0.5, -0.2,
          1.4, 0.1, -0.1, 0.2, -0.3, 0.1, 0.5,
          -1.1, 0.2, 0.3, -0.1, 0.5, 0.2, -0.1,
          0.2, -0.3, -1.2, 0.5, -0.3, 0.1, 0.3,
          0.2, -1.5, 0.1, 0.4, -0.3, -0.2, 0.6,
          0.1, 0.4, -0.2, 0.3, 0.9, -0.5, 0.8,
          -0.2, 0.3, -0.1, 0.8, -0.4, 0.2, 1.4,
          0.1, -0.2, 0.2, -0.3, 1.2, -0.3, 0.1,
          0.3, -0.1, 0.2, 0.5, -0.3, -2.1, 1.2,
          0.4, -0.3, 1.5, -0.3, 1.8, -0.1, 0.4,
          1.2, -0.3, 0.4, 0.8, 1.2, -0.4, -0.8,
          0.4, -0.2, 0.5, 1.4, 0.7, -0.2, -0.9,
          -0.1, 0.4, -0.2, 0.3, 1.5, 0.1, -0.6,
          -0.1, -0.3, 0.2, 1.1, -1.2, 1.3, 2.1,
          0.1, -0.4, 0.2, 1.3, 1.2, 0.3, -1.1,
          0.1, -1.4, 0.2, 0.3, 0.2, 0.3, -0.7,
          -0.3, -0.5, 0.4, -1.5, -0.2, 1.3, -2.1;
  // clang-format on
  for (int i = 0; i < C_rows; ++i) {
    C->row(i).normalize();
  }
  *d = (*C) * (q_free / 2).array().tan().matrix();
  if (!geometry::optimization::HPolyhedron(*C, *d).IsBounded()) {
    throw std::runtime_error("C*t <= d is not bounded");
  }
}

void SearchCspacePolytope(const std::string& write_file) {
  // Ensure that we have the MOSEK license for the entire duration of this test,
  // so that we do not have to release and re-acquire the license for every
  // test.
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  const IiwaDiagram iiwa_diagram{};
  auto diagram_context = iiwa_diagram.diagram().CreateDefaultContext();
  auto& plant_context =
      iiwa_diagram.plant().GetMyMutableContextFromRoot(diagram_context.get());
  const auto q0 = FindInitialPosture(iiwa_diagram.plant(), &plant_context);
  iiwa_diagram.plant().SetPositions(&plant_context, q0);
  iiwa_diagram.diagram().Publish(*diagram_context);

  Eigen::MatrixXd C_init;
  Eigen::VectorXd d_init;
  BuildCandidateCspacePolytope(q0, &C_init, &d_init);

  const double separating_polytope_delta{0.001};
  const CspaceFreeRegion dut(
      iiwa_diagram.diagram(), &(iiwa_diagram.plant()),
      &(iiwa_diagram.scene_graph()), SeparatingPlaneOrder::kAffine,
      CspaceRegionType::kGenericPolytope, separating_polytope_delta);

  CspaceFreeRegion::FilteredCollisionPairs filtered_collision_pairs{};

  CspaceFreeRegion::BinarySearchOption binary_search_option{
      .epsilon_max = 0.01,
      .epsilon_min = 1E-6,
      .max_iters = 2,
      .compute_polytope_volume = true};
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, false);
  Eigen::VectorXd d_binary_search;
  Eigen::VectorXd q_star = Eigen::Matrix<double, 7, 1>::Zero();
  dut.CspacePolytopeBinarySearch(q_star, filtered_collision_pairs, C_init,
                                 d_init, binary_search_option, solver_options,
                                 q0, std::nullopt, &d_binary_search);
  CspaceFreeRegion::BilinearAlternationOption bilinear_alternation_option{
      .max_iters = 10,
      .convergence_tol = 0.001,
      .redundant_tighten = 0.5,
      .compute_polytope_volume = true};
  Eigen::MatrixXd C_final;
  Eigen::VectorXd d_final;
  Eigen::MatrixXd P_final;
  Eigen::VectorXd q_final;
  dut.CspacePolytopeBilinearAlternation(
      q_star, filtered_collision_pairs, C_init, d_binary_search,
      bilinear_alternation_option, solver_options, q0, std::nullopt, &C_final,
      &d_final, &P_final, &q_final);

  const Eigen::VectorXd t_upper =
      (iiwa_diagram.plant().GetPositionUpperLimits() / 2)
          .array()
          .tan()
          .matrix();
  const Eigen::VectorXd t_lower =
      (iiwa_diagram.plant().GetPositionLowerLimits() / 2)
          .array()
          .tan()
          .matrix();
  WriteCspacePolytopeToFile(C_final, d_final, t_lower, t_upper, write_file, 10);
  // Now partition the certified region C_final * t <= d_final, t_lower <= t <=
  // t_upper into boxes.
  Eigen::MatrixXd C_bar;
  Eigen::VectorXd d_bar;
  GetCspacePolytope(C_final, d_final, t_lower, t_upper, &C_bar, &d_bar);
  const int num_boxes = 10;
  geometry::optimization::FindInscribedBox find_box(C_bar, d_bar, {},
                                                    std::nullopt);
  find_box.MaximizeBoxVolume();
  std::vector<geometry::optimization::AxisAlignedBox> boxes;
  solvers::GurobiSolver gurobi_solver;
  for (int i = 0; i < num_boxes; ++i) {
    const auto result_box =
        gurobi_solver.Solve(find_box.prog(), std::nullopt, solver_options);
    geometry::optimization::AxisAlignedBox box(
        result_box.GetSolution(find_box.box_lo()),
        result_box.GetSolution(find_box.box_up()));
    drake::log()->info(fmt::format("Box volume {}", box.volume()));
    boxes.push_back(box);
    const auto obstacle = box.Scale(0.9);
    find_box.AddObstacle(obstacle);
  }
}

void VisualizePostures(const std::string& cspace_polytope_file,
                       int num_postures) {
  IiwaDiagram iiwa_diagram{};
  auto diagram_context = iiwa_diagram.diagram().CreateDefaultContext();
  auto& plant_context =
      iiwa_diagram.plant().GetMyMutableContextFromRoot(diagram_context.get());
  Eigen::MatrixXd C;
  Eigen::VectorXd d, t_lower, t_upper;
  ReadCspacePolytopeFromFile(cspace_polytope_file, &C, &d, &t_lower, &t_upper);
  const int nt = t_lower.rows();

  Eigen::MatrixXd t_samples = Eigen::MatrixXd::Random(nt, num_postures);
  // Project t_samples.col(i) to the polytope.
  solvers::MathematicalProgram prog;
  auto t_project = prog.NewContinuousVariables(nt);
  prog.AddBoundingBoxConstraint(t_lower, t_upper, t_project);
  prog.AddLinearConstraint(C, Eigen::VectorXd::Constant(d.rows(), -kInf), d,
                           t_project);
  auto cost = prog.AddQuadraticErrorCost(Eigen::MatrixXd::Identity(nt, nt),
                                         Eigen::VectorXd::Zero(nt), t_project);
  for (int i = 0; i < num_postures; ++i) {
    t_samples.col(i) =
        (t_samples.col(i).array() * (t_upper - t_lower).array() / 2).matrix() +
        (t_upper + t_lower) / 2;
    cost.evaluator()->UpdateCoefficients(Eigen::MatrixXd::Identity(nt, nt),
                                         -t_samples.col(i), 0.);
    const auto result = solvers::Solve(prog);
    DRAKE_DEMAND(result.is_success());
    const auto t_val = result.GetSolution(t_project);
    const Eigen::VectorXd q_val = (t_val.array().atan() * 2).matrix();
    iiwa_diagram.plant().SetPositions(&plant_context, q_val);
    iiwa_diagram.diagram().Publish(*diagram_context);
    std::cin.get();
  }
}

int DoMain() {
  const std::string cspace_polytope_file = "iiwa_shelf_cspace_polytope.txt";
  SearchCspacePolytope(cspace_polytope_file);
  VisualizePostures(cspace_polytope_file, 10);
  return 0;
}
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
