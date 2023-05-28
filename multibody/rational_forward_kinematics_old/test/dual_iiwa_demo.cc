#include <iostream>
#include <limits>

#include "drake/common/find_resource.h"
#include "drake/geometry/collision_filter_declaration.h"
#include "drake/geometry/geometry_roles.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/polytope_cover.h"
#include "drake/geometry/scene_graph_inspector.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/rational_forward_kinematics_old/cspace_free_region.h"
#include "drake/multibody/tree/multibody_tree_indexes.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
namespace rational_old {
const double kInf = std::numeric_limits<double>::infinity();

std::vector<geometry::GeometryId> GetCollisionGeometries(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraphInspector<double>& inspector,
    ModelInstanceIndex model_instance) {
  std::vector<geometry::GeometryId> ret;
  for (const auto& body_index : plant.GetBodyIndices(model_instance)) {
    const std::vector<geometry::GeometryId> body_geometries =
        inspector.GetGeometries(plant.GetBodyFrameIdOrThrow(body_index));
    ret.insert(ret.end(), body_geometries.begin(), body_geometries.end());
  }
  return ret;
}

void SetDiffuse(const MultibodyPlant<double>& plant,
                geometry::SceneGraph<double>* scene_graph,
                const BodyIndex body_index,
                const std::optional<std::string>& geometry_name,
                std::optional<double> rgba_r, std::optional<double> rgba_g,
                std::optional<double> rgba_b, std::optional<double> rgba_a) {
  const auto& inspector = scene_graph->model_inspector();
  const std::optional<geometry::FrameId> frame_id =
      plant.GetBodyFrameIdIfExists(body_index);
  if (frame_id.has_value()) {
    for (const auto& geometry_id : inspector.GetGeometries(
             frame_id.value(), geometry::Role::kIllustration)) {
      if (geometry_name.has_value()) {
        if (inspector.GetName(geometry_id) != *geometry_name) {
          continue;
        }
      }
      const geometry::GeometryProperties* props =
          inspector.GetProperties(geometry_id, geometry::Role::kIllustration);
      if (props == nullptr || !props->HasProperty("phong", "diffuse")) {
        continue;
      }
      const auto old_rgba =
          props->GetProperty<geometry::Rgba>("phong", "diffuse");
      double rgba_r_val = rgba_r.has_value() ? rgba_r.value() : old_rgba.r();
      double rgba_g_val = rgba_g.has_value() ? rgba_g.value() : old_rgba.g();
      double rgba_b_val = rgba_b.has_value() ? rgba_b.value() : old_rgba.b();
      double rgba_a_val = rgba_a.has_value() ? rgba_a.value() : old_rgba.a();
      if (geometry_name.has_value()) {
        std::cout << *geometry_name << " ";
      }
      std::cout << rgba_r_val << " " << rgba_g_val << " " << rgba_b_val << " "
                << rgba_a_val << "\n";
      geometry::IllustrationProperties new_props =
          geometry::MakePhongIllustrationProperties(
              Eigen::Vector4d(rgba_r_val, rgba_g_val, rgba_b_val, rgba_a_val));
      scene_graph->AssignRole(*plant.get_source_id(), geometry_id, new_props,
                              geometry::RoleAssign::kReplace);
    }
  }
}

double CalcMinDistance(
    const MultibodyPlant<double>& plant,
    const geometry::SceneGraphInspector<double>& inspector,
    const systems::Context<double>& context,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  const auto& query_port = plant.get_geometry_query_input_port();
  const auto& query_object =
      query_port.Eval<geometry::QueryObject<double>>(context);
  double min_distance = kInf;
  for (const auto& geometry_pair : inspector.GetCollisionCandidates()) {
    if (!IsGeometryPairCollisionIgnored(geometry_pair.first,
                                        geometry_pair.second,
                                        filtered_collision_pairs)) {
      const geometry::SignedDistancePair<double> signed_distance_pair =
          query_object.ComputeSignedDistancePairClosestPoints(
              geometry_pair.first, geometry_pair.second);
      min_distance = std::min(min_distance, signed_distance_pair.distance);
    }
  }
  return min_distance;
}

class DualIiwaDiagram {
 public:
  // @param end_effector_only If set to true, then we only consider the self
  // collision between the end-effector link on each IIWA.
  // @param num_dual_iiwa Set this to != 1 only if you visualize multiple
  // postures. Set this to 1 one you search for Cspace polytopes.
  explicit DualIiwaDiagram(bool end_effector_only, int num_dual_iiwa = 1)
      : meshcat_{std::make_shared<geometry::Meshcat>()} {
    systems::DiagramBuilder<double> builder;
    auto [plant, sg] = AddMultibodyPlantSceneGraph(&builder, 0.);
    plant_ = &plant;
    scene_graph_ = &sg;

    multibody::Parser parser(plant_);
    const std::string iiwa_file_path = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/sdf/"
        "iiwa14_coarse_collision_visual_weld_wrist.sdf");
    for (int dual_iiwa_count = 0; dual_iiwa_count < num_dual_iiwa;
         ++dual_iiwa_count) {
      const auto left_iiwa = parser.AddModelFromFile(
          iiwa_file_path, fmt::format("left_iiwa{}", dual_iiwa_count));
      left_iiwas_.push_back(left_iiwa);

      const math::RigidTransformd X_WL(Eigen::Vector3d(0, -0.3, 0));
      plant_->WeldFrames(plant_->world_frame(),
                         plant_->GetFrameByName("iiwa_link_0", left_iiwa),
                         X_WL);
      const auto right_iiwa = parser.AddModelFromFile(
          iiwa_file_path, fmt::format("right_iiwa{}", dual_iiwa_count));
      right_iiwas_.push_back(right_iiwa);
      const math::RigidTransformd X_WR(Eigen::Vector3d(0, 0.3, 0));
      plant_->WeldFrames(plant_->world_frame(),
                         plant_->GetFrameByName("iiwa_link_0", right_iiwa),
                         X_WR);

      const auto& inspector = scene_graph_->model_inspector();
      const auto left_iiwa_geometries =
          GetCollisionGeometries(*plant_, inspector, left_iiwa);
      const auto right_iiwa_geometries =
          GetCollisionGeometries(*plant_, inspector, right_iiwa);
      // SceneGraph should ignore the self collisions on each individual iiwa.
      scene_graph_->collision_filter_manager().Apply(
          geometry::CollisionFilterDeclaration().ExcludeWithin(
              geometry::GeometrySet(left_iiwa_geometries)));
      scene_graph_->collision_filter_manager().Apply(
          geometry::CollisionFilterDeclaration().ExcludeWithin(
              geometry::GeometrySet(right_iiwa_geometries)));
      if (end_effector_only) {
        auto left_iiwa_lower_geometries = left_iiwa_geometries;
        for (auto it = left_iiwa_lower_geometries.begin();
             it != left_iiwa_lower_geometries.end();) {
          if (inspector.GetFrameId(*it) ==
              plant_->GetBodyFrameIdIfExists(
                  plant_->GetBodyByName("iiwa_link_6", left_iiwa).index())) {
            it = left_iiwa_lower_geometries.erase(it);
          } else {
            ++it;
          }
        }
        auto right_iiwa_lower_geometries = right_iiwa_geometries;
        for (auto it = right_iiwa_lower_geometries.begin();
             it != right_iiwa_lower_geometries.end();) {
          if (inspector.GetFrameId(*it) ==
              plant_->GetBodyFrameIdIfExists(
                  plant_->GetBodyByName("iiwa_link_6", right_iiwa).index())) {
            it = right_iiwa_lower_geometries.erase(it);
          } else {
            ++it;
          }
        }
        scene_graph_->collision_filter_manager().Apply(
            geometry::CollisionFilterDeclaration().ExcludeBetween(
                geometry::GeometrySet(left_iiwa_geometries),
                geometry::GeometrySet(right_iiwa_lower_geometries)));
        scene_graph_->collision_filter_manager().Apply(
            geometry::CollisionFilterDeclaration().ExcludeBetween(
                geometry::GeometrySet(right_iiwa_geometries),
                geometry::GeometrySet(left_iiwa_lower_geometries)));
      }
    }

    plant_->Finalize();
    // Set the intensity of the iiwa arm visualization.
    for (int dual_iiwa_count = 1; dual_iiwa_count < num_dual_iiwa;
         ++dual_iiwa_count) {
      Eigen::Vector4d diffuse(0.4, 0.4, 0.4, 1.);
      if (dual_iiwa_count == 1) {
        diffuse << 1., 0.4, 0.4, 0.6;
      } else if (dual_iiwa_count == 2) {
        diffuse << 0.4, 1., 0.4, 0.6;
      } else if (dual_iiwa_count == 3) {
        diffuse << 0.4, 0.4, 1., 0.6;
      }
      for (const auto& body_index :
           plant_->GetBodyIndices(left_iiwas_[dual_iiwa_count])) {
        SetDiffuse(*plant_, scene_graph_, body_index, std::nullopt, diffuse(0),
                   diffuse(1), diffuse(2), diffuse(3));
      }
      for (const auto& body_index :
           plant_->GetBodyIndices(right_iiwas_[dual_iiwa_count])) {
        SetDiffuse(*plant_, scene_graph_, body_index, std::nullopt, diffuse(0),
                   diffuse(1), diffuse(2), diffuse(3));
      }
    }
    ColorIiwa(left_iiwas_[0], "left_iiwa0::iiwa_link_6_coarse");
    ColorIiwa(right_iiwas_[0], "right_iiwa0::iiwa_link_5_coarse1");

    geometry::MeshcatVisualizerParams meshcat_params{};
    meshcat_params.role = geometry::Role::kIllustration;
    visualizer_ = &geometry::MeshcatVisualizer<double>::AddToBuilder(
        &builder, *scene_graph_, meshcat_, meshcat_params);
    diagram_ = builder.Build();
  }

  void ColorIiwa(const ModelInstanceIndex model_instance,
                 const std::string& highlight_geo_name) {
    for (const auto& body_index : plant_->GetBodyIndices(model_instance)) {
      const auto frame_id = plant_->GetBodyFrameIdOrThrow(body_index);
      for (const auto& geo_id : scene_graph_->model_inspector().GetGeometries(
               frame_id, geometry::Role::kIllustration)) {
        const auto geo_name = scene_graph_->model_inspector().GetName(geo_id);
        std::srand(geo_id.get_value());
        double rgb_r = std::rand() / static_cast<float>(RAND_MAX) * 0.2;
        double rgb_g = std::rand() / static_cast<float>(RAND_MAX) * 0.4 + 0.1;
        double rgb_b = std::rand() / static_cast<float>(RAND_MAX) * 0.4 + 0.5;
        double rgb_a = 0.9;
        if (geo_name.find("visual") <= geo_name.size()) {
          rgb_a = 0;
        }
        if (geo_name == highlight_geo_name) {
          rgb_r = 1;
          rgb_g = 0;
          rgb_b = 0;
          rgb_a = 1;
        }
        SetDiffuse(*plant_, scene_graph_, body_index, geo_name, rgb_r, rgb_g,
                   rgb_b, rgb_a);
      }
    }
  }

  const systems::Diagram<double>& diagram() const { return *diagram_; }

  const multibody::MultibodyPlant<double>& plant() const { return *plant_; }

  const geometry::SceneGraph<double>& scene_graph() const {
    return *scene_graph_;
  }

  geometry::Meshcat* meshcat() { return meshcat_.get(); }

  const std::vector<ModelInstanceIndex>& left_iiwas() const {
    return left_iiwas_;
  }

  const std::vector<ModelInstanceIndex>& right_iiwas() const {
    return right_iiwas_;
  }

 private:
  std::unique_ptr<systems::Diagram<double>> diagram_;
  multibody::MultibodyPlant<double>* plant_;
  geometry::SceneGraph<double>* scene_graph_;
  std::shared_ptr<geometry::Meshcat> meshcat_;
  geometry::MeshcatVisualizer<double>* visualizer_;
  std::vector<multibody::ModelInstanceIndex> left_iiwas_;
  std::vector<multibody::ModelInstanceIndex> right_iiwas_;
};

Eigen::VectorXd FindInitialPosture(const MultibodyPlant<double>& plant,
                                   ModelInstanceIndex left_iiwa,
                                   ModelInstanceIndex right_iiwa,
                                   systems::Context<double>* plant_context) {
  InverseKinematics ik(plant, plant_context);
  const auto& left_link7 = plant.GetFrameByName("iiwa_link_7", left_iiwa);
  const auto& right_link7 = plant.GetFrameByName("iiwa_link_7", right_iiwa);
  ik.AddPositionConstraint(left_link7, Eigen::Vector3d::Zero(),
                           plant.world_frame(), Eigen::Vector3d(0, 0, -kInf),
                           Eigen::Vector3d::Constant(kInf));
  ik.AddPositionConstraint(
      right_link7, Eigen::Vector3d::Zero(), plant.world_frame(),
      Eigen::Vector3d(0, -kInf, -kInf), Eigen::Vector3d(kInf, 0, kInf));
  ik.AddMinimumDistanceConstraint(0.02);
  constexpr int nq = 12;

  Eigen::Matrix<double, nq, 1> q_init;
  q_init << 0.1, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2, -0.1, 0.3, 0.2, 0.5, 0.7;
  ik.get_mutable_prog()->SetInitialGuess(ik.q(), q_init);
  const auto result = solvers::Solve(ik.prog());
  if (!result.is_success()) {
    drake::log()->warn("Cannot find the posture\n");
  }
  return result.GetSolution(ik.q());
}

void BuildCandidateCspacePolytope(const Eigen::VectorXd& q_free,
                                  Eigen::MatrixXd* C, Eigen::VectorXd* d) {
  const int C_rows = 30;
  const int nq = q_free.rows();
  C->resize(C_rows, nq);
  *C = Eigen::MatrixXd::Random(C_rows, nq) * 0.1;
  for (int i = 0; i < nq; ++i) {
    (*C)(2 * i, i) = 5;
    (*C)(2 * i + 1, i) = -5;
  }
  for (int i = 0; i < C_rows; ++i) {
    C->row(i).normalize();
  }
  *d = (*C) * (q_free / 2).array().tan().matrix() +
       0.0001 * Eigen::VectorXd::Ones(C_rows);
  if (!geometry::optimization::HPolyhedron(*C, *d).IsBounded()) {
    throw std::runtime_error("C*t <= d is not bounded");
  }
}

void SearchCspacePolytope(
    const std::string& write_file,
    const std::optional<std::string>& read_file = std::nullopt,
    bool do_bisection = true) {
  // Ensure that we have the MOSEK license for the entire duration of this test,
  // so that we do not have to release and re-acquire the license for every
  // test.
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  const DualIiwaDiagram dual_iiwa_diagram{true};
  auto diagram_context = dual_iiwa_diagram.diagram().CreateDefaultContext();
  auto& plant_context = dual_iiwa_diagram.plant().GetMyMutableContextFromRoot(
      diagram_context.get());
  const auto q0 = FindInitialPosture(
      dual_iiwa_diagram.plant(), dual_iiwa_diagram.left_iiwas()[0],
      dual_iiwa_diagram.right_iiwas()[0], &plant_context);
  dual_iiwa_diagram.plant().SetPositions(&plant_context, q0);
  dual_iiwa_diagram.diagram().Publish(*diagram_context);
  std::cout << "type to continue\n";
  std::cin.get();

  Eigen::MatrixXd C_init;
  Eigen::VectorXd d_init;
  if (read_file.has_value()) {
    std::unordered_map<SortedPair<geometry::GeometryId>,
                       std::pair<BodyIndex, Eigen::VectorXd>>
        separating_planes_read;
    ReadCspacePolytopeFromFile(
        read_file.value(), dual_iiwa_diagram.plant(),
        dual_iiwa_diagram.scene_graph().model_inspector(), &C_init, &d_init,
        &separating_planes_read);
  } else {
    BuildCandidateCspacePolytope(q0, &C_init, &d_init);
  }

  const CspaceFreeRegion dut(
      dual_iiwa_diagram.diagram(), &(dual_iiwa_diagram.plant()),
      &(dual_iiwa_diagram.scene_graph()), SeparatingPlaneOrder::kAffine,
      CspaceRegionType::kGenericPolytope);

  CspaceFreeRegion::FilteredCollisionPairs filtered_collision_pairs{};

  CspaceFreeRegion::BinarySearchOption binary_search_option{
      .epsilon_max = 0.055,
      .epsilon_min = 0.,
      .max_iters = 3,
      .search_d = false,
      .compute_polytope_volume = false,
      .num_threads = 10,
      .check_epsilon_min = false};

  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, false);
  CspaceFreeRegionSolution cspace_free_region_solution;
  const int nq = dual_iiwa_diagram.plant().num_positions();
  Eigen::VectorXd q_star = Eigen::VectorXd::Zero(nq);

  Eigen::MatrixXd C_bilinear_start = C_init;
  Eigen::VectorXd d_bilinear_start =
      d_init +
      binary_search_option.epsilon_max * Eigen::VectorXd::Ones(d_init.rows());
  const Eigen::VectorXd t_lower =
      (dual_iiwa_diagram.plant().GetPositionLowerLimits() / 2).array().tan();
  const Eigen::VectorXd t_upper =
      (dual_iiwa_diagram.plant().GetPositionUpperLimits() / 2).array().tan();
  if (do_bisection) {
    dut.CspacePolytopeBinarySearch(
        q_star, filtered_collision_pairs, C_init, d_init, binary_search_option,
        solver_options, q0, std::nullopt, &cspace_free_region_solution);
    WriteCspacePolytopeToFile(
        cspace_free_region_solution, dual_iiwa_diagram.plant(),
        dual_iiwa_diagram.scene_graph().model_inspector(), write_file, 10);
    C_bilinear_start = cspace_free_region_solution.C;
    d_bilinear_start = cspace_free_region_solution.d;
  }
  // Bilinear alternation
  CspaceFreeRegion::BilinearAlternationOption bilinear_alternation_option{
      .max_iters = 3,
      .convergence_tol = 0.,
      .lagrangian_backoff_scale = 0.02,
      .redundant_tighten = 0.5,
      .compute_polytope_volume = false,
      .num_threads = 10};

  std::vector<double> polytope_volumes, ellipsoid_determinants;
  dut.CspacePolytopeBilinearAlternation(
      q_star, filtered_collision_pairs, C_bilinear_start, d_bilinear_start,
      bilinear_alternation_option, solver_options, q0, std::nullopt,
      &cspace_free_region_solution, &polytope_volumes, &ellipsoid_determinants);
  WriteCspacePolytopeToFile(
      cspace_free_region_solution, dual_iiwa_diagram.plant(),
      dual_iiwa_diagram.scene_graph().model_inspector(), write_file, 10);
}

void VisualizePostures(const std::string& read_file) {
  const int num_postures = 3;
  DualIiwaDiagram iiwa_diagram(false, num_postures);
  Eigen::MatrixXd C;
  Eigen::VectorXd d;
  std::unordered_map<SortedPair<geometry::GeometryId>,
                     std::pair<BodyIndex, Eigen::VectorXd>>
      separating_planes;
  ReadCspacePolytopeFromFile(read_file, iiwa_diagram.plant(),
                             iiwa_diagram.scene_graph().model_inspector(), &C,
                             &d, &separating_planes);
  const Eigen::VectorXd t_lower =
      (iiwa_diagram.plant().GetPositionLowerLimits().array() / 2)
          .tan()
          .matrix();
  const Eigen::VectorXd t_upper =
      (iiwa_diagram.plant().GetPositionUpperLimits().array() / 2)
          .tan()
          .matrix();
  // Solve a program such that the sum of inter-posture distance is maximized.
  solvers::MathematicalProgram prog;
  const int nt = C.cols();
  auto t = prog.NewContinuousVariables(nt, 2);
  for (int i = 0; i < 2; ++i) {
    prog.AddLinearConstraint(C, Eigen::VectorXd::Constant(d.rows(), -kInf), d,
                             t.col(i));
    prog.AddBoundingBoxConstraint(t_lower, t_upper, t.col(i));
  }
  // Now add the cost max (t.col(i) - t.col(j))^2
  for (int i = 0; i < 2; ++i) {
    for (int j = i + 1; j < 2; ++j) {
      // A * [t.col(i); t.col(j)] = t.col(i) - t.col(j)
      Eigen::MatrixXd A(nt, 2 * nt);
      A << Eigen::MatrixXd::Identity(nt, nt),
          -Eigen::MatrixXd::Identity(nt, nt);
      prog.AddQuadraticCost(-A.transpose() * A, Eigen::VectorXd::Zero(2 * nt),
                            {t.col(i), t.col(j)}, false /* is_convex=false */);
    }
    prog.SetInitialGuess(t.col(i), Eigen::VectorXd::Random(nt));
  }
  const auto result = solvers::Solve(prog);
  std::cout << result.get_solution_result() << "\n";
  DRAKE_DEMAND(result.is_success());

  Eigen::MatrixXd t_sol(nt, 2);
  for (int i = 0; i < 2; ++i) {
    t_sol.col(i) = result.GetSolution(t.col(i));
  }
  Eigen::MatrixXd q_sol = 2 * (t_sol.array().atan().matrix());
  std::cout << "postures\n" << q_sol.transpose() << "\n";

  auto diagram_context = iiwa_diagram.diagram().CreateDefaultContext();
  auto& plant_context =
      iiwa_diagram.plant().GetMyMutableContextFromRoot(diagram_context.get());
  Eigen::MatrixXd t_interpolate(nt, num_postures);
  t_interpolate.col(0) = t_sol.col(0);
  t_interpolate.rightCols<1>() = t_sol.col(1);
  for (int i = 1; i < num_postures - 1; ++i) {
    t_interpolate.col(i) =
        t_interpolate.col(0) +
        (t_sol.col(1) - t_sol.col(0)) / (num_postures - 1) * i;
  }
  Eigen::MatrixXd q_interpolate = 2 * (t_interpolate.array().atan().matrix());
  for (int i = 0; i < num_postures; ++i) {
    iiwa_diagram.plant().SetPositions(&plant_context,
                                      iiwa_diagram.left_iiwas()[i],
                                      q_interpolate.col(i).head(nt / 2));
    iiwa_diagram.plant().SetPositions(&plant_context,
                                      iiwa_diagram.right_iiwas()[i],
                                      q_interpolate.col(i).tail(nt / 2));
  }
  iiwa_diagram.diagram().Publish(*diagram_context);
  std::cout << "Type to continue\n";
  std::cin.get();
}

// Generate a square on the xy plane as a mesh, with N points on each edge.
void GeneratePlaneMesh(double width, int N, Eigen::Matrix3Xd* vertices,
                       Eigen::Matrix3Xi* faces) {
  vertices->resize(3, N * N);
  Eigen::RowVectorXd coord =
      Eigen::RowVectorXd::LinSpaced(N, -width / 2, width / 2);
  for (int i = 0; i < N; ++i) {
    vertices->block(0, i * N, 1, N) = coord;
    vertices->block(1, i * N, 1, N) = coord(i) * Eigen::RowVectorXd::Ones(N);
    vertices->block(2, i * N, 1, N) = Eigen::RowVectorXd::Zero(N);
  }
  faces->resize(3, (N - 1) * (N - 1) * 2);
  for (int i = 0; i < N - 1; ++i) {
    for (int j = 0; j < N - 1; ++j) {
      faces->col(i * (N - 1) * 2 + 2 * j) << i * N + j, i * N + j + 1,
          (i + 1) * N + j;
      faces->col(i * (N - 1) * 2 + 2 * j + 1) << i * N + j + 1, (i + 1) * N + j,
          (i + 1) * N + j + 1;
    }
  }
}

void VisualizeOnePosture(const std::string& read_file) {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  DualIiwaDiagram iiwa_diagram(false);
  Eigen::MatrixXd C;
  Eigen::VectorXd d;
  std::unordered_map<SortedPair<geometry::GeometryId>,
                     std::pair<BodyIndex, Eigen::VectorXd>>
      separating_planes;
  ReadCspacePolytopeFromFile(read_file, iiwa_diagram.plant(),
                             iiwa_diagram.scene_graph().model_inspector(), &C,
                             &d, &separating_planes);
  const Eigen::VectorXd t_lower =
      (iiwa_diagram.plant().GetPositionLowerLimits().array() / 2)
          .tan()
          .matrix();
  const Eigen::VectorXd t_upper =
      (iiwa_diagram.plant().GetPositionUpperLimits().array() / 2)
          .tan()
          .matrix();
  // Find a posture
  const int nq = iiwa_diagram.plant().num_positions();
  auto diagram_context = iiwa_diagram.diagram().CreateDefaultContext();
  const auto& plant = iiwa_diagram.plant();
  auto& plant_context =
      plant.GetMyMutableContextFromRoot(diagram_context.get());

  // Sample many postures, find the one that has the closest collision distance.
  const int num_samples = 10000;
  Eigen::MatrixXd t_samples = Eigen::MatrixXd::Random(nq, num_samples);
  Eigen::VectorXd t_sol(nq);
  Eigen::VectorXd q_sol(nq);
  double min_distance = kInf;
  bool do_sample = false;
  if (do_sample) {
    for (int sample_count = 0; sample_count < num_samples; sample_count++) {
      solvers::MathematicalProgram prog;
      Eigen::VectorXd t_sample = t_samples.col(sample_count);
      auto t = prog.NewContinuousVariables(nq);
      prog.AddLinearConstraint(C, Eigen::VectorXd::Constant(d.rows(), -kInf), d,
                               t);
      prog.AddBoundingBoxConstraint(t_lower, t_upper, t);

      prog.AddQuadraticErrorCost(Eigen::MatrixXd::Identity(nq, nq), t_sample,
                                 t);
      const auto result = solvers::Solve(prog);
      DRAKE_DEMAND(result.is_success());
      Eigen::VectorXd t_proj = result.GetSolution(t);
      Eigen::VectorXd q_proj = 2 * (t_proj.array().atan().matrix());
      plant.SetPositions(&plant_context, q_proj);
      double distance =
          CalcMinDistance(plant, iiwa_diagram.scene_graph().model_inspector(),
                          plant_context, {});
      if (distance < min_distance) {
        q_sol = q_proj;
        t_sol = t_proj;
        std::cout << "sample_count: " << sample_count
                  << " distance: " << distance << "\n";
        min_distance = distance;
      }
    }
  } else {
    // This posture is found from previous random samples.
    q_sol << 0.664318, 0.582318, -0.517396, 0.295508, 0.366561, 0.345124,
        0.957058, 0.0691687, 0.772129, 0.911172, 0.194716, 0.339097;
    t_sol << 0.344939, 0.299676, -0.264628, 0.148839, 0.185361, 0.174295,
        0.518743, 0.0345981, 0.406462, 0.489963, 0.0976667, 0.171192;
  }
  plant.SetPositions(&plant_context, q_sol);
  iiwa_diagram.diagram().Publish(*diagram_context);
  std::cout << "minimal distance: "
            << CalcMinDistance(plant,
                               iiwa_diagram.scene_graph().model_inspector(),
                               plant_context, {})
            << "\n";
  std::cout << "q_sol: " << q_sol.transpose() << "\n";
  std::cout << "t_sol: " << t_sol.transpose() << "\n";

  // Compute the plane.
  const BodyIndex left_link =
      plant.GetBodyByName("iiwa_link_6", iiwa_diagram.left_iiwas()[0]).index();
  const geometry::FrameId left_link_frame =
      plant.GetBodyFrameIdOrThrow(left_link);
  const geometry::FrameId right_link = plant.GetBodyFrameIdOrThrow(
      plant.GetBodyByName("iiwa_link_5", iiwa_diagram.right_iiwas()[0])
          .index());
  const geometry::GeometryId left_geo_id =
      iiwa_diagram.scene_graph().model_inspector().GetGeometryIdByName(
          left_link_frame, geometry::Role::kProximity,
          "left_iiwa0::iiwa_link_6_collision");
  const geometry::GeometryId right_geo_id =
      iiwa_diagram.scene_graph().model_inspector().GetGeometryIdByName(
          right_link, geometry::Role::kProximity,
          "right_iiwa0::iiwa_link_5_collision1");
  auto it = separating_planes.find(
      SortedPair<geometry::GeometryId>(left_geo_id, right_geo_id));
  DRAKE_DEMAND(it != separating_planes.end());
  const BodyIndex expressed_body = it->second.first;
  const Eigen::VectorXd separating_plane_vars = it->second.second;
  Eigen::Vector3d a_E;
  double b_E;
  CalcPlane(separating_plane_vars, t_sol, SeparatingPlaneOrder::kAffine, &a_E,
            &b_E);
  // In the expressed frame, the plane is
  // a_E.dot(p_EV) + b_E = 0
  // In the world frame, the plane is a_E.T * R_EW * p_WV + a_E.dot(p_EW) + b_E
  // = 0.
  const auto X_EW = plant.CalcRelativeTransform(
      plant_context, plant.get_body(expressed_body).body_frame(),
      plant.world_frame());
  Eigen::Vector3d a_W = X_EW.rotation().inverse() * a_E;
  double b_W = a_E.dot(X_EW.translation()) + b_E;
  // Now find the projection of the left link origin to the plane.
  Eigen::Vector3d p_WL;
  plant.CalcPointsPositions(
      plant_context, plant.get_body(left_link).body_frame(),
      Eigen::Vector3d::Zero(), plant.world_frame(), &p_WL);
  const Eigen::Vector3d p_WL_project =
      p_WL + (-b_W - a_W.dot(p_WL)) / (a_W.dot(a_W)) * a_W;
  const Eigen::Vector3d a_W_normalized = a_W / a_W.norm();
  Eigen::Matrix3Xd plane_mesh_vertices;
  Eigen::Matrix3Xi plane_mesh_faces;
  GeneratePlaneMesh(0.8, 25, &plane_mesh_vertices, &plane_mesh_faces);
  plane_mesh_vertices =
      (math::RotationMatrix<double>::MakeFromOneUnitVector(a_W_normalized, 2) *
       plane_mesh_vertices)
          .colwise() +
      p_WL_project;
  iiwa_diagram.meshcat()->SetTriangleMesh(
      "separating_plane", plane_mesh_vertices, plane_mesh_faces,
      geometry::Rgba(0.1, 0.9, 0.1, 1.0), true, 1.0);
  std::cout << "Type to continue";
  std::cin.get();
}

int DoMain() {
  const std::string write_file = "dual_iiwa_cspace_polytope3.txt";
  const std::string read_file = "dual_iiwa_cspace_polytope.txt";
  // SearchCspacePolytope(write_file, std::nullopt, false);
  // VisualizePostures(write_file);
  VisualizeOnePosture(write_file);
  return 0;
}
}  // namespace rational_old
}  // namespace multibody
}  // namespace drake

int main() { drake::multibody::DoMain(); }
