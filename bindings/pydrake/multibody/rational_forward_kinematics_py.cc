#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/value_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/multibody/rational_forward_kinematics_old/collision_geometry.h"
#include "drake/multibody/rational_forward_kinematics_old/cspace_free_line.h"
#include "drake/multibody/rational_forward_kinematics_old/cspace_free_region.h"
#include "drake/multibody/rational_forward_kinematics_old/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics_old/rational_forward_kinematics.h"
#include "drake/multibody/rational_forward_kinematics_old/rational_forward_kinematics_internal.h"

namespace drake {
namespace pydrake {

template <typename T>

void DoPoseDeclaration(py::module m, T) {
  py::tuple param = GetPyParam<T>();
  using Class = multibody::rational_old::RationalForwardKinematicsOld::Pose<T>;
  auto cls = DefineTemplateClassWithDefault<Class>(
      m, "RationalForwardKinematicsPoseOld", param);
  cls.def("translation", [](const Class& self) { return self.p_AB; })
      .def("rotation", [](const Class& self) { return self.R_AB; })
      .def_readwrite("frame_A_index", &Class::frame_A_index)
      .def("asRigidTransformExpr",
          [](const Class& self) { return self.asRigidTransformExpression(); });

  DefCopyAndDeepCopy(&cls);
}

// SeparatingPlane
template <typename T>
void DoScalarDependentDefinitions(py::module m, T) {
  py::tuple param = GetPyParam<T>();
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  {
    using Class = multibody::rational_old::SeparatingPlane<T>;
    auto cls =
        DefineTemplateClassWithDefault<Class>(m, "SeparatingPlaneOld", param)
            .def_readonly("a", &Class::a, py_rvp::copy)
            .def_readonly("b", &Class::b)
            .def_readonly(
                "positive_side_geometry", &Class::positive_side_geometry)
            .def_readonly(
                "negative_side_geometry", &Class::negative_side_geometry)
            .def_readonly("expressed_link", &Class::expressed_link)
            .def_readonly("order", &Class::order)
            .def_readonly(
                "decision_variables", &Class::decision_variables, py_rvp::copy);
    DefCopyAndDeepCopy(&cls);
    AddValueInstantiation<Class>(m);
  }
}

PYBIND11_MODULE(rational_forward_kinematics, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::multibody::rational_old;
  //  constexpr auto& doc = pydrake_doc.drake.multibody;
  m.doc() = "RationalForwardKinematicsOld module";

  //  py::module::import("pydrake.math");
  //  py::module::import("pydrake.multibody.plant");
  //  py::module::import("pydrake.solvers.mathematicalprogram");
  {
    py::class_<CspaceLineTuple>(m, "CspaceLineTuple")
        .def(py::init<const symbolic::Variable&,
                 const drake::VectorX<symbolic::Variable>&,
                 const drake::VectorX<symbolic::Variable>&,
                 const symbolic::Polynomial&, const VerificationOption&>(),
            py::arg("mu"), py::arg("s0"), py::arg("s0"), py::arg("s1"),
            py::arg("m_rational_numerator"))
        //  TODO(Alexandre.Amice) bind AddPositivityConstraintToProgram.
        //        .def("AddPositivityConstraintToProgram",
        //            &Class::AddPositivityConstraintToProgram, py::arg("env"),
        //            py::arg("prog"), cls_doc.AddPositivityConstraintToProgram)
        .def("get_p", &CspaceLineTuple::get_p)
        .def("get_psatz_variables_and_psd_constraints",
            &CspaceLineTuple::get_psatz_variables_and_psd_constraints);
  }
  // RationalForwardKinematics Class
  {
    using Class = RationalForwardKinematicsOld;
    // no class docs built
    //    constexpr auto& cls_doc = doc.RationalForwardKinematicsOld;
    py::class_<Class>(m, "RationalForwardKinematicsOld")
        .def(py::init<const multibody::MultibodyPlant<double>&>(),
            py::arg("plant"),
            //              Keep alive, reference: `self` keeps `plant` alive.
            py::keep_alive<1, 2>()  // BR
            )
        .def("CalcLinkPoses", &Class::CalcLinkPoses, py::arg("q_star"),
            py::arg("expressed_body_index")
            //             cls_doc.CalcLinkPoses
            )
        .def("CalcLinkPosesAsMultilinearPolynomials",
            &Class::CalcLinkPosesAsMultilinearPolynomials, py::arg("q_star"),
            py::arg("expressed_body_index")
            //             cls_doc.CalcLinkPoses
            )
        .def("ConvertMultilinearPolynomialToRationalFunction",
            &Class::ConvertMultilinearPolynomialToRationalFunction, py::arg("e")
            //             cls_doc.CalcLinkPoses
            )
        .def("plant", &Class::plant
            //             cls_doc.CalcLinkPoses
            )
        .def("t", &Class::t
            //             cls_doc.CalcLinkPoses
            )
        .def("ComputeTValue",
            overload_cast_explicit<Eigen::VectorXd,
                const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>&, bool>(
                &Class::ComputeTValue),
            py::arg("q_val"), py::arg("q_star_val"),
            py::arg("clamp_angle") = false
            //             cls_doc.CalcLinkPoses
            )

        .def("ComputeQValue",
            overload_cast_explicit<Eigen::VectorXd,
                const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>&>(
                &Class::ComputeQValue),
            py::arg("t_val"), py::arg("q_star_val")
            //             cls_doc.CalcLinkPoses
            )
        .def(
            "FindTOnPath", &Class::FindTOnPath, py::arg("start"), py::arg("end")
            //             cls_doc.CalcLinkPoses
            )
        .def("CalcLinkPoseAsMultilinearPolynomials",
            &Class::CalcLinkPoseAsMultilinearPolynomial, py::arg("q_star"),
            py::arg("link_index"), py::arg("expressed_body_index")
            //             cls_doc.CalcLinkPoses
        );
  }  // RationalForwardKinematicsOld Class
  // RationalForwardKinematicsOld Util methods
  {
    //      constexpr auto& cls_doc =
    //      doc.rational_forward_kinematics_old.generate_monomial_basis;
    m.def("GenerateMonomialBasisWithOrderUpToOne",
        &GenerateMonomialBasisWithOrderUpToOne, py::arg("t_angles"));
    m.def("GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo",
        &GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo,
        py::arg("t_angles"));
  }  // RationalForwardKinematicsOld Util methods

  // find link in middle of body
  m.def("FindBodyInTheMiddleOfChainOld",
      &drake::multibody::rational_old::internal::FindBodyInTheMiddleOfChain,
      py::arg("plant"), py::arg("start"), py::arg("end"));

  // Pose
  py::enum_<multibody::rational_old::CollisionGeometryType>(
      m, "CollisionGeometryTypeOld")
      .value("kPolytope",
          multibody::rational_old::CollisionGeometryType::kPolytope)
      .value("kSphere", multibody::rational_old::CollisionGeometryType::kSphere)
      .value(
          "kCapsule", multibody::rational_old::CollisionGeometryType::kCapsule);

  py::class_<multibody::rational_old::CollisionGeometry>(
      m, "CollisionGeometryOld")
      .def("type", &CollisionGeometry::type)
      .def("geometry", &CollisionGeometry::geometry, py_rvp::reference_internal)
      .def("body_index", &CollisionGeometry::body_index)
      .def("id", &CollisionGeometry::id);

  m.def("GetVerticesOld", &GetVertices, py::arg("shape"));

  py::enum_<multibody::rational_old::SeparatingPlaneOrder>(
      m, "SeparatingPlaneOrderOld")
      .value(
          "kConstant", multibody::rational_old::SeparatingPlaneOrder::kConstant)
      .value("kAffine", multibody::rational_old::SeparatingPlaneOrder::kAffine);

  // PlaneSide
  py::enum_<PlaneSide>(m, "PlaneSideOld")
      .value("kPositive", PlaneSide::kPositive)
      .value("kNegative", PlaneSide::kNegative);

  py::class_<VerificationOption>(m, "VerificationOptionOld")
      .def(py::init<>())
      .def_readonly(
          "link_polynomial_type", &VerificationOption::link_polynomial_type)
      .def_readonly("lagrangian_type", &VerificationOption::lagrangian_type);

  // LinkOnPlaneSideRational
  py::class_<LinkOnPlaneSideRational>(m, "LinkOnPlaneSideRational")
      .def_readonly("rational", &LinkOnPlaneSideRational::rational)
      .def_readonly("link_geometry", &LinkOnPlaneSideRational::link_geometry)
      .def_readonly("expressed_body_index",
          &LinkOnPlaneSideRational::expressed_body_index)
      .def_readonly("other_side_link_geometry",
          &LinkOnPlaneSideRational::other_side_link_geometry)
      .def_readonly("a_A", &LinkOnPlaneSideRational::a_A, py_rvp::copy)
      .def_readonly("b", &LinkOnPlaneSideRational::b)
      .def_readonly("plane_side", &LinkOnPlaneSideRational::plane_side)
      .def_readonly("plane_order", &LinkOnPlaneSideRational::plane_order)
      .def_readonly("lorentz_cone_constraints",
          &LinkOnPlaneSideRational::lorentz_cone_constraints);

  // CspaceRegionType
  py::enum_<CspaceRegionType>(m, "CspaceRegionType")
      .value("kGenericPolytope", CspaceRegionType::kGenericPolytope)
      .value(
          "kAxisAlignedBoundingBox", CspaceRegionType::kAxisAlignedBoundingBox);

  // EllipsoidVolume
  py::enum_<EllipsoidVolume>(m, "EllipsoidVolume")
      .value("kLog", EllipsoidVolume::kLog)
      .value("kNthRoot", EllipsoidVolume::kNthRoot);

  // BilinearAlternationOption
  py::class_<CspaceFreeRegion::BilinearAlternationOption>(
      m, "BilinearAlternationOption")
      .def(py::init<>())
      .def_readwrite(
          "max_iters", &CspaceFreeRegion::BilinearAlternationOption::max_iters)
      .def_readwrite("convergence_tol",
          &CspaceFreeRegion::BilinearAlternationOption::convergence_tol)
      .def_readwrite("lagrangian_backoff_scale",
          &CspaceFreeRegion::BilinearAlternationOption::
              lagrangian_backoff_scale)
      .def_readwrite("polytope_backoff_scale",
          &CspaceFreeRegion::BilinearAlternationOption::polytope_backoff_scale)
      .def_readwrite(
          "verbose", &CspaceFreeRegion::BilinearAlternationOption::verbose)
      .def_readwrite("redundant_tighten",
          &CspaceFreeRegion::BilinearAlternationOption::redundant_tighten)
      .def_readwrite("compute_polytope_volume",
          &CspaceFreeRegion::BilinearAlternationOption::compute_polytope_volume)
      .def_readwrite("ellipsoid_volume",
          &CspaceFreeRegion::BilinearAlternationOption::ellipsoid_volume)
      .def_readwrite("num_threads",
          &CspaceFreeRegion::BilinearAlternationOption::num_threads);

  // BinarySearchOption
  py::class_<CspaceFreeRegion::BinarySearchOption>(m, "BinarySearchOption")
      .def(py::init<>())
      .def_readwrite(
          "epsilon_max", &CspaceFreeRegion::BinarySearchOption::epsilon_max)
      .def_readwrite("verbose", &CspaceFreeRegion::BinarySearchOption::verbose)
      .def_readwrite("lagrangian_backoff_scale",
          &CspaceFreeRegion::BinarySearchOption::lagrangian_backoff_scale)
      .def_readwrite(
          "epsilon_min", &CspaceFreeRegion::BinarySearchOption::epsilon_min)
      .def_readwrite(
          "max_iters", &CspaceFreeRegion::BinarySearchOption::max_iters)
      .def_readwrite(
          "search_d", &CspaceFreeRegion::BinarySearchOption::search_d)
      .def_readwrite("compute_polytope_volume",
          &CspaceFreeRegion::BinarySearchOption::compute_polytope_volume)
      .def_readwrite("verbose", &CspaceFreeRegion::BinarySearchOption::verbose)
      .def_readwrite(
          "num_threads", &CspaceFreeRegion::BinarySearchOption::num_threads);

  // CspaceFreeRegionSolution
  py::class_<CspaceFreeRegionSolution>(m, "CspaceFreeRegionSolution")
      .def_readwrite("C", &CspaceFreeRegionSolution::C)
      .def_readwrite("d", &CspaceFreeRegionSolution::d)
      .def_readwrite("P", &CspaceFreeRegionSolution::P)
      .def_readwrite("q", &CspaceFreeRegionSolution::q)
      .def_readwrite(
          "separating_planes", &CspaceFreeRegionSolution::separating_planes);
  // CspaceFreeRegion
  py::class_<CspaceFreeRegion> cspace_cls(m, "CspaceFreeRegion");

  cspace_cls.def(py::init<const systems::Diagram<double>&,
                     const multibody::MultibodyPlant<double>*,
                     const geometry::SceneGraph<double>*, SeparatingPlaneOrder,
                     CspaceRegionType, double>(),
      py::arg("diagram"), py::arg("plant"), py::arg("scene_graph"),
      py::arg("plane_order"), py::arg("cspace_region_type"),
      py::arg("separating_polytope_delta") = 1.);

  cspace_cls
      .def("map_geometries_to_separating_planes",
          &CspaceFreeRegion::map_geometries_to_separating_planes)
      .def("GenerateRationalsForLinkOnOneSideOfPlane",
          &CspaceFreeRegion::GenerateRationalsForLinkOnOneSideOfPlane,
          py::arg("q_star"), py::arg("filtered_collision_pairs"))
      .def_property_readonly("rational_forward_kinematics",
          &CspaceFreeRegion::rational_forward_kinematics);

  // CspacePolytopeTuple
  py::class_<CspaceFreeRegion::CspacePolytopeTuple>(
      cspace_cls, "CspacePolytopeTuple")
      .def_readonly("rational_numerator",
          &CspaceFreeRegion::CspacePolytopeTuple::rational_numerator)
      .def_readonly("polytope_lagrangian_gram_lower_start",
          &CspaceFreeRegion::CspacePolytopeTuple::
              polytope_lagrangian_gram_lower_start)
      .def_readonly("t_lower_lagrangian_gram_lower_start",
          &CspaceFreeRegion::CspacePolytopeTuple::
              t_lower_lagrangian_gram_lower_start)
      .def_readonly("t_upper_lagrangian_gram_lower_start",
          &CspaceFreeRegion::CspacePolytopeTuple::
              t_upper_lagrangian_gram_lower_start)
      .def_readonly("verified_polynomial_gram_lower_start",
          &CspaceFreeRegion::CspacePolytopeTuple::
              verified_polynomial_gram_lower_start)
      .def_readonly("monomial_basis",
          &CspaceFreeRegion::CspacePolytopeTuple::monomial_basis);

  cspace_cls
      // note that the output of this function is now useless as
      // separating_plane_to_lorentz_cone_constraints can return as a list of
      // empty lists and so has the wrong type to pass to
      // ConstructLagrangainProgram
      .def(
          "GenerateTuplesForBilinearAlternation",
          [](const CspaceFreeRegion* self,
              const Eigen::Ref<const Eigen::VectorXd>& q_star,
              const CspaceFreeRegion::FilteredCollisionPairs&
                  filtered_collision_pairs,
              int C_rows) {
            std::vector<CspaceFreeRegion::CspacePolytopeTuple>
                alternation_tuples;
            VectorX<symbolic::Polynomial> d_minus_Ct;
            Eigen::VectorXd t_lower;
            Eigen::VectorXd t_upper;
            VectorX<symbolic::Polynomial> t_minus_t_lower;
            VectorX<symbolic::Polynomial> t_upper_minus_t;
            MatrixX<symbolic::Variable> C;
            VectorX<symbolic::Variable> d;
            VectorX<symbolic::Variable> lagrangian_gram_vars;
            VectorX<symbolic::Variable> verified_gram_vars;
            VectorX<symbolic::Variable> separating_plane_vars;
            std::vector<std::vector<int>> separating_plane_to_tuples;
            std::vector<
                std::vector<solvers::Binding<solvers::LorentzConeConstraint>>>
                separating_plane_to_lorentz_cone_constraints;
            self->GenerateTuplesForBilinearAlternation(q_star,
                filtered_collision_pairs, C_rows, &alternation_tuples,
                &d_minus_Ct, &t_lower, &t_upper, &t_minus_t_lower,
                &t_upper_minus_t, &C, &d, &lagrangian_gram_vars,
                &verified_gram_vars, &separating_plane_vars,
                &separating_plane_to_tuples,
                &separating_plane_to_lorentz_cone_constraints);
            return std::make_tuple(alternation_tuples, d_minus_Ct, t_lower,
                t_upper, t_minus_t_lower, t_upper_minus_t, C, d,
                lagrangian_gram_vars, verified_gram_vars, separating_plane_vars,
                separating_plane_to_tuples,
                separating_plane_to_lorentz_cone_constraints);
          },
          py::arg("q_star"), py::arg("filtered_collision_pairs"),
          py::arg("C_rows"))
      .def(
          "ConstructLagrangianProgram",
          [](const CspaceFreeRegion* self,
              const std::vector<CspaceFreeRegion::CspacePolytopeTuple>&
                  alternation_tuples,
              const Eigen::Ref<const Eigen::MatrixXd>& C,
              const Eigen::Ref<const Eigen::VectorXd>& d,
              const VectorX<symbolic::Variable>& lagrangian_gram_vars,
              const VectorX<symbolic::Variable>& verified_gram_vars,
              const VectorX<symbolic::Variable>& separating_plane_vars,
              const std::vector<
                  solvers::Binding<solvers::LorentzConeConstraint>>&
                  separating_plane_lorentz_cone_constraints,
              const Eigen::Ref<const Eigen::VectorXd>& t_lower,
              const Eigen::Ref<const Eigen::VectorXd>& t_upper,
              const VerificationOption& option,
              std::optional<double> redundant_tighten) {
            auto prog = self->ConstructLagrangianProgram(alternation_tuples, C,
                d, lagrangian_gram_vars, verified_gram_vars,
                separating_plane_vars,
                separating_plane_lorentz_cone_constraints, t_lower, t_upper,
                option, redundant_tighten, nullptr, nullptr);
            return prog;
          },
          py::arg("alternation_tuples"), py::arg("C"), py::arg("d"),
          py::arg("lagrangian_gram_vars"), py::arg("verified_gram_vars"),
          py::arg("separating_plane_vars"),
          py::arg("separating_plane_lorentz_cone_constraints"),
          py::arg("t_lower"), py::arg("t_upper"), py::arg("option"),
          py::arg("redundant_tighten") = std::nullopt)
      .def("ConstructPolytopeProgram",
          &CspaceFreeRegion::ConstructPolytopeProgram,
          py::arg("alternation_tuples"), py::arg("C"), py::arg("d"),
          py::arg("d_minus_Ct"), py::arg("lagrangian_gram_var_vals"),
          py::arg("verified_gram_vars"), py::arg("separating_plane_vars"),
          py::arg("separating_plane_lorentz_cone_constraints"),
          py::arg("t_minus_t_lower"), py::arg("t_upper_minus_t"),
          py::arg("option"))
      .def(
          "CspacePolytopeBilinearAlternation",
          [](const CspaceFreeRegion* self,
              const Eigen::Ref<const Eigen::VectorXd>& q_star,
              const CspaceFreeRegion::FilteredCollisionPairs&
                  filtered_collision_pairs,
              const Eigen::Ref<const Eigen::MatrixXd>& C_init,
              const Eigen::Ref<const Eigen::VectorXd>& d_init,
              const CspaceFreeRegion::BilinearAlternationOption&
                  bilinear_alternation_option,
              const solvers::SolverOptions& solver_options,
              const std::optional<Eigen::MatrixXd>& t_inner_pts,
              const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
                  inner_polytope) {
            CspaceFreeRegionSolution cspace_free_region_solution;
            std::vector<double> polytope_volumes;
            std::vector<double> ellipsoid_determinants;
            self->CspacePolytopeBilinearAlternation(q_star,
                filtered_collision_pairs, C_init, d_init,
                bilinear_alternation_option, solver_options, t_inner_pts,
                inner_polytope, &cspace_free_region_solution, &polytope_volumes,
                &ellipsoid_determinants);
            // TODO(Alex.Amice) reconcile this binding and
            // CspacePolytopeBinarySearch returns
            return std::make_tuple(cspace_free_region_solution,
                polytope_volumes, ellipsoid_determinants);
          },
          py::arg("q_star"), py::arg("filtered_collision_pairs"),
          py::arg("C_init"), py::arg("d_init"),
          py::arg("bilinear_alternation_option"), py::arg("solver_options"),
          py::arg("t_inner_pts") = std::nullopt,
          py::arg("inner_polytope") = std::nullopt)
      .def(
          "CspacePolytopeBinarySearch",
          [](const CspaceFreeRegion* self,
              const Eigen::Ref<const Eigen::VectorXd>& q_star,
              const CspaceFreeRegion::FilteredCollisionPairs&
                  filtered_collision_pairs,
              const Eigen::Ref<const Eigen::MatrixXd>& C,
              const Eigen::Ref<const Eigen::VectorXd>& d_init,
              const CspaceFreeRegion::BinarySearchOption& binary_search_option,
              const solvers::SolverOptions& solver_options,
              const std::optional<Eigen::MatrixXd>& t_inner_pts,
              const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
                  inner_polytope) {
            CspaceFreeRegionSolution cspace_free_region_solution;
            self->CspacePolytopeBinarySearch(q_star, filtered_collision_pairs,
                C, d_init, binary_search_option, solver_options, t_inner_pts,
                inner_polytope, &cspace_free_region_solution);
            // TODO(Alex.Amice) reconcile this binding and
            // CspacePolytopeBilinearAlternation returns
            return cspace_free_region_solution;
          },
          py::arg("q_star"), py::arg("filtered_collision_pairs"), py::arg("C"),
          py::arg("d_init"), py::arg("binary_search_option"),
          py::arg("solver_options"), py::arg("t_inner_pts") = std::nullopt,
          py::arg("inner_polytope") = std::nullopt)
      .def("IsPostureInCollision", &CspaceFreeRegion::IsPostureInCollision)
      .def("separating_planes", &CspaceFreeRegion::separating_planes,
          py_rvp::reference);

  m.def("GetCollisionGeometries", &GetCollisionGeometries, py::arg("diagram"),
      py::arg("plant"), py::arg("scene_graph"));

  m.def("AddInscribedEllipsoid", &AddInscribedEllipsoid, py::arg("prog"),
       py::arg("C"), py::arg("d"), py::arg("t_lower"), py::arg("t_upper"),
       py::arg("P"), py::arg("q"), py::arg("constrain_P_psd") = true)
      .def(
          "AddInscribedEllipsoid",
          [](solvers::MathematicalProgram* prog,
              const Eigen::Ref<const Eigen::MatrixXd>& C,
              const Eigen::Ref<const Eigen::VectorXd>& d,
              const Eigen::Ref<const Eigen::VectorXd>& t_lower,
              const Eigen::Ref<const Eigen::VectorXd>& t_upper,
              bool constrain_P_psd) {
            const auto P =
                prog->NewSymmetricContinuousVariables(t_lower.rows(), "P");
            const auto q = prog->NewContinuousVariables(t_lower.rows(), "q");
            AddInscribedEllipsoid(
                prog, C, d, t_lower, t_upper, P, q, constrain_P_psd);
            return std::make_tuple(P, q);
          },
          py::arg("prog"), py::arg("C"), py::arg("d"), py::arg("t_lower"),
          py::arg("t_upper"), py::arg("constrain_P_psd") = true);

  m.def(
       "AddCspacePolytopeContainment",
       [](solvers::MathematicalProgram* prog,
           const MatrixX<symbolic::Variable>& C,
           const VectorX<symbolic::Variable>& d,
           const Eigen::Ref<const Eigen::MatrixXd>& C_inner,
           const Eigen::Ref<const Eigen::VectorXd>& d_inner,
           const Eigen::Ref<const Eigen::VectorXd>& t_lower,
           const Eigen::Ref<const Eigen::VectorXd>& t_upper) {
         return AddCspacePolytopeContainment(
             prog, C, d, C_inner, d_inner, t_lower, t_upper);
       },
       py::arg("prog"), py::arg("C"), py::arg("d"), py::arg("C_inner"),
       py::arg("d_inner"), py::arg("t_lower"), py::arg("t_upper"))
      .def(
          "AddCspacePolytopeContainment",
          [](solvers::MathematicalProgram* prog,
              const MatrixX<symbolic::Variable>& C,
              const VectorX<symbolic::Variable>& d,
              const Eigen::Ref<const Eigen::MatrixXd>& inner_pts) {
            return AddCspacePolytopeContainment(prog, C, d, inner_pts);
          },
          py::arg("prog"), py::arg("C"), py::arg("d"), py::arg("inner_pts"))
      .def(
          "FindRedundantInequalities",
          [](const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
              const Eigen::VectorXd& t_lower, const Eigen::VectorXd& t_upper,
              double tighten) {
            std::unordered_set<int> C_redundant_indices,
                t_lower_redundant_indices, t_upper_redundant_indices;
            FindRedundantInequalities(C, d, t_lower, t_upper, tighten,
                &C_redundant_indices, &t_lower_redundant_indices,
                &t_upper_redundant_indices);
            return std::make_tuple(C_redundant_indices,
                t_lower_redundant_indices, t_upper_redundant_indices);
          },
          py::arg("C"), py::arg("d"), py::arg("t_lower"), py::arg("t_upper"),
          py::arg("tighten"))
      .def("FindEpsilonLower", &FindEpsilonLower, py::arg("C"), py::arg("d"),
          py::arg("t_lower"), py::arg("t_upper"),
          py::arg("t_inner_pts") = std::nullopt,
          py::arg("inner_polytope") = std::nullopt);

  m.def("CalcCspacePolytopeVolume", &CalcCspacePolytopeVolume, py::arg("C"),
      py::arg("d"), py::arg("t_lower"), py::arg("t_upper"));
  m.def(
      "ReadCspacePolytopeFromFile",
      [](const std::string& filename,
          const multibody::MultibodyPlant<double>& plant,
          const geometry::SceneGraphInspector<double>& inspector) {
        Eigen::MatrixXd C;
        Eigen::VectorXd d;
        std::unordered_map<SortedPair<geometry::GeometryId>,
            std::pair<multibody::BodyIndex, Eigen::VectorXd>>
            separating_planes;
        ReadCspacePolytopeFromFile(
            filename, plant, inspector, &C, &d, &separating_planes);
        return std::make_tuple(C, d, separating_planes);
      },
      py::arg("filename"), py::arg("plant"), py::arg("scene_graph"));

  // CspaceFreeLine
  py::class_<CspaceFreeLine>(m, "CspaceFreeLine")
      .def(py::init<const systems::Diagram<double>&,
               const multibody::MultibodyPlant<double>*,
               const geometry::SceneGraph<double>*,
               multibody::rational_old::SeparatingPlaneOrder,
               std::optional<Eigen::VectorXd>,
               const CspaceFreeRegion::FilteredCollisionPairs&,
               const multibody::rational_old::VerificationOption&>(),
          py::arg("diagram"), py::arg("plant"), py::arg("scene_graph"),
          py::arg("plane_order"), py::arg("q_star") = std::nullopt,
          py::arg("filtered_collision_pairs") =
              CspaceFreeRegion::FilteredCollisionPairs(),
          py::arg("option") = VerificationOption())
      .def("get_mu", &CspaceFreeLine::get_mu)
      .def("get_s0", &CspaceFreeLine::get_s0)
      .def("get_s1", &CspaceFreeLine::get_s1)
      .def("get_tuples", &CspaceFreeLine::get_tuples)
      .def("separating_planes", &CspaceFreeLine::separating_planes)
      .def("map_geometries_to_separating_planes",
            [](const CspaceFreeLine* self) {
              // Template deduction for drake::SortedPair<GeometryId> does not
              // work. Here we manually make a map of tuples instead.
              py::dict ret;
              for (auto [k, v] : self->map_geometries_to_separating_planes()) {
                ret[py::make_tuple(k.first(), k.second())] = v;
              }
              return ret;
            })
      .def("separating_plane_to_tuples",
          &CspaceFreeLine::separating_plane_to_tuples)
      .def("MakeCertificationProgram",
          &CspaceFreeLine::MakeCertificationProgram, py_rvp::reference_internal,
          py::arg("s0"), py::arg("s1"))
      .def(
          "CertifyTangentConfigurationSpaceLine",
          [](CspaceFreeLine* self, const Eigen::Ref<const Eigen::VectorXd>& s0,
              const Eigen::Ref<const Eigen::VectorXd>& s1,
              const solvers::SolverOptions& solver_options) {
            std::vector<SeparatingPlane<double>> separating_planes_sol;
            bool safe = self->CertifyTangentConfigurationSpaceLine(
                s0, s1, solver_options, &separating_planes_sol);
            return std::make_tuple(safe, separating_planes_sol);
          },
          py::arg("s0"), py::arg("s1"),
          py::arg("solver_options") = solvers::SolverOptions())
      .def(
          "CertifyTangentConfigurationSpaceLines",
          [](CspaceFreeLine* self, const Eigen::Ref<const Eigen::MatrixXd>& s0,
              const Eigen::Ref<const Eigen::MatrixXd>& s1,
              const solvers::SolverOptions& solver_options) {
            std::vector<std::vector<SeparatingPlane<double>>>
                separating_planes_sol;
            std::vector<bool> safe =
                self->CertifyTangentConfigurationSpaceLines(
                    s0, s1, solver_options, &separating_planes_sol);
            return std::make_tuple(safe, separating_planes_sol);
          },
          py::arg("s0"), py::arg("s1"),
          py::arg("solver_options") = solvers::SolverOptions());
  type_pack<symbolic::Polynomial, symbolic::RationalFunction> sym_pack;
  type_visit([m](auto dummy) { DoPoseDeclaration(m, dummy); }, sym_pack);
  type_visit([m](auto dummy) { DoScalarDependentDefinitions(m, dummy); },
      type_pack<double, symbolic::Variable>());
}

}  // namespace pydrake
}  // namespace drake
