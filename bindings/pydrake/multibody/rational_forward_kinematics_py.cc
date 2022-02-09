//
// Created by amice on 11/8/21.
//
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/value_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"
#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/multibody/rational_forward_kinematics/centrally_symmetric_hpolytope.h"
#include "drake/multibody/rational_forward_kinematics/maximal_epsilon_expansion.h"

namespace drake {
namespace pydrake {

template <typename T>

void DoPoseDeclaration(py::module m, T) {
  py::tuple param = GetPyParam<T>();
  using Class = multibody::RationalForwardKinematics::Pose<T>;
  auto cls = DefineTemplateClassWithDefault<Class>(
      m, "RationalForwardKinematicsPose", param);
  cls.def("translation", [](const Class& self) { return self.p_AB; })
      .def("rotation", [](const Class& self) { return self.R_AB; })
      .def_readwrite("frame_A_index",
          &multibody::RationalForwardKinematics::Pose<T>::frame_A_index)
      .def("asRigidTransformExpr",
          [](const Class& self) { return self.asRigidTransformExpression(); });

  DefCopyAndDeepCopy(&cls);
}

PYBIND11_MODULE(rational_forward_kinematics, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::multibody;
  //  constexpr auto& doc = pydrake_doc.drake.multibody;
  m.doc() = "RationalForwardKinematics module";

  py::module::import("pydrake.math");
  py::module::import("pydrake.multibody.plant");
  // RationalForwardKinematics Class
  {
    using Class = RationalForwardKinematics;
    // no class docs built
    //    constexpr auto& cls_doc = doc.RationalForwardKinematics;
    py::class_<Class>(m, "RationalForwardKinematics")
        .def(py::init<const MultibodyPlant<double>&>(), py::arg("plant"),
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
  }  // RationalForwardKinematics Class
  // RationalForwardKinematics Util methods
  {
    //      constexpr auto& cls_doc =
    //      doc.rational_forward_kinematics.generate_monomial_basis;
    m.def("GenerateMonomialBasisWithOrderUpToOne",
        &GenerateMonomialBasisWithOrderUpToOne, py::arg("t_angles"));
    m.def("GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo",
        &GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo,
        py::arg("t_angles"));
  }  // RationalForwardKinematics Util methods

  //  type_visit([m](auto dummy) { DoPoseDeclaration(m, dummy); },
  //      CommonScalarPack{});
  type_pack<symbolic::Polynomial, symbolic::RationalFunction> sym_pack;
  type_visit([m](auto dummy) { DoPoseDeclaration(m, dummy); }, sym_pack);

  // find link in middle of body
  m.def("FindBodyInTheMiddleOfChain",
      &drake::multibody::internal::FindBodyInTheMiddleOfChain, py::arg("plant"),
      py::arg("start"), py::arg("end"));

  // Pose
  constexpr auto& doc = pydrake_doc.drake.multibody;
  py::class_<multibody::ConvexPolytope>(
      m, "ConvexPolytope", doc.ConvexPolytope.doc)
      .def(py::init<BodyIndex, geometry::GeometryId,
               const Eigen::Ref<const Eigen::Matrix3Xd>>(),
          py::arg("body_index"), py::arg("id"), py::arg("vertices"),
          doc.ConvexPolytope.doc)
      .def("p_BV", &ConvexPolytope::p_BV, doc.ConvexPolytope.p_BV.doc)
      .def("get_id", &ConvexPolytope::get_id, doc.ConvexGeometry.get_id.doc)
      .def("body_index", &ConvexPolytope::body_index,
          doc.ConvexGeometry.body_index.doc);

  py::enum_<multibody::SeparatingPlaneOrder>(
      m, "SeparatingPlaneOrder", doc.SeparatingPlaneOrder.doc)
      .value("kConstant", multibody::SeparatingPlaneOrder::kConstant,
          doc.SeparatingPlaneOrder.kConstant.doc)
      .value("kAffine", multibody::SeparatingPlaneOrder::kAffine,
          doc.SeparatingPlaneOrder.kAffine.doc);

  // SeparatingPlane
  py::class_<multibody::SeparatingPlane>(
      m, "SeparatingPlane", doc.SeparatingPlane.doc)
      .def_readonly("a", &SeparatingPlane::a, py_rvp::copy, doc.SeparatingPlane.a.doc)
      .def_readonly("b", &SeparatingPlane::b, doc.SeparatingPlane.b.doc)
      .def_readonly("positive_side_polytope",
          &SeparatingPlane::positive_side_polytope,
          doc.SeparatingPlane.positive_side_polytope.doc)
      .def_readonly("negative_side_polytope",
          &SeparatingPlane::negative_side_polytope,
          doc.SeparatingPlane.negative_side_polytope.doc)
      .def_readonly("expressed_link", &SeparatingPlane::expressed_link,
          doc.SeparatingPlane.expressed_link.doc)
      .def_readonly(
          "order", &SeparatingPlane::order, doc.SeparatingPlane.order.doc)
      .def_readonly("decision_variables", &SeparatingPlane::decision_variables, py_rvp::copy, doc.SeparatingPlane.a.doc);

  // PlaneSide
  py::enum_<PlaneSide>(m, "PlaneSide", doc.PlaneSide.doc)
      .value("kPositive", PlaneSide::kPositive)
      .value("kNegative", PlaneSide::kNegative);

  py::class_<VerificationOption>(
      m, "VerificationOption", doc.VerificationOption.doc)
      .def(py::init<>())
      .def_readonly("link_polynomial_type",
          &VerificationOption::link_polynomial_type,
          doc.VerificationOption.link_polynomial_type.doc)
      .def_readonly("lagrangian_type", &VerificationOption::lagrangian_type,
          doc.VerificationOption.lagrangian_type.doc);

  // LinkVertexOnPlaneSideRational
  py::class_<LinkVertexOnPlaneSideRational>(
      m, "LinkVertexOnPlaneSideRational", doc.LinkVertexOnPlaneSideRational.doc)
      .def_readonly("rational", &LinkVertexOnPlaneSideRational::rational,
          doc.LinkVertexOnPlaneSideRational.rational.doc)
      .def_readonly("link_polytope",
          &LinkVertexOnPlaneSideRational::link_polytope,
          doc.LinkVertexOnPlaneSideRational.link_polytope.doc)
      .def_readonly("expressed_body_index",
          &LinkVertexOnPlaneSideRational::expressed_body_index,
          doc.LinkVertexOnPlaneSideRational.expressed_body_index.doc)
      .def_readonly("other_side_link_polytope",
          &LinkVertexOnPlaneSideRational::other_side_link_polytope,
          doc.LinkVertexOnPlaneSideRational.other_side_link_polytope.doc)
      .def_readonly("a_A", &LinkVertexOnPlaneSideRational::a_A,
          doc.LinkVertexOnPlaneSideRational.a_A.doc)
      .def_readonly("b", &LinkVertexOnPlaneSideRational::b,
          doc.LinkVertexOnPlaneSideRational.b.doc)
      .def_readonly("plane_side", &LinkVertexOnPlaneSideRational::plane_side,
          doc.LinkVertexOnPlaneSideRational.plane_side.doc)
      .def_readonly("plane_order", &LinkVertexOnPlaneSideRational::plane_order,
          doc.LinkVertexOnPlaneSideRational.plane_order.doc);

  // CspaceRegionType
  py::enum_<CspaceRegionType>(m, "CspaceRegionType", doc.CspaceRegionType.doc)
      .value("kGenericPolytope", CspaceRegionType::kGenericPolytope)
      .value(
          "kAxisAlignedBoundingBox", CspaceRegionType::kAxisAlignedBoundingBox);

  // EllipsoidVolume
  py::enum_<EllipsoidVolume>(m, "EllipsoidVolume", doc.EllipsoidVolume.doc)
      .value("kLog", EllipsoidVolume::kLog)
      .value("kNthRoot", EllipsoidVolume::kNthRoot);

  // BilinearAlternationOption
  py::class_<CspaceFreeRegion::BilinearAlternationOption>(m,
      "BilinearAlternationOption",
      doc.CspaceFreeRegion.BilinearAlternationOption.doc)
      .def(py::init<>())
      .def_readwrite("max_iters",
          &CspaceFreeRegion::BilinearAlternationOption::max_iters,
          doc.CspaceFreeRegion.BilinearAlternationOption.max_iters.doc)
      .def_readwrite("convergence_tol",
          &CspaceFreeRegion::BilinearAlternationOption::convergence_tol,
          doc.CspaceFreeRegion.BilinearAlternationOption.convergence_tol.doc)
      .def_readwrite("lagrangian_backoff_scale",
          &CspaceFreeRegion::BilinearAlternationOption::
              lagrangian_backoff_scale,
          doc.CspaceFreeRegion.BilinearAlternationOption
              .lagrangian_backoff_scale.doc)
      .def_readwrite("polytope_backoff_scale",
          &CspaceFreeRegion::BilinearAlternationOption::polytope_backoff_scale,
          doc.CspaceFreeRegion.BilinearAlternationOption.polytope_backoff_scale
              .doc)
      .def_readwrite("verbose",
          &CspaceFreeRegion::BilinearAlternationOption::verbose,
          doc.CspaceFreeRegion.BilinearAlternationOption.verbose.doc)
      .def_readwrite("redundant_tighten",
          &CspaceFreeRegion::BilinearAlternationOption::redundant_tighten,
          doc.CspaceFreeRegion.BilinearAlternationOption.redundant_tighten.doc)
      .def_readwrite("compute_polytope_volume",
          &CspaceFreeRegion::BilinearAlternationOption::compute_polytope_volume,
          doc.CspaceFreeRegion.BilinearAlternationOption.compute_polytope_volume
              .doc)
      .def_readwrite("ellipsoid_volume",
          &CspaceFreeRegion::BilinearAlternationOption::ellipsoid_volume,
          doc.CspaceFreeRegion.BilinearAlternationOption.ellipsoid_volume.doc)
      .def_readwrite("num_threads",
          &CspaceFreeRegion::BilinearAlternationOption::num_threads,
          doc.CspaceFreeRegion.BilinearAlternationOption.num_threads.doc);

  // BinarySearchOption
  py::class_<CspaceFreeRegion::BinarySearchOption>(
      m, "BinarySearchOption", doc.CspaceFreeRegion.BinarySearchOption.doc)
      .def(py::init<>())
      .def_readwrite("epsilon_max",
          &CspaceFreeRegion::BinarySearchOption::epsilon_max,
          doc.CspaceFreeRegion.BinarySearchOption.epsilon_max.doc)
      .def_readwrite("verbose", &CspaceFreeRegion::BinarySearchOption::verbose,
          doc.CspaceFreeRegion.BinarySearchOption.verbose.doc)
      .def_readwrite("lagrangian_backoff_scale",
          &CspaceFreeRegion::BinarySearchOption::lagrangian_backoff_scale,
          doc.CspaceFreeRegion.BinarySearchOption.lagrangian_backoff_scale.doc)
      .def_readwrite("epsilon_min",
          &CspaceFreeRegion::BinarySearchOption::epsilon_min,
          doc.CspaceFreeRegion.BinarySearchOption.epsilon_min.doc)
      .def_readwrite("max_iters",
          &CspaceFreeRegion::BinarySearchOption::max_iters,
          doc.CspaceFreeRegion.BinarySearchOption.max_iters.doc)
      .def_readwrite("search_d",
          &CspaceFreeRegion::BinarySearchOption::search_d,
          doc.CspaceFreeRegion.BinarySearchOption.search_d.doc)
      .def_readwrite("compute_polytope_volume",
          &CspaceFreeRegion::BinarySearchOption::compute_polytope_volume,
          doc.CspaceFreeRegion.BinarySearchOption.compute_polytope_volume.doc)
      .def_readwrite("verbose", &CspaceFreeRegion::BinarySearchOption::verbose,
          doc.CspaceFreeRegion.BinarySearchOption.verbose.doc)
      .def_readwrite("num_threads",
          &CspaceFreeRegion::BinarySearchOption::num_threads,
          doc.CspaceFreeRegion.BinarySearchOption.num_threads.doc);

  //CspaceFreeRegionSolution
  py::class_<CspaceFreeRegionSolution>(m, "CspaceFreeRegionSolution", doc.CspaceFreeRegionSolution.doc)
      .def_readwrite("C", &CspaceFreeRegionSolution::C, doc.CspaceFreeRegionSolution.C.doc)
      .def_readwrite("d", &CspaceFreeRegionSolution::d, doc.CspaceFreeRegionSolution.d.doc)
      .def_readwrite("P", &CspaceFreeRegionSolution::P, doc.CspaceFreeRegionSolution.P.doc)
      .def_readwrite("q", &CspaceFreeRegionSolution::q, doc.CspaceFreeRegionSolution.q.doc)
      .def_readwrite("separating_planes", &CspaceFreeRegionSolution::separating_planes,
                     doc.CspaceFreeRegionSolution.separating_planes.doc);
  // VectorBisectionSearchOption
  py::class_<CspaceFreeRegion::VectorBisectionSearchOption>(m,
      "VectorBisectionSearchOption",
      doc.CspaceFreeRegion.VectorBisectionSearchOption.doc)
      .def(py::init<>())
      .def_readwrite("epsilon_max",
          &CspaceFreeRegion::VectorBisectionSearchOption::epsilon_max,
          doc.CspaceFreeRegion.VectorBisectionSearchOption.epsilon_max.doc)
      .def_readwrite("epsilon_min",
          &CspaceFreeRegion::VectorBisectionSearchOption::epsilon_min,
          doc.CspaceFreeRegion.VectorBisectionSearchOption.epsilon_min.doc)
      .def_readwrite("max_iters",
          &CspaceFreeRegion::VectorBisectionSearchOption::max_iters,
          doc.CspaceFreeRegion.VectorBisectionSearchOption.max_iters.doc)
      .def_readwrite("search_d",
          &CspaceFreeRegion::VectorBisectionSearchOption::search_d,
          doc.CspaceFreeRegion.VectorBisectionSearchOption.search_d.doc)
      .def_readwrite("max_feasible_iters",
          &CspaceFreeRegion::VectorBisectionSearchOption::
              max_feasible_iters,
          doc.CspaceFreeRegion.VectorBisectionSearchOption
              .max_feasible_iters.doc)
      .def_readwrite("num_threads",
          &CspaceFreeRegion::VectorBisectionSearchOption::
              num_threads,
          doc.CspaceFreeRegion.VectorBisectionSearchOption
              .num_threads.doc);

  //interleavedRegionSearchOptions
   py::class_<CspaceFreeRegion::InterleavedRegionSearchOptions>(m,
      "InterleavedRegionSearchOptions",
      doc.CspaceFreeRegion.InterleavedRegionSearchOptions.doc)
      .def(py::init<>())
      .def_readwrite("vector_bisection_search_options",
          &CspaceFreeRegion::InterleavedRegionSearchOptions::vector_bisection_search_options,
          doc.CspaceFreeRegion.InterleavedRegionSearchOptions.vector_bisection_search_options.doc)
      .def_readwrite("scalar_binary_search_options",
          &CspaceFreeRegion::InterleavedRegionSearchOptions::scalar_binary_search_options,
          doc.CspaceFreeRegion.InterleavedRegionSearchOptions.scalar_binary_search_options.doc)
      .def_readwrite("bilinear_alternation_options",
          &CspaceFreeRegion::InterleavedRegionSearchOptions::bilinear_alternation_options,
          doc.CspaceFreeRegion.InterleavedRegionSearchOptions.bilinear_alternation_options.doc)
      .def_readwrite("max_method_switch",
          &CspaceFreeRegion::InterleavedRegionSearchOptions::max_method_switch,
          doc.CspaceFreeRegion.InterleavedRegionSearchOptions.max_method_switch.doc)
      .def_readwrite("use_vector_bisection_search",
          &CspaceFreeRegion::InterleavedRegionSearchOptions::use_vector_bisection_search,
          doc.CspaceFreeRegion.InterleavedRegionSearchOptions.use_vector_bisection_search.doc);

  // CspaceFreeRegion
  py::class_<CspaceFreeRegion> cspace_cls(
      m, "CspaceFreeRegion", doc.CspaceFreeRegion.doc);

  cspace_cls.def(
      py::init<const systems::Diagram<double>&, const MultibodyPlant<double>*,
          const geometry::SceneGraph<double>*, SeparatingPlaneOrder,
          CspaceRegionType>(),
      doc.CspaceFreeRegion.ctor.doc);

  cspace_cls
      .def("map_polytopes_to_separating_planes",
          &CspaceFreeRegion::map_polytopes_to_separating_planes,
          doc.CspaceFreeRegion.map_polytopes_to_separating_planes.doc)
      .def("GenerateLinkOnOneSideOfPlaneRationals",
          &CspaceFreeRegion::GenerateLinkOnOneSideOfPlaneRationals,
          py::arg("q_star"), py::arg("filtered_collision_pairs"),
          doc.CspaceFreeRegion.GenerateLinkOnOneSideOfPlaneRationals.doc)
      .def_property_readonly("rational_forward_kinematics",
          &CspaceFreeRegion::rational_forward_kinematics,
          doc.CspaceFreeRegion.rational_forward_kinematics.doc)
      .def_property_readonly("plane_order", &CspaceFreeRegion::plane_order,
          doc.CspaceFreeRegion.plane_order.doc)
      .def_property_readonly("cspace_region_type",
          &CspaceFreeRegion::cspace_region_type,
          doc.CspaceFreeRegion.cspace_region_type.doc);
  // CspacePolytopeTuple
  py::class_<CspaceFreeRegion::CspacePolytopeTuple>(cspace_cls,
      "CspacePolytopeTuple", doc.CspaceFreeRegion.CspacePolytopeTuple.doc)
      .def_readonly("rational_numerator",
          &CspaceFreeRegion::CspacePolytopeTuple::rational_numerator,
          doc.CspaceFreeRegion.CspacePolytopeTuple.rational_numerator.doc)
      .def_readonly("polytope_lagrangian_gram_lower_start",
          &CspaceFreeRegion::CspacePolytopeTuple::
              polytope_lagrangian_gram_lower_start,
          doc.CspaceFreeRegion.CspacePolytopeTuple
              .polytope_lagrangian_gram_lower_start.doc)
      .def_readonly("t_lower_lagrangian_gram_lower_start",
          &CspaceFreeRegion::CspacePolytopeTuple::
              t_lower_lagrangian_gram_lower_start,
          doc.CspaceFreeRegion.CspacePolytopeTuple
              .t_lower_lagrangian_gram_lower_start.doc)
      .def_readonly("t_upper_lagrangian_gram_lower_start",
          &CspaceFreeRegion::CspacePolytopeTuple::
              t_upper_lagrangian_gram_lower_start,
          doc.CspaceFreeRegion.CspacePolytopeTuple
              .t_upper_lagrangian_gram_lower_start.doc)
      .def_readonly("verified_polynomial_gram_lower_start",
          &CspaceFreeRegion::CspacePolytopeTuple::
              verified_polynomial_gram_lower_start,
          doc.CspaceFreeRegion.CspacePolytopeTuple
              .verified_polynomial_gram_lower_start.doc)
      .def_readonly("monomial_basis",
          &CspaceFreeRegion::CspacePolytopeTuple::monomial_basis,
          doc.CspaceFreeRegion.CspacePolytopeTuple.monomial_basis.doc);

  cspace_cls
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
            self->GenerateTuplesForBilinearAlternation(q_star,
                filtered_collision_pairs, C_rows, &alternation_tuples,
                &d_minus_Ct, &t_lower, &t_upper, &t_minus_t_lower,
                &t_upper_minus_t, &C, &d, &lagrangian_gram_vars,
                &verified_gram_vars, &separating_plane_vars,
                &separating_plane_to_tuples);
            return std::make_tuple(alternation_tuples, d_minus_Ct, t_lower,
                t_upper, t_minus_t_lower, t_upper_minus_t, C, d,
                lagrangian_gram_vars, verified_gram_vars, separating_plane_vars,
                separating_plane_to_tuples);
          },
          py::arg("q_star"), py::arg("filtered_collision_pairs"),
          py::arg("C_rows"),
          doc.CspaceFreeRegion.GenerateTuplesForBilinearAlternation.doc)
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
              const Eigen::Ref<const Eigen::VectorXd>& t_lower,
              const Eigen::Ref<const Eigen::VectorXd>& t_upper,
              const VerificationOption& option,
              std::optional<double> redundant_tighten) {
            auto prog = self->ConstructLagrangianProgram(alternation_tuples, C,
                d, lagrangian_gram_vars, verified_gram_vars,
                separating_plane_vars, t_lower, t_upper, option,
                redundant_tighten, nullptr, nullptr);
            return prog;
          },
          py::arg("alternation_tuples"), py::arg("C"), py::arg("d"),
          py::arg("lagrangian_gram_vars"), py::arg("verified_gram_vars"),
          py::arg("separating_plane_vars"), py::arg("t_lower"),
          py::arg("t_upper"), py::arg("option"), py::arg("redundant_tighten"),
          doc.CspaceFreeRegion.ConstructLagrangianProgram.doc)
      .def("ConstructPolytopeProgram",
          &CspaceFreeRegion::ConstructPolytopeProgram,
          py::arg("alternation_tuples"), py::arg("C"), py::arg("d"),
          py::arg("d_minus_Ct"), py::arg("lagrangian_gram_var_vals"),
          py::arg("verified_gram_vars"), py::arg("separating_plane_vars"),
          py::arg("t_minus_t_lower"), py::arg("t_upper_minus_t"),
          py::arg("option"), doc.CspaceFreeRegion.ConstructPolytopeProgram.doc)
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
              const std::optional<Eigen::MatrixXd>& q_inner_pts,
              const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
                  inner_polytope) {
            CspaceFreeRegionSolution cspace_free_region_solution;
            self->CspacePolytopeBilinearAlternation(q_star,
                filtered_collision_pairs, C_init, d_init,
                bilinear_alternation_option, solver_options, q_inner_pts,
                inner_polytope, &cspace_free_region_solution);
            return cspace_free_region_solution;
          },
          py::arg("q_star"), py::arg("filtered_collision_pairs"),
          py::arg("C_init"), py::arg("d_init"),
          py::arg("bilinear_alternation_option"), py::arg("solver_options"),
          py::arg("q_inner_pts") = std::nullopt,
          py::arg("inner_polytope") = std::nullopt,
          doc.CspaceFreeRegion.CspacePolytopeBilinearAlternation.doc)
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
              const std::optional<Eigen::MatrixXd>& q_inner_pts,
              const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
                  inner_polytope) {
            CspaceFreeRegionSolution cspace_free_region_solution;
            self->CspacePolytopeBinarySearch(q_star, filtered_collision_pairs,
                C, d_init, binary_search_option, solver_options, q_inner_pts,
                inner_polytope, &cspace_free_region_solution);
            return cspace_free_region_solution;
          },
          py::arg("q_star"), py::arg("filtered_collision_pairs"), py::arg("C"),
          py::arg("d_init"), py::arg("binary_search_option"),
          py::arg("solver_options"), py::arg("q_inner_pts") = std::nullopt,
          py::arg("inner_polytope") = std::nullopt,
          doc.CspaceFreeRegion.CspacePolytopeBinarySearch.doc)
      .def(
          "CspacePolytopeBisectionSearchVector",
          [](const CspaceFreeRegion* self,
              const Eigen::Ref<const Eigen::VectorXd>& q_star,
              const CspaceFreeRegion::FilteredCollisionPairs&
                  filtered_collision_pairs,
              const Eigen::Ref<const Eigen::MatrixXd>& C,
              const Eigen::Ref<const Eigen::VectorXd>& d_init,
              const CspaceFreeRegion::VectorBisectionSearchOption&
                  vector_bisection_search_option,
              const solvers::SolverOptions& solver_options,
              const std::optional<Eigen::MatrixXd>& q_inner_pts,
              const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
                  inner_polytope) {
            CspaceFreeRegionSolution cspace_free_region_solution;
            self->CspacePolytopeBisectionSearchVector(q_star,
                filtered_collision_pairs, C, d_init,
                vector_bisection_search_option, solver_options, q_inner_pts,
                inner_polytope, &cspace_free_region_solution);
            return cspace_free_region_solution;
          },
          py::arg("q_star"), py::arg("filtered_collision_pairs"), py::arg("C"),
          py::arg("d_init"), py::arg("vector_bisection_search_option"),
          py::arg("solver_options"), py::arg("q_inner_pts") = std::nullopt,
          py::arg("inner_polytope") = std::nullopt,
          doc.CspaceFreeRegion.CspacePolytopeBisectionSearchVector.doc)
      .def(
          "InterleavedCSpacePolytopeSearch",
          [](const CspaceFreeRegion* self,
              const Eigen::Ref<const Eigen::VectorXd>& q_star,
              const CspaceFreeRegion::FilteredCollisionPairs&
                  filtered_collision_pairs,
              const Eigen::Ref<const Eigen::MatrixXd>& C_init,
              const Eigen::Ref<const Eigen::VectorXd>& d_init,
              const int num_round_robin_rounds,
              const CspaceFreeRegion::InterleavedRegionSearchOptions&
                  interleaved_region_search_options,
              const solvers::SolverOptions& solver_options,
              const std::optional<Eigen::MatrixXd>& q_inner_pts,
              const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
                  inner_polytope) {
            CspaceFreeRegionSolution cspace_free_region_solution;
            self->InterleavedCSpacePolytopeSearch(q_star,
                filtered_collision_pairs, C_init, d_init, num_round_robin_rounds,
                interleaved_region_search_options, solver_options, q_inner_pts,
                inner_polytope, &cspace_free_region_solution);
            return cspace_free_region_solution;
          },
          py::arg("q_star"), py::arg("filtered_collision_pairs"),
          py::arg("C_init"), py::arg("d_init"), py::arg("num_round_robin_rounds"),
          py::arg("interleaved_region_search_options"), py::arg("solver_options"),
          py::arg("q_inner_pts") = std::nullopt,
          py::arg("inner_polytope") = std::nullopt,
          doc.CspaceFreeRegion.CspacePolytopeBilinearAlternation.doc)
      .def(
          "CspacePolytopeRoundRobinBisectionSearch",
          [](const CspaceFreeRegion* self,
              const Eigen::Ref<const Eigen::VectorXd>& q_star,
              const CspaceFreeRegion::FilteredCollisionPairs&
                  filtered_collision_pairs,
              const Eigen::Ref<const Eigen::MatrixXd>& C,
              const Eigen::Ref<const Eigen::VectorXd>& d_init,
              const int num_rounds,
              const CspaceFreeRegion::VectorBisectionSearchOption&
                  vector_bisection_search_option,
              const solvers::SolverOptions& solver_options,
              const std::optional<Eigen::MatrixXd>& q_inner_pts,
              const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
                  inner_polytope) {
            CspaceFreeRegionSolution cspace_free_region_solution;
            self->CspacePolytopeRoundRobinBisectionSearch(q_star,
                filtered_collision_pairs, C, d_init, num_rounds,
                vector_bisection_search_option, solver_options, q_inner_pts,
                inner_polytope, &cspace_free_region_solution);
            return cspace_free_region_solution;
          },
          py::arg("q_star"), py::arg("filtered_collision_pairs"), py::arg("C"),
          py::arg("d_init"), py::arg("num_rounds"), py::arg("vector_bisection_search_option"),
          py::arg("solver_options"), py::arg("q_inner_pts") = std::nullopt,
          py::arg("inner_polytope") = std::nullopt,
          doc.CspaceFreeRegion.CspacePolytopeRoundRobinBisectionSearch.doc)
      .def(
          "CspacePolytopeRoundRobinBisectionSearchForSeedPoints",
          [](const CspaceFreeRegion* self,
              const Eigen::Ref<const Eigen::VectorXd>& q_star,
            const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs,
            const std::vector<Eigen::Ref<Eigen::MatrixXd>>& C_mat_vect,
            const std::vector<Eigen::Ref<Eigen::VectorXd>>& d_init_vect,
            const int num_rounds,
            const std::vector<CspaceFreeRegion::VectorBisectionSearchOption>& vector_bisection_search_option_vect,
            const solvers::SolverOptions& solver_options,
            const std::vector<Eigen::MatrixXd>& seed_points,
            const std::optional<std::vector<std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>>>&
          inner_polytope_vect) {
            std::vector<CspaceFreeRegionSolution> cspace_free_region_solution_vect(C_mat_vect.size());
            std::vector<CspaceFreeRegionSolution*> cspace_free_region_solution_ptrs;
            for (long unsigned int i = 0; i < C_mat_vect.size(); i++){
              cspace_free_region_solution_ptrs.push_back(&(cspace_free_region_solution_vect.at(i)));
            }

            self->CspacePolytopeRoundRobinBisectionSearchForSeedPoints(q_star,
                filtered_collision_pairs, C_mat_vect, d_init_vect, num_rounds,
                vector_bisection_search_option_vect, solver_options, seed_points,
                inner_polytope_vect, cspace_free_region_solution_ptrs);
            std::cout << "returning solution" << std::endl;
            return cspace_free_region_solution_vect;
          },
          py::arg("q_star"), py::arg("filtered_collision_pairs"), py::arg("C_mat_vect"),
          py::arg("d_init_vect"), py::arg("num_rounds"), py::arg("vector_bisection_search_option_vect"),
          py::arg("solver_options"), py::arg("seed_points"),
          py::arg("inner_polytope") = std::nullopt,
          doc.CspaceFreeRegion.CspacePolytopeRoundRobinBisectionSearchForSeedPoints.doc)
      .def(
          "InterleavedCSpacePolytopeSearchForSeedPoints",
          [](const CspaceFreeRegion* self,
              const Eigen::Ref<const Eigen::VectorXd>& q_star,
            const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs,
            const std::vector<Eigen::Ref<Eigen::MatrixXd>>& C_mat_vect,
            const std::vector<Eigen::Ref<Eigen::VectorXd>>& d_init_vect,
            const int num_round_robin_rounds,
            const std::vector<CspaceFreeRegion::InterleavedRegionSearchOptions>& interleaved_region_search_option_vect,
            const solvers::SolverOptions& solver_options,
            const std::vector<Eigen::MatrixXd>& seed_points,
            const std::optional<std::vector<std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>>>&
          inner_polytope_vect) {
            std::vector<CspaceFreeRegionSolution> cspace_free_region_solution_vect(C_mat_vect.size());
            std::vector<CspaceFreeRegionSolution*> cspace_free_region_solution_ptrs;
            for (long unsigned int i = 0; i < C_mat_vect.size(); i++){
              cspace_free_region_solution_ptrs.push_back(&(cspace_free_region_solution_vect.at(i)));
            }

            self->InterleavedCSpacePolytopeSearchForSeedPoints(q_star,
                filtered_collision_pairs, C_mat_vect, d_init_vect, num_round_robin_rounds,
                interleaved_region_search_option_vect, solver_options, seed_points,
                inner_polytope_vect, cspace_free_region_solution_ptrs);
            std::cout << "returning solution" << std::endl;
            return cspace_free_region_solution_vect;
          },
          py::arg("q_star"), py::arg("filtered_collision_pairs"), py::arg("C_mat_vect"),
          py::arg("d_init_vect"), py::arg("num_round_robin_rounds"), py::arg("interleaved_region_search_option_vect"),
          py::arg("solver_options"), py::arg("seed_points"),
          py::arg("inner_polytope") = std::nullopt,
          doc.CspaceFreeRegion.InterleavedCSpacePolytopeSearchForSeedPoints.doc)

      .def("IsPostureInCollision", &CspaceFreeRegion::IsPostureInCollision,
          doc.CspaceFreeRegion.IsPostureInCollision.doc)
      .def("separating_planes", &CspaceFreeRegion::separating_planes,
          py_rvp::reference, doc.CspaceFreeRegion.separating_planes.doc);

  m.def("GetConvexPolytopes", &GetConvexPolytopes, py::arg("diagram"),
      py::arg("plant"), py::arg("scene_graph"), doc.GetConvexPolytopes.doc);

  m.def("AddInscribedEllipsoid", &AddInscribedEllipsoid, py::arg("prog"),
       py::arg("C"), py::arg("d"), py::arg("t_lower"), py::arg("t_upper"),
       py::arg("P"), py::arg("q"), py::arg("constrain_P_psd") = true,
       doc.AddInscribedEllipsoid.doc)
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
          py::arg("t_upper"), py::arg("constrain_P_psd") = true,
          doc.AddInscribedEllipsoid.doc);
  m.def(
      "FindRedundantInequalities",
      [](const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
          const Eigen::VectorXd& t_lower, const Eigen::VectorXd& t_upper,
          double tighten) {
        std::unordered_set<int> C_redundant_indices, t_lower_redundant_indices,
            t_upper_redundant_indices;
        FindRedundantInequalities(C, d, t_lower, t_upper, tighten,
            &C_redundant_indices, &t_lower_redundant_indices,
            &t_upper_redundant_indices);
        std::make_tuple(C_redundant_indices, t_lower_redundant_indices,
            t_upper_redundant_indices);
      },
      py::arg("C"), py::arg("d"), py::arg("t_lower"), py::arg("t_upper"),
      py::arg("tighten"), doc.FindRedundantInequalities.doc);

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
       py::arg("d_inner"), py::arg("t_lower"), py::arg("t_upper"),
       doc.AddCspacePolytopeContainment.doc_7args)
      .def(
          "AddCspacePolytopeContainment",
          [](solvers::MathematicalProgram* prog,
              const MatrixX<symbolic::Variable>& C,
              const VectorX<symbolic::Variable>& d,
              const Eigen::Ref<const Eigen::MatrixXd>& inner_pts) {
            return AddCspacePolytopeContainment(prog, C, d, inner_pts);
          },
          py::arg("prog"), py::arg("C"), py::arg("d"), py::arg("inner_pts"),
          doc.AddCspacePolytopeContainment.doc_4args)
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
          py::arg("tighten"), doc.FindRedundantInequalities.doc)
      .def("FindEpsilonLower", &FindEpsilonLower, py::arg("C"), py::arg("d"),
          py::arg("t_lower"), py::arg("t_upper"),
          py::arg("t_inner_pts") = std::nullopt,
          py::arg("inner_polytope") = std::nullopt, doc.FindEpsilonLower.doc)
      .def("FindEpsilonLowerVector", &FindEpsilonLowerVector, py::arg("C"), py::arg("d"),
          py::arg("t_lower"), py::arg("t_upper"),
          py::arg("t_inner_pts"),
          py::arg("inner_polytope") = std::nullopt, doc.FindEpsilonLowerVector.doc)
      .def("FindEpsilonUpperVector", &FindEpsilonUpperVector,
           py::arg("C"), py::arg("d"),
          py::arg("t_lower"), py::arg("t_upper"));

  m.def("CalcCspacePolytopeVolume", &CalcCspacePolytopeVolume, py::arg("C"),
      py::arg("d"), py::arg("t_lower"), py::arg("t_upper"),
      doc.CalcCspacePolytopeVolume.doc);

  py::module::import("pydrake.solvers.mathematicalprogram");

  m.def("GenerateSeedingPolytope", &GenerateSeedingPolytope,
      py::arg("seed_point"), py::arg("num_perm_dim"), py::arg("num_rot"));

  m.def("MakeKCanonicalSOnMembers", &MakeKCanonicalSOnMembers,
      py::arg("k"), py::arg("n"));

  m.def("MakeKFirstDimSwapsOfDimN", &MakeKFirstDimSwapsOfDimN,
      py::arg("k"), py::arg("n"));

  m.def("SameDimensionalAffineTransform", &SameDimensionalAffineTransform,
      py::arg("C"), py::arg("d"), py::arg("P"));

  m.def("FindMaxEpsForAllIneqs", &FindMaxEpsForAllIneqs,
        py::arg("plant"), py::arg("context"),
        py::arg("q_star"),py::arg("C"),
        py::arg("d"), py::arg("eps_min"), py::arg("t_lower_limits"),
        py::arg("t_upper_limits"), py::arg("t_guess"));
}

}  // namespace pydrake
}  // namespace drake
