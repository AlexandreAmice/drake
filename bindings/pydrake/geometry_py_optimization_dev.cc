#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/sorted_pair_pybind.h"
#include "drake/bindings/pydrake/common/value_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/geometry/optimization/dev/collision_geometry.h"
#include "drake/geometry/optimization/dev/cspace_free_polytope.h"
#include "drake/geometry/optimization/dev/separating_plane.h"


namespace drake {
namespace pydrake {

// SeparatingPlane
template <typename T>
void DoSeparatingPlaneDeclaration(py::module m, T) {
  constexpr auto& doc = pydrake_doc.drake.geometry.optimization;
  py::tuple param = GetPyParam<T>();
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  {
    using Class = geometry::optimization::SeparatingPlane<T>;
    constexpr auto& cls_doc = doc.SeparatingPlane;
    auto cls = DefineTemplateClassWithDefault<Class>(
        m, "SeparatingPlane", param, cls_doc.doc)
                   .def_readonly("a", &Class::a, py_rvp::copy, cls_doc.a.doc)
                   .def_readonly("b", &Class::b, cls_doc.b.doc)
                   .def_readonly("positive_side_geometry",
                       &Class::positive_side_geometry,
                       cls_doc.positive_side_geometry.doc)
                   .def_readonly("negative_side_geometry",
                       &Class::negative_side_geometry,
                       cls_doc.negative_side_geometry.doc)
                   .def_readonly("expressed_body", &Class::expressed_body,
                       cls_doc.expressed_body.doc)
                   .def_readonly("plane_order", &Class::plane_order,
                       cls_doc.plane_order.doc)
                   .def_readonly("decision_variables",
                       &Class::decision_variables, py_rvp::copy, cls_doc.a.doc);
    DefCopyAndDeepCopy(&cls);
    AddValueInstantiation<Class>(m);
  }
}
void DefineGeometryOptimizationDev(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::geometry;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::geometry::optimization;

  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  m.doc() = "optimization dev module";
  constexpr auto& doc = pydrake_doc.drake.geometry.optimization;

  {
    // Definitions for collision_geometry.h/cc
    py::enum_<PlaneSide>(m, "PlaneSide", doc.PlaneSide.doc)
        .value("kPositive", PlaneSide::kPositive)
        .value("kNegative", PlaneSide::kNegative);

    py::enum_<GeometryType>(m, "GeometryType", doc.GeometryType.doc)
        .value("kPolytope", GeometryType::kPolytope,
            doc.GeometryType.kPolytope.doc)
        .value("kSphere", GeometryType::kSphere, doc.GeometryType.kSphere.doc)
        .value("kCylinder", GeometryType::kCylinder,
            doc.GeometryType.kCylinder.doc)
        .value(
            "kCapsule", GeometryType::kCapsule, doc.GeometryType.kCapsule.doc);

    py::class_<CollisionGeometry>(
        m, "CollisionGeometry", doc.CollisionGeometry.doc)
        .def("type", &CollisionGeometry::type, doc.CollisionGeometry.type.doc)
        .def("geometry", &CollisionGeometry::geometry,
            py_rvp::reference_internal, doc.CollisionGeometry.geometry.doc)
        .def("body_index", &CollisionGeometry::body_index,
            doc.CollisionGeometry.body_index.doc)
        .def("id", &CollisionGeometry::id, doc.CollisionGeometry.id.doc)
        .def("X_BG", &CollisionGeometry::X_BG, doc.CollisionGeometry.X_BG.doc)
        .def("num_rationals", &CollisionGeometry::num_rationals,
            doc.CollisionGeometry.num_rationals.doc);
  }
  {
    // Definitions for separating_plane.h/cc
    py::enum_<SeparatingPlaneOrder>(
        m, "SeparatingPlaneOrder", doc.SeparatingPlaneOrder.doc)
        .value("kAffine", SeparatingPlaneOrder::kAffine,
            doc.SeparatingPlaneOrder.kAffine.doc);
    type_visit([m](auto dummy) { DoSeparatingPlaneDeclaration(m, dummy); },
      type_pack<double, symbolic::Variable>());
  }
  {
    using Class = CspaceFreePolytope;
    const auto& cls_doc = doc.CspaceFreePolytope;
    py::class_<Class>(m, "CspaceFreePolytope", cls_doc.doc)
        .def(py::init<const multibody::MultibodyPlant<double>*,
                 const geometry::SceneGraph<double>*, SeparatingPlaneOrder,
                 const Eigen::Ref<const Eigen::VectorXd>&>(),
            py::arg("plant"), py::arg("scene_graph"), py::arg("plane_order"),
            py::arg("q_star"),
            // Keep alive, reference: `self` keeps `scene_graph` alive.
            py::keep_alive<1, 3>(), cls_doc.ctor.doc)
//        .def("rational_forward_kin", &Class::rational_forward_kin, py_rvp::reference_internal,
//            cls_doc.rational_forward_kin.doc)
//        .def("map_geometries_to_separating_planes",
//            &Class::map_geometries_to_separating_planes,
//            cls_doc.map_geometries_to_separating_planes.doc)
        .def("separating_planes", &Class::separating_planes,
            cls_doc.separating_planes.doc)
        .def("y_slack", &Class::y_slack, cls_doc.y_slack.doc)
        .def(
            "FindSeparationCertificateGivenPolytope",
            [](const CspaceFreePolytope* self,
                const Eigen::Ref<const Eigen::MatrixXd>& C,
                const Eigen::Ref<const Eigen::VectorXd>& d,
                const CspaceFreePolytope::IgnoredCollisionPairs&
                    ignored_collision_pairs,
                const CspaceFreePolytope::
                    FindSeparationCertificateGivenPolytopeOptions& options) {
              std::unordered_map<SortedPair<geometry::GeometryId>,
                  CspaceFreePolytope::SeparationCertificateResult>
                  certificates;
              bool success = self->FindSeparationCertificateGivenPolytope(
                  C, d, ignored_collision_pairs, options, &certificates);
              // the type std::unordered_map<SortedPair<geometry::GeometryId>,
              // CspaceFreePolytope::SeparationCertificateResult> does not map
              // to a Python type. Instead, we return a list of tuples
              // containing the geometry ids and the certificate for that pair.
              std::vector<std::tuple<geometry::GeometryId, geometry::GeometryId,
                  CspaceFreePolytope::SeparationCertificateResult>>
                  certificates_ret;
              certificates_ret.reserve(certificates.size());
              for (const auto& [key, value] : certificates) {
                certificates_ret.emplace_back(key.first(), key.second(), value);
              }
              return std::pair(success, certificates_ret);
//                return std::pair(success, certificates);
            },
            py::arg("C"), py::arg("d"), py::arg("ignored_collision_pairs"),
            py::arg("options"))
        .def("SearchWithBilinearAlternation",
            &Class::SearchWithBilinearAlternation,
            py::arg("ignored_collision_pairs"), py::arg("C_init"),
            py::arg("d_init"), py::arg("options"),
            cls_doc.SearchWithBilinearAlternation.doc)
        .def("BinarySearch", &Class::BinarySearch,
            py::arg("ignored_collision_pairs"), py::arg("C"), py::arg("d"),
            py::arg("s_center"), py::arg("options"), cls_doc.BinarySearch.doc);

    py::class_<Class::SeparationCertificateResult>(m,
        "SeparationCertificateResult", cls_doc.SeparationCertificateResult.doc)
        .def_readonly(
            "plane_index", &Class::SeparationCertificateResult::plane_index)
        .def_readonly("positive_side_rational_lagrangians",
            &Class::SeparationCertificateResult::
                positive_side_rational_lagrangians,
            cls_doc.SeparationCertificateResult
                .positive_side_rational_lagrangians.doc)
        .def_readonly("negative_side_rational_lagrangians",
            &Class::SeparationCertificateResult::
                negative_side_rational_lagrangians,
            cls_doc.SeparationCertificateResult
                .negative_side_rational_lagrangians.doc)
        .def_readonly("a", &Class::SeparationCertificateResult::a, py_rvp::copy,
            cls_doc.SeparationCertificateResult.a.doc)
        .def_readonly("b", &Class::SeparationCertificateResult::b,
            cls_doc.SeparationCertificateResult.b.doc)
        .def_readonly("plane_decision_var_vals",
            &Class::SeparationCertificateResult::plane_decision_var_vals,
            py_rvp::copy,
            cls_doc.SeparationCertificateResult.plane_decision_var_vals.doc);

    py::class_<Class::SeparatingPlaneLagrangians>(
        m, "SeparatingPlaneLagrangians", cls_doc.SeparatingPlaneLagrangians.doc)
        .def_readonly("polytope", &Class::SeparatingPlaneLagrangians::polytope,
            py_rvp::copy)
        .def_readonly("s_lower", &Class::SeparatingPlaneLagrangians::s_lower,
            py_rvp::copy)
        .def_readonly("s_upper", &Class::SeparatingPlaneLagrangians::s_upper,
            py_rvp::copy);

    py::class_<Class::FindSeparationCertificateGivenPolytopeOptions>(m,
        "FindSeparationCertificateGivenPolytopeOptions",
        cls_doc.FindSeparationCertificateGivenPolytopeOptions.doc)
        .def(py::init<>())
        .def_readwrite("num_threads",
            &Class::FindSeparationCertificateGivenPolytopeOptions::num_threads)
        .def_readwrite("verbose",
            &Class::FindSeparationCertificateGivenPolytopeOptions::verbose)
        .def_readwrite("solver_id",
            &Class::FindSeparationCertificateGivenPolytopeOptions::solver_id)
        .def_readwrite("terminate_at_failure",
            &Class::FindSeparationCertificateGivenPolytopeOptions::
                terminate_at_failure)
        .def_readwrite("solver_options",
            &Class::FindSeparationCertificateGivenPolytopeOptions::
                solver_options)
        .def_readwrite("ignore_redundant_C",
            &Class::FindSeparationCertificateGivenPolytopeOptions::
                ignore_redundant_C);

    py::class_<Class::FindPolytopeGivenLagrangianOptions>(m,
        "FindPolytopeGivenLagrangianOptions",
        cls_doc.FindPolytopeGivenLagrangianOptions.doc)
        .def(py::init<>())
        .def_readwrite("backoff_scale",
            &Class::FindPolytopeGivenLagrangianOptions::backoff_scale)
        .def_readwrite("ellipsoid_margin_epsilon",
            &Class::FindPolytopeGivenLagrangianOptions::
                ellipsoid_margin_epsilon)
        .def_readwrite(
            "solver_id", &Class::FindPolytopeGivenLagrangianOptions::solver_id)
        .def_readwrite("solver_options",
            &Class::FindPolytopeGivenLagrangianOptions::solver_options)
        .def_readwrite("s_inner_pts",
            &Class::FindPolytopeGivenLagrangianOptions::s_inner_pts)
        .def_readwrite("search_s_bounds_lagrangians",
            &Class::FindPolytopeGivenLagrangianOptions::
                search_s_bounds_lagrangians)
        .def_readwrite("ellipsoid_margin_cost",
            &Class::FindPolytopeGivenLagrangianOptions::ellipsoid_margin_cost);

    py::class_<Class::Options>(m, "Options", cls_doc.Options.doc)
        .def(py::init<>())
        .def_readwrite("with_cross_y", &Class::Options::with_cross_y);

        py::enum_<Class::EllipsoidMarginCost>(
            m, "EllipsoidMarginCost", cls_doc.EllipsoidMarginCost.doc)
            .value("kSum", Class::EllipsoidMarginCost::kSum)
            .value("kGeometricMean",
            Class::EllipsoidMarginCost::kGeometricMean);

    py::class_<Class::SearchResult>(m, "SearchResult", cls_doc.SearchResult.doc)
        .def_readonly("C", &Class::SearchResult::C)
        .def_readonly("d", &Class::SearchResult::d)
        .def_readonly("a", &Class::SearchResult::a, py_rvp::copy)
        .def_readonly("b", &Class::SearchResult::b)
        .def_readonly("num_iter", &Class::SearchResult::num_iter);

    py::class_<Class::BilinearAlternationOptions>(
        m, "BilinearAlternationOptions", cls_doc.BilinearAlternationOptions.doc)
        .def(py::init<>())
        .def_readwrite("max_iter", &Class::BilinearAlternationOptions::max_iter)
        .def_readwrite("convergence_tol",
            &Class::BilinearAlternationOptions::convergence_tol)
        .def_readwrite("find_polytope_options",
            &Class::BilinearAlternationOptions::find_polytope_options)
        .def_readwrite("find_lagrangian_options",
            &Class::BilinearAlternationOptions::find_lagrangian_options)
        .def_readwrite("ellipsoid_scaling",
            &Class::BilinearAlternationOptions::ellipsoid_scaling);

    py::class_<Class::BinarySearchOptions>(
        m, "BinarySearchOptions", cls_doc.BinarySearchOptions.doc)
        .def(py::init<>())
        .def_readwrite("scale_max", &Class::BinarySearchOptions::scale_max)
        .def_readwrite("scale_min", &Class::BinarySearchOptions::scale_min)
        .def_readwrite("max_iter", &Class::BinarySearchOptions::max_iter)
        .def_readwrite(
            "convergence_tol", &Class::BinarySearchOptions::convergence_tol)
        .def_readwrite("find_lagrangian_options",
            &Class::BinarySearchOptions::find_lagrangian_options);
  }

  m.def("GetCollisionGeometries",
      py::overload_cast<const multibody::MultibodyPlant<double>&,
          const geometry::SceneGraph<double>&>(&GetCollisionGeometries),
      py::arg("plant"), py::arg("scene_graph"), doc.GetCollisionGeometries.doc);

}
}  // namespace pydrake
}  // namespace drake