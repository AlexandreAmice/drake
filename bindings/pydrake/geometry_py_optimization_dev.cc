#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
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
PYBIND11_MODULE(optimization_dev, m) {
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
    py::enum_<PlaneSide>(m, "PlaneSide", doc.PlaneSide.doc)
        .value("kPositive", PlaneSide::kPositive)
        .value("kNegative", PlaneSide::kNegative);

    py::enum_<GeometryType>(m, "GeometryType", doc.GeometryType.doc)
        .value("kPolytope", GeometryType::kPolytope,
            doc.GeometryType.kPolytope.doc)
        .value("kSphere", GeometryType::kSphere, doc.GeometryType.kSphere.doc)
        .value(
            "kCapsule", GeometryType::kCapsule, doc.GeometryType.kCapsule.doc);

    py::class_<CollisionGeometry>(
        m, "CollisionGeometry", doc.CollisionGeometry.doc)
        .def("type", &CollisionGeometry::type, doc.CollisionGeometry.doc)
        .def("geometry", &CollisionGeometry::geometry,
            py_rvp::reference_internal, doc.CollisionGeometry.geometry.doc)
        .def("body_index", &CollisionGeometry::body_index,
            doc.CollisionGeometry.body_index.doc)
        .def("id", &CollisionGeometry::id, doc.CollisionGeometry.id.doc)
        .def("X_BG", &CollisionGeometry::X_BG, doc.CollisionGeometry.X_BG.doc)
        .def("num_rationals", &CollisionGeometry::num_rationals_per_side,
            doc.CollisionGeometry.num_rationals_per_side.doc);
  }
  {
    py::enum_<SeparatingPlaneOrder>(
        m, "SeparatingPlaneOrder", doc.SeparatingPlaneOrder.doc)
        .value("kAffine", SeparatingPlaneOrder::kAffine,
            doc.SeparatingPlaneOrder.kAffine.doc);
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
        .def("rational_forward_kin", &Class::rational_forward_kin,
            cls_doc.rational_forward_kin.doc)
        .def("map_geometries_to_separating_planes",
            &Class::map_geometries_to_separating_planes,
            cls_doc.map_geometries_to_separating_planes.doc)
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
                bool search_separating_margin,
                const CspaceFreePolytope::
                    FindSeparationCertificateGivenPolytopeOptions& options) {
              std::unordered_map<SortedPair<geometry::GeometryId>,
                  CspaceFreePolytope::SeparationCertificateResult>
                  certificates;
              bool success = self->FindSeparationCertificateGivenPolytope(C, d,
                  ignored_collision_pairs, search_separating_margin, options,
                  &certificates);
              return std::tie(success, certificates);
            },
            py::arg("C"), py::arg("d"), py::arg("ignored_collision_pairs"),
            py::arg("search_separating_margin"), py::arg("options"));

    py::class_<Class::SeparationCertificateResult>(m,
        "SeparationCertificateResult", cls_doc.SeparationCertificateResult.doc)
        .def_readonly("C", &Class::SeparationCertificateResult::C,
            cls_doc.SeparationCertificateResult.C.doc)
        .def_readonly("d", &Class::SeparationCertificateResult::d,
            cls_doc.SeparationCertificateResult.d.doc)
        .def_readonly("positive_side_lagrangians",
            &Class::SeparationCertificateResult::positive_side_lagrangians,
            cls_doc.SeparationCertificateResult.positive_side_lagrangians.doc)
        .def_readonly("negative_side_lagrangians",
            &Class::SeparationCertificateResult::negative_side_lagrangians,
            cls_doc.SeparationCertificateResult.negative_side_lagrangians.doc)
        .def_readonly("a", &Class::SeparationCertificateResult::a, py_rvp::copy,
            cls_doc.SeparationCertificateResult.a.doc)
        .def_readonly("b", &Class::SeparationCertificateResult::b,
            cls_doc.SeparationCertificateResult.b.doc);

    py::class_<Class::SeparatingPlaneLagrangians>(
        m, "SeparatingPlaneLagrangians", cls_doc.SeparatingPlaneLagrangians.doc)
        .def_readonly("polytope_lagrangians",
            &Class::SeparatingPlaneLagrangians::polytope, py_rvp::copy)
        .def_readonly("s_lower_lagrangians",
            &Class::SeparatingPlaneLagrangians::s_lower, py_rvp::copy)
        .def_readonly("s_upper_lagrangians",
            &Class::SeparatingPlaneLagrangians::s_upper, py_rvp::copy);

    py::class_<Class::UnitLengthLagrangians>(
        m, "UnitLengthLagrangians", cls_doc.UnitLengthLagrangians.doc)
        .def_readonly("polytope_lagrangians",
            &Class::UnitLengthLagrangians::polytope, py_rvp::copy)
        .def_readonly("s_lower_lagrangians",
            &Class::UnitLengthLagrangians::s_lower, py_rvp::copy)
        .def_readonly("s_upper_lagrangians",
            &Class::UnitLengthLagrangians::s_upper, py_rvp::copy)
        .def_readonly("y_lagrangians", &Class::UnitLengthLagrangians::y_square,
            py_rvp::copy);

    py::class_<Class::FindSeparationCertificateGivenPolytopeOptions>(m,
        "FindSeparationCertificateGivenPolytopeOptions",
        cls_doc.FindSeparationCertificateGivenPolytopeOptions.doc)
        .def_readwrite("num_threads",
            &Class::FindSeparationCertificateGivenPolytopeOptions::num_threads)
        .def_readwrite("verbose",
            &Class::FindSeparationCertificateGivenPolytopeOptions::verbose)
        .def_readwrite("solver_id",
            &Class::FindSeparationCertificateGivenPolytopeOptions::solver_id)
        .def_readwrite("terminate_at_failure",
            &Class::FindSeparationCertificateGivenPolytopeOptions::
                terminate_at_failure)
        .def_readwrite("backoff_scale",
            &Class::FindSeparationCertificateGivenPolytopeOptions::
                backoff_scale)
        .def_readwrite("solver_options",
            &Class::FindSeparationCertificateGivenPolytopeOptions::
                solver_options);
  }
  type_visit([m](auto dummy) { DoSeparatingPlaneDeclaration(m, dummy); },
      type_pack<double, symbolic::Variable>());
}
}  // namespace pydrake
}  // namespace drake