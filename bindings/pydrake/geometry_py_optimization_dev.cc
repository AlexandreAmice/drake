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

void DefineCalcPlane(py::module m) {
  constexpr auto& doc = pydrake_doc.drake.geometry.optimization;
  using namespace drake::geometry::optimization;
  m.def(
      "CalcPlane",
      [](const VectorX<symbolic::Variable>& decision_variables,
          const VectorX<symbolic::Variable>& s_for_plane,
          SeparatingPlaneOrder order) {
        Vector3<symbolic::Polynomial> a;
        symbolic::Polynomial b;
        CalcPlane<symbolic::Variable, symbolic::Variable, symbolic::Polynomial>(
            decision_variables, s_for_plane, order, &a, &b);
        return std::make_tuple(a, b);
      },
      py::arg("decision_variables"), py::arg("s_for_plane"), py::arg("order"),
      doc.CalcPlane.doc);
  m.def(
      "CalcPlane",
      [](const VectorX<double>& decision_variables,
          const VectorX<symbolic::Variable>& s_for_plane,
          SeparatingPlaneOrder order) {
        Vector3<symbolic::Polynomial> a;
        symbolic::Polynomial b;
        CalcPlane<double, symbolic::Variable, symbolic::Polynomial>(
            decision_variables, s_for_plane, order, &a, &b);
        return std::make_tuple(a, b);
      },
      py::arg("decision_variables"), py::arg("s_for_plane"), py::arg("order"),
      doc.CalcPlane.doc);
  m.def(
      "CalcPlane",
      [](const VectorX<double>& decision_variables,
          const VectorX<double>& s_for_plane, SeparatingPlaneOrder order) {
        Eigen::Vector3d a;
        double b;
        CalcPlane<double, double, double>(
            decision_variables, s_for_plane, order, &a, &b);
        return std::make_tuple(a, b);
      },
      py::arg("decision_variables"), py::arg("s_for_plane"), py::arg("order"),
      doc.CalcPlane.doc);
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
    DefineCalcPlane(m);
  }
  {
    using Class = CspaceFreePolytope;
    const auto& cls_doc = doc.CspaceFreePolytope;

    py::module::import("pydrake.multibody.rational");

    py::class_<Class> cspace_free_polytope_cls(
        m, "CspaceFreePolytope", cls_doc.doc);

    py::class_<Class::Options>(
        cspace_free_polytope_cls, "Options", cls_doc.Options.doc)
        .def(py::init<>())
        .def_readwrite("with_cross_y", &Class::Options::with_cross_y);

    cspace_free_polytope_cls
        .def(py::init<const multibody::MultibodyPlant<double>*,
                 const geometry::SceneGraph<double>*, SeparatingPlaneOrder,
                 const Eigen::Ref<const Eigen::VectorXd>&,
                 const Class::Options&>(),
            py::arg("plant"), py::arg("scene_graph"), py::arg("plane_order"),
            py::arg("q_star"), py::arg("options") = Class::Options(),
            // Keep alive, reference: `self` keeps `scene_graph` alive.
            py::keep_alive<1, 3>(), cls_doc.ctor.doc)
        .def("rational_forward_kin", &Class::rational_forward_kin,
            py_rvp::reference_internal, cls_doc.rational_forward_kin.doc)
        .def(
            "map_geometries_to_separating_planes",
            [](const CspaceFreePolytope* self) {
              // Template deduction for drake::SortedPair<GeometryId> does not
              // work. Here we manually make a map of tuples instead.
              py::dict ret;
              for (auto [k, v] : self->map_geometries_to_separating_planes()) {
                ret[py::make_tuple(k.first(), k.second())] = v;
              }
              return ret;
            },
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
            py::arg("s_center"), py::arg("options"), cls_doc.BinarySearch.doc)
        .def("ConstructPlaneSearchProgramsForPairs", &Class::ConstructPlaneSearchProgramsForPairs,
             py::arg("C"), py::arg("d"), py::arg("options"));

    py::class_<Class::SeparationCertificateProgram>(m, "SeparationCertificateProgram")
        .def("prog", &Class::SeparationCertificateProgram::get_prog, py_rvp::reference_internal)
        .def_readonly("certificate", &Class::SeparationCertificateProgram::certificate);


    py::class_<Class::SeparationCertificateResult>(cspace_free_polytope_cls,
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

    py::class_<Class::SeparatingPlaneLagrangians>(cspace_free_polytope_cls,
        "SeparatingPlaneLagrangians", cls_doc.SeparatingPlaneLagrangians.doc)
        .def_readonly("polytope", &Class::SeparatingPlaneLagrangians::polytope,
            py_rvp::copy)
        .def_readonly("s_lower", &Class::SeparatingPlaneLagrangians::s_lower,
            py_rvp::copy)
        .def_readonly("s_upper", &Class::SeparatingPlaneLagrangians::s_upper,
            py_rvp::copy);

    py::class_<Class::FindSeparationCertificateGivenPolytopeOptions>(
        cspace_free_polytope_cls,
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

    py::class_<Class::FindPolytopeGivenLagrangianOptions>(
        cspace_free_polytope_cls, "FindPolytopeGivenLagrangianOptions",
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

    py::enum_<Class::EllipsoidMarginCost>(cspace_free_polytope_cls,
        "EllipsoidMarginCost", cls_doc.EllipsoidMarginCost.doc)
        .value("kSum", Class::EllipsoidMarginCost::kSum)
        .value("kGeometricMean", Class::EllipsoidMarginCost::kGeometricMean);

    py::class_<Class::SearchResult>(
        cspace_free_polytope_cls, "SearchResult", cls_doc.SearchResult.doc)
        .def_readonly("C", &Class::SearchResult::C)
        .def_readonly("d", &Class::SearchResult::d)
        .def_readonly("a", &Class::SearchResult::a, py_rvp::copy)
        .def_readonly("b", &Class::SearchResult::b)
        .def_readonly("num_iter", &Class::SearchResult::num_iter)
        .def_readonly("certified_polytope", &Class::SearchResult::certified_polytope);

    py::class_<Class::BilinearAlternationOptions>(cspace_free_polytope_cls,
        "BilinearAlternationOptions", cls_doc.BilinearAlternationOptions.doc)
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

    py::class_<Class::BinarySearchOptions>(cspace_free_polytope_cls,
        "BinarySearchOptions", cls_doc.BinarySearchOptions.doc)
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
