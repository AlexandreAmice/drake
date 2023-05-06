#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/sorted_pair_pybind.h"
#include "drake/bindings/pydrake/common/value_pybind.h"
#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/geometry/optimization/c_iris_collision_geometry.h"
#include "drake/geometry/optimization/c_iris_separating_plane.h"
#include "drake/geometry/optimization/cspace_free_polytope.h"
#include "drake/geometry/optimization/cspace_free_structs.h"
#include "drake/geometry/optimization/cspace_separating_plane.h"
#include "drake/geometry/optimization/dev/cspace_free_path_separating_plane.h"

namespace drake {
namespace pydrake {

// CSpaceSeparatingPlane, CIrisSeparatingPlane, CSpacePathSeparatingPlane
template <typename T>
void DoSeparatingPlaneDeclaration(py::module m, T) {
  constexpr auto& doc = pydrake_doc.drake.geometry.optimization;
  py::tuple param = GetPyParam<T>();
  using BaseClass = geometry::optimization::CSpaceSeparatingPlane<T>;
  constexpr auto& base_cls_doc = doc.CSpaceSeparatingPlane;
  using CIrisClass = geometry::optimization::CIrisSeparatingPlane<T>;
  constexpr auto& ciris_cls_doc = doc.CSpaceSeparatingPlane;
  using CPathClass = geometry::optimization::CSpacePathSeparatingPlane<T>;
  constexpr auto& cpath_cls_doc = doc.CSpacePathSeparatingPlane;
  {
    auto cls =
        DefineTemplateClassWithDefault<BaseClass>(
            m, "CSpaceSeparatingPlane", param, base_cls_doc.doc)
            .def_readonly("a", &BaseClass::a, py_rvp::copy, base_cls_doc.a.doc)
            .def_readonly("b", &BaseClass::b, base_cls_doc.b.doc)
            .def_readonly("positive_side_geometry",
                &BaseClass::positive_side_geometry,
                base_cls_doc.positive_side_geometry.doc)
            .def_readonly("negative_side_geometry",
                &BaseClass::negative_side_geometry,
                base_cls_doc.negative_side_geometry.doc)
            .def_readonly("expressed_body", &BaseClass::expressed_body,
                base_cls_doc.expressed_body.doc)
            .def_readonly("decision_variables", &BaseClass::decision_variables,
                py_rvp::copy, base_cls_doc.a.doc);
    DefCopyAndDeepCopy(&cls);
    AddValueInstantiation<BaseClass>(m);
  }
  {
    auto cls =
        DefineTemplateClassWithDefault<CIrisClass, BaseClass>(
            m, "CIrisSeparatingPlane", param, ciris_cls_doc.doc)
            .def_readonly("a", &BaseClass::a, py_rvp::copy, base_cls_doc.a.doc)
            .def_readonly("b", &BaseClass::b, base_cls_doc.b.doc)
            .def_readonly("positive_side_geometry",
                &BaseClass::positive_side_geometry,
                base_cls_doc.positive_side_geometry.doc)
            .def_readonly("negative_side_geometry",
                &BaseClass::negative_side_geometry,
                base_cls_doc.negative_side_geometry.doc)
            .def_readonly("expressed_body", &BaseClass::expressed_body,
                base_cls_doc.expressed_body.doc)
            .def_readonly("plane_order", &CIrisClass::plane_order)
            .def_readonly("decision_variables", &BaseClass::decision_variables,
                py_rvp::copy, base_cls_doc.a.doc);
    DefCopyAndDeepCopy(&cls);
    AddValueInstantiation<CIrisClass>(m);
  }
  {
    auto cls =
        DefineTemplateClassWithDefault<CPathClass, BaseClass>(
            m, "CIrisSeparatingPlane", param, ciris_cls_doc.doc)
            .def_readonly("a", &BaseClass::a, py_rvp::copy, base_cls_doc.a.doc)
            .def_readonly("b", &BaseClass::b, base_cls_doc.b.doc)
            .def_readonly("positive_side_geometry",
                &BaseClass::positive_side_geometry,
                base_cls_doc.positive_side_geometry.doc)
            .def_readonly("negative_side_geometry",
                &BaseClass::negative_side_geometry,
                base_cls_doc.negative_side_geometry.doc)
            .def_readonly("expressed_body", &BaseClass::expressed_body,
                base_cls_doc.expressed_body.doc)
            .def_readonly("plane_degree", &CPathClass::plane_degree,
                cpath_cls_doc.plane_degree.doc)
            .def_readonly("decision_variables", &BaseClass::decision_variables,
                py_rvp::copy, base_cls_doc.a.doc);
    DefCopyAndDeepCopy(&cls);
    AddValueInstantiation<CIrisClass>(m);
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

    py::enum_<CIrisGeometryType>(
        m, "CIrisGeometryType", doc.CIrisGeometryType.doc)
        .value("kPolytope", CIrisGeometryType::kPolytope,
            doc.CIrisGeometryType.kPolytope.doc)
        .value("kSphere", CIrisGeometryType::kSphere,
            doc.CIrisGeometryType.kSphere.doc)
        .value("kCylinder", CIrisGeometryType::kCylinder,
            doc.CIrisGeometryType.kCylinder.doc)
        .value("kCapsule", CIrisGeometryType::kCapsule,
            doc.CIrisGeometryType.kCapsule.doc);

    py::class_<CIrisCollisionGeometry>(
        m, "CIrisCollisionGeometry", doc.CIrisCollisionGeometry.doc)
        .def("type", &CIrisCollisionGeometry::type,
            doc.CIrisCollisionGeometry.type.doc)
        .def("geometry", &CIrisCollisionGeometry::geometry,
            py_rvp::reference_internal, doc.CIrisCollisionGeometry.geometry.doc)
        .def("body_index", &CIrisCollisionGeometry::body_index,
            doc.CIrisCollisionGeometry.body_index.doc)
        .def("id", &CIrisCollisionGeometry::id,
            doc.CIrisCollisionGeometry.id.doc)
        .def("X_BG", &CIrisCollisionGeometry::X_BG,
            doc.CIrisCollisionGeometry.X_BG.doc)
        .def("num_rationals", &CIrisCollisionGeometry::num_rationals,
            doc.CIrisCollisionGeometry.num_rationals.doc);
  }
  {
    // Definitions for c_iris_separating_plane.h/cc
    py::enum_<SeparatingPlaneOrder>(
        m, "SeparatingPlaneOrder", doc.SeparatingPlaneOrder.doc)
        .value("kAffine", SeparatingPlaneOrder::kAffine,
            doc.SeparatingPlaneOrder.kAffine.doc);
    type_visit([m](auto dummy) { DoSeparatingPlaneDeclaration(m, dummy); },
        type_pack<double, symbolic::Variable>());
  }
  {
    // Definitions for cpsace_free_structs.h/cc
    constexpr auto& result_doc = doc.SeparationCertificateResultBase;
    auto prog_cls =
        py::class_<SeparationCertificateProgramBase>(
            m, "SeparationCertificateResultBase", result_doc.doc)
//            .def_readwrite("prog", &SeparationCertificateProgramBase::prog)
            .def_readwrite(
                "plane_index", &SeparationCertificateProgramBase::plane_index);
    auto result_cls =
        py::class_<SeparationCertificateResultBase>(
            m, "SeparationCertificateResultBase", result_doc.doc)
            .def_readonly("a", &SeparationCertificateResultBase::a)
            .def_readonly("b", &SeparationCertificateResultBase::b)
            .def_readonly("plane_decision_var_vals",
                &SeparationCertificateResultBase::plane_decision_var_vals)
            .def_readonly("result", &SeparationCertificateResultBase::result);
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
            py::arg("s_center"), py::arg("options"), cls_doc.BinarySearch.doc);
    {
      using BaseClass =geometry::optimization::SeparationCertificateResultBase;
      py::class_<Class::SeparationCertificateResult,
                 SeparationCertificateResultBase>(
          cspace_free_polytope_cls, "SeparationCertificateResult",
          cls_doc.SeparationCertificateResult.doc)
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
          .def_readonly("a", &BaseClass::a, py_rvp::copy)
          .def_readonly("b", &BaseClass::b)
           .def_readonly("result", &BaseClass::result)
          .def_readonly("plane_decision_var_vals",
                        &BaseClass::plane_decision_var_vals,
                        py_rvp::copy);
    }
    py::class_<Class::SeparatingPlaneLagrangians>(cspace_free_polytope_cls,
        "SeparatingPlaneLagrangians", cls_doc.SeparatingPlaneLagrangians.doc)
        .def("polytope", &Class::SeparatingPlaneLagrangians::polytope,
            py_rvp::copy)
        .def("s_lower", &Class::SeparatingPlaneLagrangians::s_lower,
            py_rvp::copy)
        .def("s_upper", &Class::SeparatingPlaneLagrangians::s_upper,
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
        .def_readonly(
            "certified_polytope", &Class::SearchResult::certified_polytope);

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

  //  m.def("GetCollisionGeometries",
  //      py::overload_cast<const multibody::MultibodyPlant<double>&,
  //          const geometry::SceneGraph<double>&>(&GetCollisionGeometries),
  //      py::arg("plant"), py::arg("scene_graph"),
  //      doc.GetCollisionGeometries.doc);
}
}  // namespace pydrake
}  // namespace drake
