#include "drake/multibody/rational_forward_kinematics/collision_geometry.h"

#include <vector>

#include <drake_vendor/libqhullcpp/Qhull.h>

#include "drake/geometry/read_obj.h"

namespace drake {
namespace multibody {
CollisionGeometry::CollisionGeometry(CollisionGeometryType type,
                                     const geometry::Shape* geometry,
                                     multibody::BodyIndex body_index,
                                     geometry::GeometryId id,
                                     math::RigidTransformd X_BG)
    : type_{type},
      geometry_{std::move(geometry)},
      body_index_{body_index},
      id_{id},
      X_BG_{std::move(X_BG)} {}

Eigen::Matrix3Xd GetVertices(const geometry::Shape& shape) {
  if (dynamic_cast<const geometry::Box*>(&shape) != nullptr) {
    const auto* box = dynamic_cast<const geometry::Box*>(&shape);
    Eigen::Matrix<double, 3, 8> vertices;
    // clang-format off
    vertices << -1, 1, 1, -1, -1, 1, 1, -1,
                1, 1, -1, -1, -1, -1, 1, 1,
                -1, -1, -1, -1, 1, 1, 1, 1;
    // clang-format on
    vertices.row(0) *= box->width() / 2;
    vertices.row(1) *= box->depth() / 2;
    vertices.row(2) *= box->height() / 2;
    return vertices;
  } else if (dynamic_cast<const geometry::Convex*>(&shape) != nullptr) {
    // Getting the vertices from an obj file. First convert it to a qhull
    // object, and then read the vertices from this qhull object
    // TODO(hongkai.dai): call VPolytope::ImplementGeometry directly when this
    // function is not private.
    const auto* convex = dynamic_cast<const geometry::Convex*>(&shape);
    const auto [tinyobj_vertices, faces, num_faces] =
        geometry::internal::ReadObjFile(convex->filename(), convex->scale(),
                                        false /* triangulate */);
    unused(faces);
    unused(num_faces);
    orgQhull::Qhull qhull;
    const int dim = 3;
    std::vector<double> tinyobj_vertices_flat(tinyobj_vertices->size() * dim);
    for (int i = 0; i < static_cast<int>(tinyobj_vertices->size()); ++i) {
      for (int j = 0; j < dim; ++j) {
        tinyobj_vertices_flat[dim * i + j] = (*tinyobj_vertices)[i](j);
      }
    }
    qhull.runQhull("", dim, tinyobj_vertices->size(),
                   tinyobj_vertices_flat.data(), "");
    if (qhull.qhullStatus() != 0) {
      throw std::runtime_error(
          fmt::format("Qhull terminated with status {} and  message:\n{}",
                      qhull.qhullStatus(), qhull.qhullMessage()));
    }
    Eigen::Matrix3Xd vertices(3, qhull.vertexCount());
    int vertex_count = 0;
    for (const auto& qhull_vertex : qhull.vertexList()) {
      vertices.col(vertex_count++) = Eigen::Map<Eigen::Vector3d>(
          qhull_vertex.point().toStdVector().data());
    }
    return vertices;
  } else {
    throw std::invalid_argument("GetVertices() only works for Box and Convex");
  }
}
}  // namespace multibody
}  // namespace drake
