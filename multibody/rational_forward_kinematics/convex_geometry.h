#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/rational_forward_kinematics/plane_side.h"
#include "drake/multibody/tree/multibody_tree_indexes.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace multibody {
enum class ConvexGeometryType {
  kPolytope,
  kCylinder,
  kEllipsoid,
};

class ConvexGeometry {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ConvexGeometry)
  typedef geometry::GeometryId Id;

  ConvexGeometryType type() const { return type_; }

  drake::multibody::BodyIndex body_index() const { return body_index_; }

  virtual ~ConvexGeometry() {}

  /** Add the constraint that the geometry is on one side of the plane
   * nᵀ(x-c) = 1. Namely nᵀ(x-c) ≤ 1 ∀ x within the convex geometry.
   * Here we assume that n is expressed in the same body frame as the geometry.
   */
  virtual void AddInsideHalfspaceConstraint(
      const Eigen::Ref<const Eigen::Vector3d>& p_BC,
      const Eigen::Ref<const drake::Vector3<drake::symbolic::Variable>>& n_B,
      drake::solvers::MathematicalProgram* prog) const = 0;

  /**
   * Adds the constraint that a point Q is within the convex geometry.
   * @param X_AB The pose of the body B (to which the geometry is attached) in
   * a frame A.
   * @param p_AQ The decision variables representing the position of point Q in
   * frame A.
   */
  virtual void AddPointInsideGeometryConstraint(
      const Eigen::Isometry3d& X_AB,
      const Eigen::Ref<const drake::Vector3<drake::symbolic::Variable>>& p_AQ,
      drake::solvers::MathematicalProgram* prog) const = 0;

  virtual const Eigen::Vector3d& p_BC() const = 0;

  drake::geometry::GeometryId get_id() const { return id_; }

  bool IsInCollision(const ConvexGeometry& other,
                     const drake::math::RigidTransform<double>& X_ASelf,
                     const drake::math::RigidTransform<double>& X_AOther) const;

 protected:
  ConvexGeometry(ConvexGeometryType type,
                 drake::multibody::BodyIndex body_index,
                 drake::geometry::GeometryId id);

 private:
  ConvexGeometryType type_;
  // The index of the body that this geometry is attached to.
  drake::multibody::BodyIndex body_index_;
  drake::geometry::GeometryId id_;
};

/**
 * A convex polyhedron can be described as ConvexHull(V) ⊕ ConvexCone(R), where
 * vᵢ are the vertices, and rⱼ are the rays.
 */
class ConvexPolytope : public ConvexGeometry {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ConvexPolytope)
  ConvexPolytope(drake::multibody::BodyIndex body_index,
                 drake::geometry::GeometryId id,
                 const Eigen::Ref<const Eigen::Matrix3Xd>& vertices);

  ConvexPolytope(drake::multibody::BodyIndex body_index,
                 drake::geometry::GeometryId id,
                 const Eigen::Ref<const Eigen::Matrix3Xd>& vertices,
                 const Eigen::Ref<const Eigen::Matrix3Xd>& rays);

  const Eigen::Matrix3Xd& p_BV() const { return p_BV_; }

  void AddInsideHalfspaceConstraint(
      const Eigen::Ref<const Eigen::Vector3d>& p_BC,
      const Eigen::Ref<const drake::Vector3<drake::symbolic::Variable>>& n_B,
      drake::solvers::MathematicalProgram* prog) const override;

  void AddPointInsideGeometryConstraint(
      const Eigen::Isometry3d& X_AB,
      const Eigen::Ref<const drake::Vector3<drake::symbolic::Variable>>& p_AQ,
      drake::solvers::MathematicalProgram* prog) const override;

  const Eigen::Vector3d& p_BC() const override { return p_BC_; }

  /**
   * r_B.col(i) is the i'th ray of the convex cone, expressed in the body frame.
   */
  const Eigen::Matrix3Xd r_B() const { return r_B_; }

 private:
  // position of all vertices V in the body frame B.
  Eigen::Matrix3Xd p_BV_;
  // The position of the geometry center in the body frame.
  Eigen::Vector3d p_BC_;
  Eigen::Matrix3Xd r_B_;
};

class Cylinder : public ConvexGeometry {
 public:
  /**
   * @param p_BO The position of cylinder center O in the body frame B.
   * @param a_B The cylinder axis a expressed in the body frame B. The height
   * of the cylinder is 2 * |a_B|
   * @param radius The radius of the cylinder.
   */
  Cylinder(drake::multibody::BodyIndex body_index,
           drake::geometry::GeometryId id,
           const Eigen::Ref<const Eigen::Vector3d>& p_BO,
           const Eigen::Ref<const Eigen::Vector3d>& a_B, double radius);

  void AddInsideHalfspaceConstraint(
      const Eigen::Ref<const Eigen::Vector3d>& p_BC,
      const Eigen::Ref<const drake::Vector3<drake::symbolic::Variable>>& n_B,
      drake::solvers::MathematicalProgram* prog) const override;

  void AddPointInsideGeometryConstraint(
      const Eigen::Isometry3d& X_AB,
      const Eigen::Ref<const drake::Vector3<drake::symbolic::Variable>>& p_AQ,
      drake::solvers::MathematicalProgram* prog) const override;

  const Eigen::Vector3d& p_BO() const { return p_BO_; }

  const Eigen::Vector3d& a_B() const { return a_B_; }

  double radius() const { return radius_; }

  const Eigen::Vector3d& p_BC() const override { return p_BO_; }

 private:
  // The position of the cylinder center O in the body frame B.
  const Eigen::Vector3d p_BO_;
  // The axis (unnormalized) of the cylinder in the body frame B . The height of
  // the cylinder is 2 * |a_B_|.
  const Eigen::Vector3d a_B_;
  // The radius of the cylinder.
  const double radius_;
  // a_B_ / |a_B_|
  const Eigen::Vector3d a_normalized_B_;
  // â₁, â₂ are the two unit length vectors that are orthotonal to a, and also
  // â₁ ⊥ â₂.
  Eigen::Vector3d a_hat1_B_, a_hat2_B_;
};


}  // namespace multibody
}  // namespace drake
