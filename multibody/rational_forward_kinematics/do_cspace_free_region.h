#pragma once
#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"
#include "drake/multibody/rational_forward_kinematics/centrally_symmetric_hpolytope.h"
namespace drake{
namespace multibody{

/**
* do cspace region search
* @return
*/
CspaceFreeRegionSolution DoCspaceFreeRegionSearch(
    const systems::Diagram<double>& diagram,
    const multibody::MultibodyPlant<double>* plant,
    const geometry::SceneGraph<double>* scene_graph,
    SeparatingPlaneOrder plane_order, CspaceRegionType cspace_region_type,
    const Eigen::Ref<const Eigen::VectorXd>& seedpoint_q,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs,
    const Eigen::Ref<const Eigen::MatrixXd>& C_init,
    const Eigen::Ref<const Eigen::VectorXd>& d_init,
    const CspaceFreeRegion::InterleavedRegionSearchOptions& interleaved_region_search_option,
    const solvers::SolverOptions& solver_options,
    const std::optional<Eigen::MatrixXd>& q_inner_pts,
    const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
        inner_polytope,
    Eigen::MatrixXd* C_final, Eigen::VectorXd* d_final,
    Eigen::MatrixXd* P_final, Eigen::VectorXd* q_final);

std::vector<CspaceFreeRegionSolution> DoCspaceFreeRegionSearchMultiSeedDefaultRegions(
        const systems::Diagram<double>& diagram,
        const multibody::MultibodyPlant<double>* plant,
        const geometry::SceneGraph<double>* scene_graph,
        SeparatingPlaneOrder plane_order, CspaceRegionType cspace_region_type,
        const Eigen::Ref<const Eigen::MatrixXd>& seedpoints_q,
        const Eigen::Ref<const Eigen::VectorXd>& q_star,
        const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs,
        const Eigen::Ref<const Eigen::MatrixXd>& C_init,
        const Eigen::Ref<const Eigen::VectorXd>& d_init,
        const CspaceFreeRegion::InterleavedRegionSearchOptions& interleaved_region_search_option,
        const solvers::SolverOptions& solver_options,
        const std::optional<std::vector<Eigen::MatrixXd>>& q_inner_pts_vect,
        const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
            inner_polytope, Eigen::VectorXd* d_final,
        Eigen::MatrixXd* P_final, Eigen::VectorXd* q_final);

} //namespace multibody
} //namespace drake