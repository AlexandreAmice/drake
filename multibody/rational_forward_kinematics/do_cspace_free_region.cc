//
// Created by amice on 2/4/22.
//

#include "drake/multibody/rational_forward_kinematics/do_cspace_free_region.h"

namespace drake{
namespace multibody{


CspaceFreeRegionSolution DoCspaceFreeRegionSearch(
        const systems::Diagram<double>& diagram,
        const multibody::MultibodyPlant<double>* plant,
        const geometry::SceneGraph<double>* scene_graph,
        SeparatingPlaneOrder plane_order, CspaceRegionType cspace_region_type,
        const Eigen::Ref<const Eigen::VectorXd>& seedpoint_q,
        const Eigen::Ref<const Eigen::VectorXd>& q_star,
        const FilteredCollisionPairs& filtered_collision_pairs,
        const Eigen::Ref<const Eigen::MatrixXd>& C_init,
        const Eigen::Ref<const Eigen::VectorXd>& d_init,
        const InterleavedRegionSearchOptions& interleaved_region_search_option,
        const solvers::SolverOptions& solver_options,
        const std::optional<Eigen::MatrixXd>& q_inner_pts_opt,
        const std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>&
            inner_polytope) {
    CspaceFreeRegion free_region = CspaceFreeRegion(diagram, plant,scene_graph,plane_order, cspace_region_type);
    Eigen::MatrixXd q_inner_pts;
    // add the seed point to required inner containment
    if (q_inner_pts_opt.has_value()){
        DRAKE_DEMAND(q_inner_pts_opt.value().rows() == seedpoint_q.rows());
        q_inner_pts.resize(q_inner_pts_opt.value().rows(), q_inner_pts_opt.value().cols() + 1);
        q_inner_pts.leftCols(q_inner_pts_opt.value().cols()) = q_inner_pts_opt.value();
        q_inner_pts.rightCols(1) = seedpoint_q;
    }
    else {
        q_inner_pts.resize(seedpoint_q.rows(), 1);
        q_inner_pts = seedpoint_q;
    }

    Eigen::MatrixXd C_final;
    Eigen::VectorXd d_final;
    Eigen::MatrixXd P_final;
    Eigen::VectorXd q_final;
    std::vector<SeparatingPlane>* separating_planes_sol;

    free_region.InterleavedCSpacePolytopeSearch(
                                    q_star,
                                    filtered_collision_pairs,
                                    C_init,
                                    d_init,
                                    interleaved_region_search_option,
                                    solver_options,
                                    q_inner_pts,
                                    inner_polytope,  &C_final, &d_final,
                                    &P_final,  &q_final,
                                    &separating_planes_sol);
    return CspaceFreeRegionSolution{C_final, d_final, P_final, q_final, separating_planes_sol};

}

} //namespace multibody
} //namespace drake