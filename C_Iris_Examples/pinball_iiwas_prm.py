import numpy as np
from functools import partial
import visualization_utils as viz_utils
from iris_plant_visualizer import IrisPlantVisualizer
import ipywidgets as widgets
from IPython.display import display
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from pathlib import Path
import os
#pydrake imports
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import Role, GeometrySet, CollisionFilterDeclaration
from pydrake.all import RigidTransform, RollPitchYaw, RevoluteJoint
from pydrake.all import RotationMatrix, FindResourceOrThrow
import pydrake.symbolic as sym
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions, ScsSolver

import time
from pydrake.polynomial import Polynomial as PolynomialCommon

from pydrake.all import RationalForwardKinematics
from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
from pydrake.geometry.optimization_dev import CspaceFreePath
import logging
drake_logger = logging.getLogger("drake")
# drake_logger.setLevel(logging.DEBUG)

# construct our robot
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parser = Parser(plant)
oneDOF_iiwa_file = FindResourceOrThrow("drake/C_Iris_Examples/assets/oneDOF_iiwa7_with_box_collision.sdf")
# with open(oneDOF_iiwa_file, 'r') as f:
#     oneDOF_iiwa_string = f.read()
box_asset_file = FindResourceOrThrow("drake/C_Iris_Examples/assets/box_small.urdf")
# with open(box_asset_file, 'r') as f:
#     box_asset_string = f.read()

models = []
models.append(parser.AddModelFromFile(box_asset_file))
models.append(parser.AddModelFromFile(oneDOF_iiwa_file, 'right_sweeper'))
models.append(parser.AddModelFromFile(oneDOF_iiwa_file, 'left_sweeper'))

# models = []
# models.append(parser.AddModelFromFile(box_asset))
# models.append(parser.AddModelFromFile(oneDOF_iiwa_asset, 'right_sweeper'))
# models.append(parser.AddModelFromFile(oneDOF_iiwa_asset, 'left_sweeper'))

locs = [[0., 0., 0.],
        [0, 1, 0.85],
        [0, -1, 0.55]]
plant.WeldFrames(plant.world_frame(),
                 plant.GetFrameByName("base", models[0]),
                 RigidTransform(locs[0]))

t1 = RigidTransform(RollPitchYaw([np.pi / 2, 0, 0]).ToRotationMatrix(), locs[1]) @ RigidTransform(
    RollPitchYaw([0, 0, np.pi / 2]), np.zeros(3))
t2 = RigidTransform(RollPitchYaw([-np.pi / 2, 0, 0]).ToRotationMatrix(), locs[2]) @ RigidTransform(
    RollPitchYaw([0, 0, np.pi / 2]), np.zeros(3))
plant.WeldFrames(plant.world_frame(),
                 plant.GetFrameByName("iiwa_oneDOF_link_0", models[1]),
                 t1)
plant.WeldFrames(plant.world_frame(),
                 plant.GetFrameByName("iiwa_oneDOF_link_0", models[2]),
                 t2)

plant.Finalize()
idx = 0
q0 = [0.0, 0.0]
val = 1.7
q_low = np.array([-val, -val])
q_high = np.array([val, val])
# set the joint limits of the plant
for model in models:
    for joint_index in plant.GetJointIndices(model):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[idx])
            joint.set_position_limits(lower_limits=np.array([q_low[idx]]), upper_limits=np.array([q_high[idx]]))
            idx += 1

# construct the RationalForwardKinematics of this plant. This object handles the
# computations for the forward kinematics in the tangent-configuration space
Ratfk = RationalForwardKinematics(plant)

# the point about which we will take the stereographic projections
q_star = np.zeros(plant.num_positions())

do_viz = True

# The object we will use to perform our certification.
t0 = time.time()
cspace_free_path = CspaceFreePath(plant, scene_graph, q_star, maximum_path_degree=1, plane_order=3)
t1 = time.time()
print(f"Time to construct line certifier = {t1 - t0}s")

# This line builds the visualization. Change the viz_role to Role.kIllustration if you
# want to see the plant with its illustrated geometry or to Role.kProximity if you want
# to see the plant with the collision geometries.
visualizer = IrisPlantVisualizer(plant, builder, scene_graph, cspace_free_path, viz_role=Role.kIllustration)
visualizer.visualize_collision_constraint(factor=1.2, num_points=100)
visualizer.meshcat_cspace.Set2dRenderMode(RigidTransform(RotationMatrix.MakeZRotation(0), np.array([0, 0, 1])))
visualizer.meshcat_task_space.Set2dRenderMode(RigidTransform(RotationMatrix.MakeZRotation(0), np.array([1, 0, 0])))

#compute limits in s-space
limits_s = []
for q in [q_low, q_high]:
    limits_s.append(Ratfk.ComputeSValue(np.array(q), q_star))
limits_s = np.array(limits_s)


visualizer.meshcat_cspace.Delete("/prm")
# draw prm
import prm
from pydrake.all import Rgba

# limits = [np.array(t_low), np.array(q_high)]

visualizer.meshcat_cspace.Delete("/prm")


def plot_prm(nodes, adjacency_list, width, color=Rgba(0.0, 0.0, 1, 1), prefix=""):
    plt_idx = 0
    for node_idx in range(nodes.shape[0]):
        pos1 = np.append(nodes[node_idx, :], 0)
        for edge_idx in range(len(adjacency_list[node_idx])):
            pos2 = np.append(nodes[adjacency_list[node_idx][edge_idx], :], 0)
            name = f"/{prefix}/prm/rm/line/{plt_idx}"
            #             for c in map(int, sqtring):
            #                 name += f"/{c}"
            visualizer.meshcat_cspace.SetLine(name, np.hstack([pos1[:, np.newaxis], pos2[:, np.newaxis]]),
                                              line_width=width, rgba=color)
            plt_idx += 1


plotting_fn_handle_good = partial(plot_prm, width=0.1, prefix="Good", color=Rgba(0, 1., 0, 1))
plotting_fn_handle_bad = partial(plot_prm, width=0.1, prefix="Bad", color=Rgba(1., 0., 1, 1))


def collision(pos, col_func_handle):
    return col_func_handle(pos)


def collision_bad(pos, col_func_handle):
    return 1 - col_func_handle(pos)


prm_col_fn_handle = partial(collision, col_func_handle=visualizer.check_collision_s_by_ik)
prm_col_fn_handle_bad = partial(collision_bad, col_func_handle=visualizer.check_collision_s_by_ik)

visualizer.check_collision_s_by_ik(np.array([0, 0]))
prm_col_fn_handle(np.array([0, 0]))


num_points = 100

PRM = prm.PRM(
            limits_s,
            num_points = num_points,
            col_func_handle = prm_col_fn_handle,
            num_neighbours = 5,
            dist_thresh = .5,
            num_col_checks = 10,
            verbose = True,
            plotcallback = plotting_fn_handle_good
            )

PRM_bad = prm.PRM(
            limits_s,
            num_points = num_points,
            col_func_handle = prm_col_fn_handle_bad,
            num_neighbours = 5,
            dist_thresh = .5,
            num_col_checks = 10,
            verbose = True,
            plotcallback = plotting_fn_handle_bad
            )

# PRM.add_start_end(start, target)
if num_points < 100:
    PRM.plot()
    PRM_bad.plot()
tot_num_edges = len(PRM.adjacency_list)* len(PRM.adjacency_list[0])
# path, sp_length = PRM.find_shortest_path()

# mat = meshcat.geometry.MeshLambertMaterial(color= 0xFFF812 , wireframe=True)
# mat.wireframeLinewidth = 2.0
# num_waypoints = len(path)
# for idx in range(num_waypoints-1):
#     vis2['prm']['path']['path' + str(idx)].set_object( meshcat_line(path[idx], path[idx+1],width = 0.01), mat)
# traj= utils.PWLinTraj(path, 5.0)

q = q0.copy()


def make_line_polys(PRM):
    endpoint_index_set = set()
    for neighbors in PRM.adjacency_list:
        for n in neighbors[1:]:
            endpoint_index_set.add((neighbors[0], n))
    polys = np.empty(shape=(plant.num_positions(), len(endpoint_index_set)), dtype=object)
    for i, (idx0, idx1) in enumerate(endpoint_index_set):
        s0 = PRM.nodes[idx0]
        s1 = PRM.nodes[idx1]
        for j in range(plant.num_positions()):
            polys[j, i] = PolynomialCommon(np.array([s0[j] - s1[j], s1[j]]))
    return polys


path_safe = make_line_polys(PRM)

cert_options = CspaceFreePath.FindSeparationCertificateGivenPathOptions()
cert_options.terminate_segment_certification_at_failure = False

cert_options.num_threads = -1
cert_options.verbose = False
cert_options.solver_id = MosekSolver.id()
cert_options.solver_options = SolverOptions()
cert_options.terminate_path_certification_at_failure = False

segment = path_safe[:,0]
geom_pair = next(iter(cspace_free_path.map_geometries_to_separating_planes().keys()))
prog = cspace_free_path.MakeIsGeometrySeparableOnPathProgram(geom_pair, segment)
t0 = time.time()
result = cspace_free_path.SolveSeparationCertificateProgram(prog, cert_options)
t1 = time.time()
print(f"Certification of safe PRM in {t1-t0}s")

t0 = time.time()
ret = cspace_free_path.FindSeparationCertificateGivenPath(path_safe, set(), cert_options)
t1 = time.time()
print(f"Certification of safe PRM in {t1-t0}s")

print("done")
