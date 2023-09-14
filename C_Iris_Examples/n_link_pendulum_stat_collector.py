import numpy as np
from functools import partial
import visualization_utils as vis_utils
from iris_plant_visualizer import IrisPlantVisualizer
import ipywidgets as widgets
from IPython.display import display
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import pickle

from pydrake.common import FindResourceOrThrow
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import Role, GeometrySet, CollisionFilterDeclaration
from pydrake.all import RigidTransform, RollPitchYaw, RevoluteJoint
from pydrake.all import RotationMatrix, MeshcatVisualizer, StartMeshcat
import pydrake.symbolic as sym
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions, ScsSolver
from pydrake.all import (
    PointCloud,
    MeshcatVisualizerParams,
    Role,
    HalfSpace,
    CoulombFriction,
    Box,
    Rgba,
    WeldJoint,
    GeometrySet,
)
import pickle
from pydrake.polynomial import Polynomial as PolynomialCommon
import time
from scipy.spatial.transform import Rotation as sp_rot
from sampling_based_motion_planners import StraightLineCollisionChecker, PRMFixedEdges


from pydrake.all import (
    RationalForwardKinematics,
    ModelInstanceIndex,
    SpatialInertia,
    RevoluteJoint,
    FixedOffsetFrame,
    MultibodyPlant,
)
from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
from pydrake.geometry.optimization_dev import CspaceFreePath
from joblib import Parallel, delayed

import numpy as np
from functools import partial
import visualization_utils as vis_utils
from iris_plant_visualizer import IrisPlantVisualizer
import ipywidgets as widgets
from IPython.display import display
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from pathlib import Path
import os

# pydrake imports
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import Role, GeometrySet, CollisionFilterDeclaration
from pydrake.all import RigidTransform, RollPitchYaw, RevoluteJoint
from pydrake.all import RotationMatrix, MeshcatVisualizer, StartMeshcat, Sphere
import pydrake.symbolic as sym
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions, ScsSolver
from pydrake.all import (
    PointCloud,
    MeshcatVisualizerParams,
    Role,
    HalfSpace,
    CoulombFriction,
    Box,
    Rgba,
    WeldJoint,
    GeometrySet,
)
from pydrake.polynomial import Polynomial as PolynomialCommon
import time
from scipy.spatial.transform import Rotation as sp_rot
from sampling_based_motion_planners import (
    StraightLineCollisionChecker,
    PRM,
    BiRRT,
    PRMFixedEdges,
)


from pydrake.all import (
    RationalForwardKinematics,
    ModelInstanceIndex,
    SpatialInertia,
    RevoluteJoint,
    FixedOffsetFrame,
    MultibodyPlant,
)
from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
from pydrake.geometry.optimization_dev import CspaceFreePath
from dataclasses import dataclass

import logging

drake_logger = logging.getLogger("drake")
drake_logger.setLevel(logging.DEBUG)


def AddPendulumLink(
    plant,
    pendulum_model_instance,
    link_dimensions,
    parent_frame,
    link_index: int,
    color=np.array([1, 0, 0, 1]),
):
    pendulum_box = Box(*link_dimensions)
    pendulum_inertia = SpatialInertia.SolidBoxWithDensity(1.0, *link_dimensions)

    link_name = f"link_{link_index}"
    pend_body = plant.AddRigidBody(link_name, pendulum_model_instance, pendulum_inertia)
    parent_to_child_attach_frame = plant.AddFrame(
        FixedOffsetFrame(
            f"{link_name}_joint_frame",
            pend_body.body_frame(),
            RigidTransform(p=(0, link_dimensions[1] / 2, 0)),
        )
    )
    next_frame = plant.AddFrame(
        FixedOffsetFrame(
            f"{link_name}_joint_frame",
            pend_body.body_frame(),
            RigidTransform(p=(0, -link_dimensions[1] / 2, 0)),
        )
    )
    # add the revolute joint,
    joint_lim = np.pi
    plant.AddJoint(
        RevoluteJoint(
            f"joint_{link_index}",
            parent_frame,
            parent_to_child_attach_frame,
            [0, 0, 1],
            -joint_lim,
            joint_lim,
        )
    )

    geom_id = plant.RegisterCollisionGeometry(
        pend_body, RigidTransform(), pendulum_box, link_name, CoulombFriction()
    )
    plant.RegisterVisualGeometry(
        pend_body, RigidTransform(), pendulum_box, link_name, color
    )

    return next_frame, geom_id


def N_Link_Pendulum(
    n: int,
    plant: MultibodyPlant,
    pendulum_model_instance: ModelInstanceIndex,
    link_dimensions: np.ndarray,
):
    assert n > 0
    parent_frame = plant.world_frame()
    colors = np.array(vis_utils.n_colors(n)) / 255
    colors = np.hstack([colors, np.ones((colors.shape[0], 1))])
    geom_ids = []
    for i in range(n):
        parent_frame, geom_id = AddPendulumLink(
            plant,
            pendulum_model_instance,
            link_dimensions,
            parent_frame,
            i + 1,
            colors[i],
        )
        geom_ids.append(geom_id)
    return geom_ids


def AddRandomBox(
    plant: MultibodyPlant,
    obstacle_model_instance: ModelInstanceIndex,
    index: int,
    pos_limits,
    size_limits,
    color=[0, 0, 0, 1],
):
    link_name = f"obstacle_{index}"

    def random_center():
        dist = pos_limits[1] * np.sqrt(np.random.uniform(0, 1))
        ang = np.random.uniform(0, 2 * np.pi)
        return dist * np.array([np.cos(ang), np.sin(ang), 0])

    center = (
        random_center()
    )  # np.append(np.random.uniform(pos_limits[0], pos_limits[1], 2), 0)

    dimensions = np.append(np.random.uniform(size_limits[0], size_limits[1], 2), 0.001)
    box = Box(*dimensions)
    inertia = SpatialInertia.SolidBoxWithDensity(1, *dimensions)

    body = plant.AddRigidBody(link_name, obstacle_model_instance, inertia)

    random_rp = np.zeros(3)
    lim = np.pi / 10
    random_rp[2] = np.random.uniform(-lim, lim)
    angle = RotationMatrix(RollPitchYaw(random_rp))
    pose = RigidTransform(p=center) @ RigidTransform(R=angle)
    origin_local = pose.inverse() @ RigidTransform()
    while np.all(
        [
            -dimensions[i] / 2 <= origin_local.translation()[i] < dimensions[i] / 2
            for i in range(3)
        ]
    ):
        random_rp[2] = np.random.uniform(-lim, lim)
        angle = RotationMatrix(RollPitchYaw(random_rp))
        center = random_center()
        pose = RigidTransform(p=center) @ RigidTransform(R=angle)
        origin_local = pose.inverse() @ RigidTransform()

    plant.AddJoint(WeldJoint(link_name, plant.world_frame(), body.body_frame(), pose))
    col_id = plant.RegisterCollisionGeometry(
        body, RigidTransform(), box, link_name, CoulombFriction()
    )
    plant.RegisterVisualGeometry(body, RigidTransform(), box, link_name, color)
    return col_id


def AddNRandomBoxes(
    n: int,
    plant: MultibodyPlant,
    obstacle_model_instance: ModelInstanceIndex,
    pos_limits=(-10, 10),
    size_limits=(0.1, 1),
    color=[0, 0, 0, 1],
):
    assert n > 0
    joint_sphere = Sphere(0.05)
    plant.RegisterVisualGeometry(
        plant.world_body(),
        RigidTransform(),
        joint_sphere,
        "pend_start",
        np.array([0, 0, 1, 1]),
    )
    geom_ids = []
    for i in range(n):
        geom_ids.append(
            AddRandomBox(
                plant, obstacle_model_instance, i, pos_limits, size_limits, color
            )
        )
    return geom_ids


def make_line_polys(prm, max_num_edges=-1):
    polys = np.empty(
        shape=(
            experiment.plant.num_positions(),
            len(prm.prm.edges()) if max_num_edges < 0 else max_num_edges,
        ),
        dtype=object,
    )
    for i, (s0, s1) in enumerate(prm.prm.edges()):
        for j in range(experiment.plant.num_positions()):
            if max_num_edges > 0 and not i < max_num_edges:
                break
            polys[j, i] = PolynomialCommon(np.array([s0[j], s1[j] - s0[j]]))

    return polys


class N_Link_K_Boxes_Experiment:
    def __init__(
        self,
        n: int,
        k: int,
        exclude_pend_self_collisions=True,
        plane_order=1,
        maximum_path_degree=1,
        meshcat_instance=None,
    ):
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=0.001
        )

        self.link_dimensions = (0.1, 0.2, 0.001)
        self.pendulum_model_instance = self.plant.AddModelInstance("pendulum")
        self.obstacle_model_instance = self.plant.AddModelInstance("obstacle")

        num_pend = n
        num_obstacles = k
        np.random.seed(num_pend * num_obstacles)

        self.pend_geom_ids = N_Link_Pendulum(
            num_pend,
            self.plant,
            self.pendulum_model_instance,
            np.array(self.link_dimensions),
        )
        self.pend_geom_set = GeometrySet(self.pend_geom_ids)

        if k > 0:
            obstacle_pos_limits = (
                self.link_dimensions[1] * num_pend * np.array([-1.25, 1.25])
            )
            obstacle_size_limits = (
                0.01 / np.log(num_obstacles),
                0.5 / np.log(num_obstacles),
            )
            self.obstacle_geom_ids = AddNRandomBoxes(
                num_obstacles,
                self.plant,
                self.obstacle_model_instance,
                obstacle_pos_limits,
                obstacle_size_limits,
            )
            self.obstacle_geom_set = GeometrySet(self.obstacle_geom_ids)
        else:
            self.obstacle_geom_ids = None
            self.obstacle_geom_set = None

        self.plant.Finalize()
        if exclude_pend_self_collisions:
            collision_filter_manager = self.scene_graph.collision_filter_manager()
            decl = CollisionFilterDeclaration().ExcludeWithin(self.pend_geom_set)
            collision_filter_manager.Apply(decl)

        if meshcat_instance is not None:
            self.visualizer = MeshcatVisualizer.AddToBuilder(
                builder, self.scene_graph, meshcat_instance
            )

        self.diagram = builder.Build()

        self.q_star = np.zeros(self.plant.num_positions())

        if maximum_path_degree > 0:
            t0 = time.time()
            self.cspace_free_path = CspaceFreePath(
                self.plant,
                self.scene_graph,
                self.q_star,
                maximum_path_degree=maximum_path_degree,
                plane_order=plane_order,
            )
            t1 = time.time()
            print(
                f"Time to construct line certifier for {n}-links, {k}-boxes = {t1 - t0}s"
            )
        else:
            print("Did not build cspace free path")
            self.cspace_free_path = None


path_to_data_save_folder = Path(
    "/home/amice/Documents/coding_projects/drake/C_Iris_Examples/final_experiment_data"
)
n = 12
plane_order = 1
maximum_path_degree = 1
k = 100
num_obstacles = k
l = 100

num_edges = l
if __name__ == "__main__":
    meshcat = StartMeshcat()
    experiment = N_Link_K_Boxes_Experiment(
        n,
        k,
        exclude_pend_self_collisions=True,
        plane_order=plane_order,
        maximum_path_degree=maximum_path_degree,
        meshcat_instance=meshcat,
    )
    diagram_context = experiment.diagram.CreateDefaultContext()
    plant_context = experiment.diagram.GetMutableSubsystemContext(
        experiment.plant, diagram_context
    )
    experiment.diagram.ForcedPublish(diagram_context)

    ######### BUILD PRM ##########
    Ratfk = RationalForwardKinematics(experiment.plant)
    diagram_col_context = experiment.diagram.CreateDefaultContext()
    plant_col_context = experiment.diagram.GetMutableSubsystemContext(
        experiment.plant, diagram_col_context
    )
    scene_graph_col_context = experiment.diagram.GetMutableSubsystemContext(
        experiment.scene_graph, diagram_col_context
    )
    query_port = experiment.scene_graph.get_query_output_port()

    def check_collision_q_by_query(q):
        if np.all(q >= experiment.plant.GetPositionLowerLimits()) and np.all(
            q <= experiment.plant.GetPositionUpperLimits()
        ):
            experiment.plant.SetPositions(plant_col_context, q)
            query_object = query_port.Eval(scene_graph_col_context)
            return 1 if query_object.HasCollisions() else 0
        else:
            return 1

    def check_collision_s_by_query(s):
        s = np.array(s)
        q = Ratfk.ComputeQValue(s, experiment.q_star)
        return check_collision_q_by_query(q)

    def sample_col_free_point():
        q = np.random.uniform(
            experiment.plant.GetPositionLowerLimits(),
            experiment.plant.GetPositionUpperLimits(),
        )
        s = Ratfk.ComputeSValue(q, experiment.q_star)
        return s

    collision_checker = StraightLineCollisionChecker(check_collision_s_by_query, 100)
    prm_save_name = (
        path_to_data_save_folder / f"{n}_link_{k}_obstacles_{l}_edges_PRM.pkl"
    )
    print(str(prm_save_name))
    if not prm_save_name.exists():
        prm = PRMFixedEdges(
            sample_col_free_point, l, collision_checker, dist_thresh=100
        )
        with open(prm_save_name, "wb") as f:
            pickle.dump(prm, f)
    else:
        with open(prm_save_name, "rb") as f:
            prm = pickle.load(f)

    diagram_vis_context = experiment.diagram.CreateDefaultContext()
    plant_vis_context = experiment.diagram.GetMutableSubsystemContext(
        experiment.plant, diagram_vis_context
    )
    vis_bundle = vis_utils.VisualizationBundle(
        experiment.diagram,
        diagram_context,
        experiment.plant,
        plant_vis_context,
        Ratfk,
        meshcat,
        experiment.q_star,
    )
    end_effector = experiment.plant.GetBodyByName(f"link_{n}")

    prm.draw_tree(vis_bundle, end_effector)
    path_safe = make_line_polys(prm)
    ############## CERTIFICATION ##################
    cert_options = CspaceFreePath.FindSeparationCertificateGivenPathOptions()
    cert_options.terminate_segment_certification_at_failure = False

    cert_options.num_threads = -1
    cert_options.verbose = False
    cert_options.solver_id = MosekSolver.id()
    cert_options.solver_options = SolverOptions()
    cert_options.terminate_path_certification_at_failure = False

    for num_links in range(2, n + 1):
        ignored_col_set = set(
            [
                elt
                for elt in zip(
                    experiment.pend_geom_ids[:-1], experiment.pend_geom_ids[1:]
                )
            ]
        )
        for j in range(
            num_links, experiment.plant.num_positions()
        ):  # +1 for num joints +1 to include last joint name
            pend_j_col_geom = experiment.pend_geom_ids[j]
            for obs_id in experiment.obstacle_geom_ids:
                ignored_col_set.add((pend_j_col_geom, obs_id))
        t0 = time.time()
        (
            statistics,
            cert_result,
        ) = experiment.cspace_free_path.FindSeparationCertificateGivenPath(
            path_safe, ignored_col_set, cert_options
        )
        t1 = time.time()
        file_name = path_to_data_save_folder / (
            f"{num_links}_links_{k}_obstacles_{l}_edges.pkl"
        )
        with open(
            file_name,
            "wb",
        ) as f:
            pickle.dump(statistics, f)
        print(f"num pairs to certify {len(statistics[0].total_time_to_certify_pair)}")
        print(f"Certification of safe PRM for {num_links} links in {t1 - t0}s")
        print(
            f"Frac edges safe = {sum([1 if s.certified_safe() else 0 for s in statistics])}/{len(statistics)}"
        )
        print(
            f"Unsafe inds = {[idx for idx, s in enumerate(statistics) if not s.certified_safe()]}"
        )
    print("DONE!")
