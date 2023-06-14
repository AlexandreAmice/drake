import numpy as np
from functools import partial
import old_vis_utils as viz_utils
from iris_plant_visualizer_old import IrisPlantVisualizer
import ipywidgets as widgets
from IPython.display import display
#pydrake imports
import pydrake
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import Role, GeometrySet, CollisionFilterDeclaration
from pydrake.solvers import mathematicalprogram as mp
from pydrake.all import RigidTransform, RollPitchYaw, RevoluteJoint

import pydrake.multibody.rational as rational_forward_kinematics
from pydrake.all import RationalForwardKinematics
from pydrake.geometry.optimization import (#IrisOptionsRationalSpace, 
                                           IrisInRationalConfigurationSpace, 
                                           HPolyhedron, 
                                           Hyperellipsoid,
                                           Iris, IrisOptions)
from dijkstraspp import DijkstraSPPsolver    

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parser = Parser(plant)
oneDOF_iiwa_asset = "../../drake/C_Iris_Examples/assets/oneDOF_iiwa7_with_box_collision.sdf"#FindResourceOrThrow("drake/C_Iris_Examples/assets/oneDOF_iiwa7_with_box_collision.sdf")
twoDOF_iiwa_asset = "../../drake/C_Iris_Examples/assets/twoDOF_iiwa7_with_box_collision.sdf"#FindResourceOrThrow("drake/C_Iris_Examples/assets/twoDOF_iiwa7_with_box_collision.sdf")

box_asset = "../../drake/C_Iris_Examples/assets/box_small.urdf" #FindResourceOrThrow("drake/C_Iris_Examples/assets/box_small.urdf")

models = []
models.append(parser.AddModelFromFile(box_asset))
models.append(parser.AddModelFromFile(twoDOF_iiwa_asset))
models.append(parser.AddModelFromFile(oneDOF_iiwa_asset))



locs = [[0.,0.,0.],
        [0.,.55,0.],
        [0.,-.55,0.]]
plant.WeldFrames(plant.world_frame(), 
                 plant.GetFrameByName("base", models[0]),
                 RigidTransform(locs[0]))
plant.WeldFrames(plant.world_frame(), 
                 plant.GetFrameByName("iiwa_twoDOF_link_0", models[1]), 
                 RigidTransform(RollPitchYaw([0,0, -np.pi/2]).ToRotationMatrix(), locs[1]))
plant.WeldFrames(plant.world_frame(), 
                 plant.GetFrameByName("iiwa_oneDOF_link_0", models[2]), 
                 RigidTransform(RollPitchYaw([0,0, -np.pi/2]).ToRotationMatrix(), locs[2]))


plant.Finalize()

idx = 0
q0 = [0.0, 0.0, 0.0]
q_low  = [-1.7, -2., -1.7]
q_high = [ 1.7,  2.,  1.7]
# set the joint limits of the plant
for model in models:
    for joint_index in plant.GetJointIndices(model):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[idx])
            joint.set_position_limits(lower_limits= np.array([q_low[idx]]), upper_limits= np.array([q_high[idx]]))
            idx += 1
        
# construct the RationalForwardKinematics of this plant. This object handles the
# computations for the forward kinematics in the tangent-configuration space
print(plant)
Ratfk = RationalForwardKinematics(plant)
print(plant)
# the point about which we will take the stereographic projections
q_star = np.zeros(3)

#compute limits in t-space
limits_t = []
for q in [q_low, q_high]:
    limits_t.append(Ratfk.ComputeSValue(np.array(q), q_star))

do_viz = True

# This line builds the visualization. Change the viz_role to Role.kIllustration if you
# want to see the plant with its illustrated geometry or to Role.kProximity
visualizer = IrisPlantVisualizer(plant, builder, scene_graph, viz_role=Role.kIllustration)
diagram = visualizer.diagram

# This line will run marching cubes to generate a mesh of the C-space obstacle
# Increase N to increase the resolution of the C-space obstacle.
visualizer.visualize_collision_constraint(N = 60)

#dummy classes to prevent mixups
class SPoint:
    def __init__(self, array:np.ndarray):
        self.ar = array
       

class QPoint:
    def __init__(self, array:np.ndarray):
        self.ar = array

q = np.array([1.7, 2.0, 1.7])