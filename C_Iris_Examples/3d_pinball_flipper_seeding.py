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
from visibility_utils import point_in_regions, point_near_regions
from pydrake.all import MosekSolver, SolverOptions, CommonSolverOption
# The object we will use to perform our certification.
from pydrake.geometry.optimization_dev import (CspaceFreePolytope,
                                               CIrisSeparatingPlane,
                                               SeparatingPlaneOrder)
from pydrake.geometry.optimization import ComputeVisibilityGraph

from tqdm import tqdm
from scipy.sparse import lil_matrix
from visibility_seeding import VisSeeder
from visibility_logging import Logger


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



# filter fused joints self collisions so they don't interfere with collision engine
digaram = visualizer.diagram
context = visualizer.diagram_context
plant_context = plant.GetMyMutableContextFromRoot(context)
sg_context = scene_graph.GetMyContextFromRoot(context)
inspector = scene_graph.model_inspector()

pairs = scene_graph.get_query_output_port().Eval(sg_context).inspector().GetCollisionCandidates()

gids = [gid for gid in inspector.GetGeometryIds(GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity)]
get_name_of_gid = lambda gid : inspector.GetName(gid)
gids.sort(key=get_name_of_gid)
iiwa_oneDOF_gids = [gid for gid in gids if "iiwa7_oneDOF::" in get_name_of_gid(gid)]
iiwa_twoDOF_gids = [gid for gid in gids if "iiwa7_twoDOF::" in get_name_of_gid(gid)]

oneDOF_fused_col_geom = iiwa_oneDOF_gids[2:]
iiwa_oneDOF_fused_set = GeometrySet(oneDOF_fused_col_geom)
twoDOF_fused_col_geom = iiwa_twoDOF_gids[4:]
iiwa_twoDOF_fused_set = GeometrySet(twoDOF_fused_col_geom)
scene_graph.collision_filter_manager()\
            .Apply(CollisionFilterDeclaration().ExcludeWithin(iiwa_oneDOF_fused_set))
scene_graph.collision_filter_manager()\
            .Apply(CollisionFilterDeclaration().ExcludeWithin(iiwa_twoDOF_fused_set))


domain = HPolyhedron.MakeBox( Ratfk.ComputeSValue(q_low, np.zeros((3,))), Ratfk.ComputeSValue(q_high, np.zeros((3,))))
snopt_iris_options = IrisOptions()
snopt_iris_options.require_sample_point_is_contained = True
snopt_iris_options.iteration_limit = 20
snopt_iris_options.configuration_space_margin = 1e-4
#snopt_iris_options.max_faces_per_collision_pair = 60
snopt_iris_options.termination_threshold = -1
#snopt_iris_options.q_star = np.zeros(3)
snopt_iris_options.relative_termination_threshold = 0.05



cspace_free_polytope = CspaceFreePolytope(plant, scene_graph,
                                          SeparatingPlaneOrder.kAffine, q_star)
solver_id = MosekSolver.id()

# set up the certifier and the options for different search techniques
solver_options = SolverOptions()
# set this to 1 if you would like to see the solver output in terminal.
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 0)

# The options for when we search for new planes and positivity certificates given the polytopes
find_separation_certificate_given_polytope_options = CspaceFreePolytope.FindSeparationCertificateGivenPolytopeOptions()
find_separation_certificate_given_polytope_options.num_threads = -1
find_separation_certificate_given_polytope_options.verbose = False
find_separation_certificate_given_polytope_options.solver_options = solver_options
find_separation_certificate_given_polytope_options.ignore_redundant_C = False
find_separation_certificate_given_polytope_options.solver_id = solver_id

# The options for when we search for a new polytope given positivity certificates.
find_polytope_given_lagrangian_option = CspaceFreePolytope.FindPolytopeGivenLagrangianOptions()
find_polytope_given_lagrangian_option.solver_options = solver_options
find_polytope_given_lagrangian_option.ellipsoid_margin_cost = CspaceFreePolytope.EllipsoidMarginCost.kGeometricMean
find_polytope_given_lagrangian_option.search_s_bounds_lagrangians = True
find_polytope_given_lagrangian_option.ellipsoid_margin_epsilon = 1e-4
find_polytope_given_lagrangian_option.solver_id = solver_id

# bilinear_alternation_options = CspaceFreePolytope.BilinearAlternationOptions()
# bilinear_alternation_options.max_iter = 10
# bilinear_alternation_options.convergence_tol = 1e-3
# bilinear_alternation_options.find_polytope_options = find_polytope_given_lagrangian_option
# bilinear_alternation_options.find_lagrangian_options = find_separation_certificate_given_polytope_options

binary_search_options = CspaceFreePolytope.BinarySearchOptions()
binary_search_options.find_lagrangian_options = find_separation_certificate_given_polytope_options
binary_search_options.scale_min = 1e-3
binary_search_options.scale_max = 1.1
binary_search_options.max_iter = 6


q_min = np.array(q_low)
q_max = np.array(q_high)
q_diff = q_max-q_min

alpha = 0.05
eps = 0.1
epsilon_sample = 0.1


for seed in [3, 4, 5]:
    for N in [1, 40, 400]:

        np.random.seed(seed)
        def estimate_coverage(regions, pts = 5000):
            pts_ = [sample_cfree_SPoint(epsilon=epsilon_sample) for _ in range(pts)]
            inreg = 0
            for pt in pts_:
                if point_in_regions(pt, regions): inreg+=1
            return inreg/pts

        def sample_cfree_SPoint(MAXIT=100, epsilon = 0.1):
            it = 0
            while it<MAXIT:
                rand = np.random.rand(3)
                q_s = q_min + rand*q_diff
                col = False
                if not visualizer.col_func_handle(q_s):
                    for _ in range(10):
                        r  = 2*epsilon*(np.random.rand(3)-0.5)
                        col |= (visualizer.col_func_handle(q_s+r) > 0)
                    if not col:
                        return Ratfk.ComputeSValue(q_s, q_star)
                else:
                    it+=1
            raise ValueError("no col free point found")

        def sample_cfree_QPoint_in_regions(regions):
            for _ in range(1000):
                pt = sample_cfree_SPoint()
                if point_in_regions(pt, regions): return Ratfk.ComputeQValue(pt, q_star)
            return None

        def sample_cfree_handle(n, m, regions=None, epsilon=epsilon_sample):
            points = np.zeros((n,3))
            if regions is None: regions = []		
            for i in range(n):
                bt_tries = 0
                while bt_tries<m:
                    point = sample_cfree_SPoint(MAXIT=1000, epsilon = epsilon)
                    if point_near_regions(point, regions, tries = 100, eps = 0.1*epsilon):
                        bt_tries+=1
                    else:
                        break
                if bt_tries == m:
                    return points, True
                points[i] = point
            return points, False

        def vis(t1, t2, regions, num_checks, visualizer):
            t1flat = t1.reshape(-1)
            t2flat = t2.reshape(-1)
            if np.linalg.norm(t1-t2) < 1e-6:
                return (1-visualizer.col_func_handle(Ratfk.ComputeQValue(t1flat, np.zeros(3))))>0
                        
            tvec = np.linspace(0,1, num_checks)
            for t in tvec:
                tinterp = t1flat*t + (1-t)*t2flat
                if visualizer.col_func_handle(Ratfk.ComputeQValue(tinterp, np.zeros(3))):
                    return False
                elif point_in_regions(tinterp, regions):
                    return False
            else:
                return True
            
        visibility_checker = partial(vis, num_checks = 40, visualizer = visualizer)

        def vgraph_builder(points, regions):
            n = len(points)
            adj_mat = lil_matrix((n,n))
            for i in tqdm(range(n)):
                point = points[i, :]
                for j in range(len(points[:i])):
                    other = points[j]
                    result = visibility_checker(point, other, regions)
                    #print(result)
                    if result:
                        adj_mat[i,j] = adj_mat[j,i] = 1
            return adj_mat#.toarray()


        def SNOPT_IRIS(s_seeds,  region_obstacles, logger,  old_regions, plant, context, snoptiris_options, Ratforwardkin, coverage_threshold):
            regions = []
            for s_seed in s_seeds:
                q_seed = Ratforwardkin.ComputeQValue(s_seed.reshape(-1,1), np.zeros((3,1)))
                #print('snopt iris call')
                snoptiris_options.configuration_obstacles = []
                if len(region_obstacles):
                    snopt_iris_options.configuration_obstacles = region_obstacles
                plant.SetPositions(plant.GetMyMutableContextFromRoot(context), q_seed.reshape(-1,1))
                try:
                    r = IrisInRationalConfigurationSpace(plant, plant.GetMyContextFromRoot(context), q_star, snoptiris_options)
                    #run the certifier
                    cert = cspace_free_polytope.BinarySearch(set(),
                                                    r.A(),
                                                    r.b(), 
                                                    np.array(s_seed),
                                                    binary_search_options)
                    regions.append(cert.certified_polytope)
                    if logger is not None: logger.log_region(cert.certified_polytope)
                    curr_cov = estimate_coverage(regions+old_regions)
                    if curr_cov>= coverage_threshold:
                        return regions, True
                except:
                    print("error, SNOPT IRIS FAILED")
            return regions, False

        SNOPT_IRIS_Handle = partial(SNOPT_IRIS,
                                    plant = plant,
                                    context = visualizer.diagram_context,
                                    snoptiris_options = snopt_iris_options,
                                    Ratforwardkin = Ratfk,
                                    coverage_threshold = 1-eps
                                    )



        logger = Logger("3DOf_pinball", seed, N, alpha, eps, estimate_coverage)
        VS = VisSeeder(N = N,
                    alpha = alpha,
                    eps = eps,
                    max_iterations = 300,
                    sample_cfree = sample_cfree_handle,
                    build_vgraph = vgraph_builder,
                    iris_w_obstacles = SNOPT_IRIS_Handle,
                    verbose = True,
                    logger = logger
                    )
        out = VS.run()