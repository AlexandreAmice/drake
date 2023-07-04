import numpy as np
from functools import partial
# import ipywidgets as widgets
# from IPython.display import display
#pydrake imports
from pydrake.all import RationalForwardKinematics
from pydrake.geometry.optimization import IrisOptions, IrisInRationalConfigurationSpace, HPolyhedron, Hyperellipsoid
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions
from pydrake.all import PiecewisePolynomial, InverseKinematics, Sphere, Rgba, RigidTransform, RotationMatrix, IrisInConfigurationSpace
import time
import pydrake
from ur3e_demo import UrDiagram, SetDiffuse
import ur3e_demo
import visualization_utils as viz_utils
#setup visibility seeding helpers
from visibility_utils import point_in_regions, point_near_regions
from tqdm import tqdm
from scipy.sparse import lil_matrix
from visibility_seeding import VisSeeder
from doublegreedyhiddenset import DoubleGreedySeeding
from visibility_logging import Logger
import os

alpha = 0.05
eps = 0.15
N = 5000
seed = 1


np.random.seed(seed)

snopt_iris_options = IrisOptions()
snopt_iris_options.require_sample_point_is_contained = True
snopt_iris_options.iteration_limit = 8
snopt_iris_options.configuration_space_margin = 1.8e-3
#snopt_iris_options.max_faces_per_collision_pair = 60
snopt_iris_options.termination_threshold = -1
#snopt_iris_options.q_star = np.zeros(3)
snopt_iris_options.num_collision_infeasible_samples = 19
snopt_iris_options.relative_termination_threshold = 0.02


poi = []
poi.append(np.array([-0.32743, -0.92743,  0.47257,  0.07257, -0.02743]))
poi.append(np.array([-1.62743, -1.32743,  2.57257, -1.22743, -0.02743]))
poi.append(np.array([-1.62743, -2.02743,  1.67257,  0.37257, -0.02743]))
poi.append(np.array([-1.72743, -1.82743, -2.02743,  0.67257, -0.02743]))
poi.append(np.array([-1.72743, -1.92743,  1.07257, -2.32743, -0.02743]))
poi.append(np.array([-1.72743, -1.32743, -0.92743, -1.02743, -0.02743]))
poi.append(np.array([-1.52743, -2.42743,  0.87257, -1.62743, -0.02743]))
poi.append(np.array([-2.12743, -0.52743, -1.92743,  2.47257, -1.02743]))


ur = UrDiagram(num_ur = 1, weld_wrist = True, add_shelf = False,
				add_gripper = True)
diagram_context = ur.diagram.CreateDefaultContext()
diagram = ur.diagram.ForcedPublish(diagram_context)

plant_context = ur.plant.GetMyMutableContextFromRoot(
		diagram_context)
scene_graph_context = ur.scene_graph.GetMyMutableContextFromRoot(
	diagram_context)
inspector = ur.scene_graph.model_inspector()        

q = np.zeros(ur.plant.num_positions())
ik = InverseKinematics(ur.plant, plant_context)
collision_constraint = ik.AddMinimumDistanceConstraint(0.001, 0.001)

def eval_cons(q, c, tol):
	return 1-1*float(c.evaluator().CheckSatisfied(q.reshape(-1,1), tol))
	
col_func_handle = partial(eval_cons, c=collision_constraint, tol=0.01)

scaler = 1 #np.array([0.8, 1., 0.8, 1, 0.8, 1, 0.8]) #do you even geometry bro?
q_min = np.array(ur.plant.GetPositionLowerLimits())*scaler
q_max = np.array(ur.plant.GetPositionUpperLimits())*scaler
q_diff = q_max-q_min

def sample_cfree_QPoint(MAXIT=1500, epsilon_sample = 0.05):
	it = 0
	while it<MAXIT:
		rand = np.random.rand(5)
		q_s = q_min + rand*q_diff
		col = False
		for _ in range(60):
			r  = 2*epsilon_sample*(np.random.rand(5)-0.5)
			col |= (col_func_handle(q_s+r) > 0)
			if col: break
		if not col:
			return q_s #Ratfk.ComputeQValue(q_s, q_star)
		it+=1
	raise ValueError("no col free point found")

def estimate_coverage(regions, pts = 7500):
	pts_ = [sample_cfree_QPoint() for _ in range(pts)]
	inreg = 0
	for pt in pts_:
		if point_in_regions(pt, regions): inreg+=1
	return inreg/pts

def SNOPT_IRIS(q_seeds,  region_obstacles, logger, old_regions, plant, context, snoptiris_options, coverage_threshold):
	regions = []
	for reg_indx, q_seed in enumerate(q_seeds):
		#q_seed = Ratforwardkin.ComputeQValue(s_seed.reshape(-1,1), np.zeros((7,1)))
		#print('snopt iris call')
		snoptiris_options.configuration_obstacles = []
		if len(region_obstacles):
			snoptiris_options.configuration_obstacles = region_obstacles
		plant.SetPositions(plant.GetMyMutableContextFromRoot(context), q_seed.reshape(-1,1))
		try:
			#r = IrisInRationalConfigurationSpace(plant, plant.GetMyContextFromRoot(context), q_star, snoptiris_options)
			r = IrisInConfigurationSpace(plant, plant_context, snoptiris_options)
			r_red = r.ReduceInequalities()
			print(f"[SNOPT IRIS]: Region:{reg_indx} / {len(q_seeds)}")
			#run the certifier
			# cert = cspace_free_polytope.BinarySearch(set(),
			# 								r.A(),
			# 								r.b(), 
			# 								np.array(s_seed),
			# 								binary_search_options)
			if logger is not None: logger.log_region(r_red)
			# r = cert.certified_polytope
			regions.append(r_red)
			curr_cov = estimate_coverage(regions+old_regions)
			if curr_cov>= coverage_threshold:
				return regions, True
		except:
			print("error, SNOPT IRIS FAILED")
	return regions, False

SNOPT_IRIS_Handle = partial(SNOPT_IRIS,
							plant = ur.plant,
							context = diagram_context,
							snoptiris_options = snopt_iris_options,
							coverage_threshold = 1-eps
							#binary_search_options = binary_search_options,
							#Ratforwardkin = Ratfk,
							)

def sample_cfree_handle(n, m, regions=None, epsilon_sample = 0.05):
	points = np.zeros((n,5))
	if regions is None: regions = []		
	for i in range(n):
		bt_tries = 0
		while bt_tries<m:
			point = sample_cfree_QPoint(epsilon_sample=epsilon_sample)
			col = point_near_regions(point, regions, tries=100, eps = 0.05*epsilon_sample)
			if col:
				bt_tries += 1
				if bt_tries == m:
					return points, True 
			else:
				break
		points[i] = point
	return points, False

def vis(q1, q2, regions, num_checks):
	q1flat = q1.reshape(-1)
	q2flat = q2.reshape(-1)
	if np.linalg.norm(q1-q2) < 1e-6:
		return (1-col_func_handle(q1))>0
				
	tvec = np.linspace(0,1, num_checks)
	for t in tvec:
		qinterp = q1flat*t + (1-t)*q2flat
		if col_func_handle(qinterp.reshape(-1,1)):
			return False
		elif point_in_regions(qinterp, regions):
			return False
	else:
		return True
	
visibility_checker = partial(vis, num_checks = 80)

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
	return adj_mat


# from doublegreedyhiddenset import HiddensetDoubleGreedy

# dg = HiddensetDoubleGreedy(
#     alpha=alpha,
#     eps = 0.001,
#     max_samples = 4000,
#     sample_node_handle=sample_cfree_handle,
#     los_handle=visibility_checker,
#     verbose=True
# )
# dg.construct_independent_set([])
# hidden_set = dg.refine_independent_set_greedy([])

# for p1 in hidden_set:
# 	for p2 in hidden_set:
# 		if p1.squeeze()[0] != p2.squeeze()[0]:
# 			if visibility_checker(p1, p2, []):
# 				print("ERROR")
# print('d')
logger = Logger("5DOf_ur_DG_", seed, N, alpha, eps, estimate_coverage)
VS = DoubleGreedySeeding(N = N,
			alpha = alpha,
			eps = eps,
			max_iterations = 10,
			sample_cfree = sample_cfree_handle,
			los_handle = visibility_checker,
			iris_w_obstacles = SNOPT_IRIS_Handle,
			verbose = True,
			logger = logger
			)
out = VS.run()