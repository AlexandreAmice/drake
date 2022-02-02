from operator import pos
import numpy as np
from numpy.core.fromnumeric import argmin
from numpy.random.mtrand import seed
from pydrake.all import (MathematicalProgram, Variable, HPolyhedron, le, SnoptSolver, Solve) 

class IrisNode:
    def __init__(self, region, ellipse, parent = None):
        self.ellipse = ellipse
        self.region = region
        self.parent = parent
        self.children = []
        self.id = None

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm 

class RRTIRIS:
    def __init__(self, 
                 start, 
                 goal, 
                 limits, 
                 default_iris_domain,
                 iris_handle,
                 offset_scaler,
                 init_goal_sample_rate = 0.05,
                 goal_sample_rate_scaler = 0.1,
                 verbose = False,
                 plotcallback = None,
                 ):

        self.dim = len(start)
        self.iris_handle = iris_handle
        self.default_iris_domain = default_iris_domain

        self.start = start
        self.goal = goal
        self.limits = limits
        self.min_pos = self.limits[0]
        self.max_pos = self.limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 
        self.offset_scaler = offset_scaler
        self.init_goal_sample_rate = init_goal_sample_rate
        self.goal_sample_rate = init_goal_sample_rate
        self.goal_sample_rate_scaler = goal_sample_rate_scaler
        self.verbose = verbose
        self.plotcallback = plotcallback
        self.do_plot = True if self.plotcallback is not None else False

        self.node_ellipses = []
        self.node_centers = [] 
        self.node_volumes = []
        self.node_regions = []
        self.seed_points = [start]
        self.out_set_seed_points = []
        region, ellipse = self.generate_region(start, [])
        root = IrisNode(region, ellipse)
        root.id = 0
        self.nodes = [root]
        self.node_ellipses = [ellipse]
        self.node_centers = [self.nodes[0].ellipse.center()] 
        self.node_volumes = [self.nodes[0].ellipse.Volume()]
        self.node_regions = [region]

        if self.do_plot:
            self.plotcallback(region, start, start, len(self.nodes))
        
        self.distance_to_go = 1e9


    def sample_node_pos(self, collision_free = True, MAXIT = 1e4):
        rand = np.random.rand()
        if rand < self.goal_sample_rate:
            return self.goal
        
        rand = np.random.rand(self.dim)
        pos_samp = self.min_pos + rand*self.min_max_diff 

        good_sample = not self.in_regions(pos_samp)

        it = 0
        while not good_sample and it < MAXIT:
            rand = np.random.rand(self.dim)
            pos_samp = self.min_pos + rand*self.min_max_diff 
            good_sample = not self.in_regions(pos_samp)
            it+=1

        if not good_sample:
            print("[RRT ERROR] Could not find collision free point in MAXIT")
            raise NotImplementedError
        return pos_samp

    def in_regions(self,sample):
        in_reg = False
        for reg in self.node_regions:
            in_reg |= reg.PointInSet(sample)
        return in_reg
            
    def check_region(self, region, ellipse):
        #use ellipses to check regions for similarity
        #compare center and volume
        #vol = ellipse.Volume()
        #center_origin = np.linalg.norm(ellipse.center())
        #min_vol_diff = np.min(np.abs(np.array(self.node_volumes) - vol))
        #if min_vol_diff< 0.001 or center_origin<1e-7:
        #    return False
        #else:
        #    return True
        return True

    def run(self, n_it):
        for it in range(n_it):
            #print(it)
            pos_samp = self.sample_node_pos()
            seed_point, min_dist, closest_points, nearest_id = self.get_closest_point_in_regions(pos_samp) 
            if self.verbose:
                print('[RRT IRIS] Pos_samp', pos_samp, ' seed ', seed_point, ' dist ', min_dist)
            self.seed_points.append(seed_point)
            self.out_set_seed_points.append(pos_samp)
            
            parent_node = self.nodes[nearest_id]
            region, ellipse = self.generate_region(seed_point, closest_points)
            if self.check_region(region, ellipse):
                child_node = IrisNode(region, ellipse)
                child_node.id = it + 1
                self.nodes.append(child_node)
                self.node_ellipses.append(ellipse)
                self.node_centers.append(ellipse.center())
                self.node_volumes.append(ellipse.Volume())
                self.node_regions.append(region)
                child_node.parent = parent_node
                
                _, dist_to_target, _, _ = self.get_closest_point_in_regions(self.goal)

                print(dist_to_target) 
                if self.do_plot:
                    self.plotcallback(region, seed_point, pos_samp, len(self.nodes))   
            else:
                print('[RRT IRIS] Region check failed')
            

            if dist_to_target<self.distance_to_go:
                self.distance_to_go = dist_to_target
                self.closest_id = child_node.id
                self.goal_sample_rate = np.clip(0.5 - self.distance_to_go*self.goal_sample_rate_scaler, a_min = self.init_goal_sample_rate, a_max = 1)
                if self.verbose:
                    print("[RRT IRIS] it: {iter} distance to target: {dist: .3f} goalsample prob: {prob: .3f}".format(iter =it, dist = self.distance_to_go, prob = self.goal_sample_rate))

            if self.distance_to_go <= 1e-4:
                return True, self.node_regions, self.node_ellipses

        return False, self.node_regions, self.node_ellipses

    def generate_region(self, seed_point, closest_points_in_regions):
        #compute polytope cuts of existing regions
        A = []
        B = []
        for ellipse, closest_point in zip(self.node_ellipses, closest_points_in_regions):
            a, b = self.compute_cut(closest_point, ellipse)
            A.append(a)
            B.append(b)

        A = np.array(A)
        B = np.array(B)
        if len(A):
            domain = HPolyhedron(A = np.vstack((self.default_iris_domain.A(), A)), b = np.hstack((self.default_iris_domain.b(), B)))
        else:
            domain = self.default_iris_domain
        poly = self.iris_handle(seed_point, domain)
        return poly, poly.MaximumVolumeInscribedEllipsoid()

    def compute_cut(self, seed, ellipse):
        center = ellipse.center()
        vec = seed - center 
        a = normalize(vec)
        b = a@(center + self.offset_scaler*(seed - center))
        return -a, -b

    def get_closest_point_in_regions(self, sample):
        num_regions = len(self.nodes) 
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.dim*num_regions, 'x')
        prog.SetInitialGuess(x, np.array(self.node_centers).reshape(-1,) )  
        cost = x - np.tile(sample, (1, num_regions)).squeeze()
        cost = cost@cost.T
        prog.AddCost(cost)
        for idx in range(num_regions):
            A = self.nodes[idx].region.A()
            b = self.nodes[idx].region.b()
            prog.AddConstraint(le(A@x[idx*self.dim:(idx + 1)*self.dim], b))
        solver = SnoptSolver()
        result = solver.Solve(prog)
        x_sol = result.GetSolution(x).reshape(num_regions, self.dim)
        dists = np.linalg.norm(x_sol-sample, axis = 1)
        closest = np.argmin(dists)
        min_dist = dists[closest]
        min_point = x_sol[closest, :]
        return min_point, min_dist, x_sol, closest