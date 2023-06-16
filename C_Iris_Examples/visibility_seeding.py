import numpy as np
from visibility_mis_solver import solve_max_independent_set_integer
from visibility_utils import shrink_regions
from time import strftime,gmtime
import pickle
from pydrake.all import HPolyhedron

class VisSeeder:
    def __init__(self,
                 N = 400, #computational cost per iteration
                 alpha = 0.05, #bernoulli test confidence
                 eps = 0.05, #bernoulli test uncovered space threshold
                 max_iterations = 10,
                 sample_cfree = None,
                 build_vgraph = None,
                 iris_w_obstacles = None,
                 verbose = False,
                 logger = None
                 ):
        
        self.logger = logger
        if self.logger is not None: self.logger.time()
        self.vb = verbose
        self.sample_cfree = sample_cfree
        self.build_vgraph = build_vgraph
        self.iris_w_obstacles = iris_w_obstacles
        self.N = N
        self.alpha = alpha
        self.eps = eps
        self.maxit = max_iterations
        self.M = int(np.log(1-(1-alpha)**(1/N))/np.log((1-eps)) + 0.5)
        if self.vb: 
            print(strftime("[%H:%M:%S] ", gmtime()) +'[VisSeeder] Point Insertion attempts M:', str(self.M))
            print(strftime("[%H:%M:%S] ", gmtime()) +f"[VisSeeder] {1-self.alpha:.2f} probability that unseen region is less than {100*eps:.1f} '%' of Cfree ")
        
        self.vgraph_points = []
        self.vgraph_admat = []
        self.seed_points = []
        self.regions = []
        self.region_groups = []

    def run(self):
        done = False
        it = 0
        while it<self.maxit:

            #sample N points in cfree
            points, b_test_is_full = self.sample_cfree(self.N, self.M, self.regions)
            self.vgraph_points.append(points)
            if b_test_is_full:
                if self.vb : print(strftime("[%H:%M:%S] ", gmtime()) +'[VisSeeder] Bernoulli test failed')
                done = True 
                break
            if self.logger is not None: self.logger.time()

            #build visibility graph
            self.sregs = shrink_regions(self.regions, offset_fraction=0.25)
            ad_mat = self.build_vgraph(points, self.sregs)
            self.vgraph_admat.append(ad_mat)
            if self.logger is not None: self.logger.time()

            #solve MHS
            _, mhs_idx = solve_max_independent_set_integer(ad_mat)
            self.seed_points +=[points[mhs_idx, :].squeeze()]
            if self.vb : print(strftime("[%H:%M:%S] ", gmtime()) +'[VisSeeder] Found ', len(mhs_idx), ' hidden points')
            if self.logger is not None: self.logger.time()

            #grow the regions with obstacles
            regions_step = self.iris_w_obstacles(points[mhs_idx, :].squeeze(), self.sregs, self.logger)
            self.regions += regions_step
            self.region_groups.append(regions_step)
            if self.logger is not None: self.logger.time()
            if self.logger is not None: self.logger.log(self, it)

            it+=1
        return self.regions
    
    def save_state(self, path):
        region_groups_A = [[r.A() for r in g] for g in self.region_groups] 
        region_groups_b = [[r.b() for r in g] for g in self.region_groups]
        data = {
        'vg':self.vgraph_points,
        'vad':self.vgraph_admat,
        'sp':self.seed_points,
        'ra':region_groups_A,
        'rb':region_groups_b}
        with open(path+".pkl", 'wb') as f:
            pickle.dump(data,f)

    def load_state(self, path):
        with open(path,'rb') as f:
            data = pickle.load(f)

        self.region_groups = [[HPolyhedron(a,b) for a,b in zip(ga, gb)] for ga, gb in zip(data['ra'], data['rb'])]    
        self.regions = []
        for g in self.region_groups:
            for r in g:
                self.regions.append(r)
        self.vgraph_points = data['vg']
        self.vgraph_admat = data['vad']
        self.seed_points = data['sp']

