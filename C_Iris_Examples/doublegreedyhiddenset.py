import numpy as np
import networkx as nx
from time import strftime, gmtime
from tqdm import tqdm
from scipy.sparse import lil_matrix
from visibility_mis_solver import solve_max_independent_set_integer
from visibility_utils import shrink_regions

class DoubleGreedySeeding:
    def __init__(self,
                 N = 400, #computational cost per iteration
                 alpha = 0.05, #bernoulli test confidence
                 eps = 0.05, #bernoulli test uncovered space threshold
                 max_iterations = 10,
                 sample_cfree = None,
                 los_handle = None,
                 iris_w_obstacles = None,
                 verbose = False,
                 logger = None,
                 terminate_on_iris_step = True
                 ):
        
        self.logger = logger
        self.terminate_on_iris_step = terminate_on_iris_step
        if self.logger is not None: self.logger.time()
        self.vb = verbose
        self.sample_cfree = sample_cfree
        self.los_handle = los_handle
        self.iris_w_obstacles = iris_w_obstacles
        self.N = N
        self.alpha = alpha
        self.eps = eps
        self.maxit = max_iterations

        
        #self.M = int(np.log(1-(1-alpha)**(1/N))/np.log((1-eps)) + 0.5)
        if self.vb: 
            #print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedySeeder] Point Insertion attempts M:', str(self.M))
            print(strftime("[%H:%M:%S] ", gmtime()) +f"[DoubleGreedySeeder] {1-self.alpha:.2f} probability that unseen region is less than {100*eps:.1f} '%' of Cfree ")
        
        
        self.vgraph_points = []
        self.vgraph_admat = []
        self.seed_points = []
        self.regions = []
        self.region_groups = []

    def run(self):
        done = False
        it = 0
        while it<self.maxit:
            #build partial vgraph
            self.dg = HiddensetDoubleGreedy(
                    alpha=self.alpha,
                    eps = 0.0001,
                    max_samples = self.N,
                    sample_node_handle=self.sample_cfree,
                    los_handle=self.los_handle,
                    verbose=self.vb
                    ) 
            
            self.sregs = shrink_regions(self.regions, offset_fraction=0.25)  
            self.dg.construct_independent_set(self.sregs)
            if self.logger is not None: self.logger.time()
            hidden_set = self.dg.refine_independent_set_greedy(self.sregs)
            hidden_set = np.array(hidden_set)
            if self.logger is not None: self.logger.time()
            # #sample N points in cfree
            # points, b_test_is_full = self.sample_cfree(self.N, self.M, self.regions)
            # self.vgraph_points.append(points)
            # if b_test_is_full:
            #     if self.vb : print(strftime("[%H:%M:%S] ", gmtime()) +'[VisSeeder] Bernoulli test failed')
            #     done = True 
            #     if self.logger is not None: self.logger.log_string(strftime("[%H:%M:%S] ", gmtime()) +'[VisSeeder] Bernoulli test failed')
            #     return self.regions
            # if self.logger is not None: self.logger.time()

            # #build visibility graph
            # ad_mat = self.build_vgraph(points, self.sregs)
            # self.vgraph_admat.append(ad_mat)
            # if self.logger is not None: self.logger.time()

            # #solve MHS
            # _, mhs_idx = solve_max_independent_set_integer(ad_mat)

            self.seed_points +=[hidden_set.squeeze()]
            if self.vb : print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedySeeder] Found ', len(hidden_set), ' hidden points')
            if self.logger is not None: self.logger.time()

            #grow the regions with obstacles
            regions_step, is_full_iris = self.iris_w_obstacles(hidden_set.squeeze().reshape(len(hidden_set),-1), self.sregs, self.logger, self.regions)
            self.regions += regions_step
            self.region_groups.append(regions_step)
            if self.logger is not None: self.logger.time()
            if self.logger is not None: self.logger.log(self, it)
            if is_full_iris and self.terminate_on_iris_step:
                print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedySeeder] Coverage met, terminated on Iris step')
                if self.logger is not None: self.logger.log_string(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedySeeder] Coverage met, terminated on Iris step')
                return self.regions
            it+=1
        if self.logger is not None: self.logger.log_string(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedySeeder] Maxit reached')
        return self.regions

class HiddensetDoubleGreedy:
    def __init__(self,
                    alpha = 0.05,
                    eps = 0.05,
                    max_samples = 500,
                    sample_node_handle = None,
                    los_handle= None,
                    verbose = False
                    ):
        np.random.seed(1)
        self.verbose = verbose
        self.alpha = alpha
        self.eps = eps
        self.max_samples = max_samples
        self.M = int(np.log(alpha)/np.log(1-eps))

        self.sample_node = sample_node_handle
        self.is_los = los_handle
        self.sample_set = {}
        self.hidden_set = []
        self.points = []
        if self.verbose: 
            print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Point insertion attempts M:', str(self.M))
            print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] {} probability that unseen region is less than {} "%" of Cfree '.format(1-self.alpha, 100*eps))

    def construct_independent_set(self, regions):
        it = 0
        while it < self.M:
            p, is_full = self.sample_node(1, self.M, regions)
            self.points.append(p)
            visible_points = []
            add_to_sample_set = False
            for idx_point in self.hidden_set:
                hidden_point = self.points[idx_point]
                if self.is_los(p, hidden_point, regions):
                    add_to_sample_set = True
                    visible_points.append(hidden_point)
            if add_to_sample_set:
                self.sample_set[str(p)] = [p, visible_points]
            else:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +"[DoubleGreedy] New hidden point placed N = ", str(len(self.hidden_set)), "it = ", it) 
                self.hidden_set.append(len(self.points)-1)
                it = 0
                    #update visibility 
                for s_key in self.sample_set.keys():
                    p_sample = self.sample_set[s_key][0]
                    if self.is_los(p, p_sample, regions):
                        self.sample_set[s_key][1].append(p)
            it+=1
            if len(self.points)>=self.max_samples:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Point budget exceeded', len(self.points))
                return  [self.points[ih] for ih in self.hidden_set]
        if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Sample set size',len(self.sample_set.keys()))

        return [self.points[ih] for ih in self.hidden_set]

    def compute_kernel_of_hidden_point(self, point):
        ker = []
        for sampdat in self.sample_set.values():
            pos = sampdat[0]
            if len(sampdat[1])==1 and np.array_equal(sampdat[1][0], self.points[point]):
                ker.append(pos)
        return ker

    def vgraph_builder(self, points, regions):
        n = len(points)
        adj_mat = lil_matrix((n,n))
        for i in tqdm(range(n)):
            point = points[i, :]
            for j in range(len(points[:i])):
                other = points[j]
                result = self.is_los(point, other, regions)
                #print(result)
                if result:
                    adj_mat[i,j] = adj_mat[j,i] = 1
        return adj_mat

    def get_new_set_candidates(self, point, regions):
        if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Computing Kernel', )
        kernel_points = self.compute_kernel_of_hidden_point(point)
        if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Kernel of size', len(kernel_points), 'found')
        kernel_points += [self.points[point]]
        if len(kernel_points)>1:
            kernel_points_arr = np.array(kernel_points).squeeze()
            adj_mat = self.vgraph_builder(kernel_points_arr, regions)
            cost, new_cands = solve_max_independent_set_integer(adj_mat)

            # graph = nx.Graph()
            #     for i1, v1 in enumerate(kernel_points):
            #         for i2, v2 in enumerate(kernel_points):
            #             if i1!=i2 and self.is_los(v1, v2, regions):
            #                 graph.add_edge(i1,i2)
            # if len(graph.edges):
            #     new_cands = nx.maximal_independent_set(graph)
            # else:
            #     raise ValueError("no edges detected, must be error")
            return [kernel_points[c] for c in new_cands]
        else:
            return [self.points[point]]

    def refine_independent_set_greedy(self, regions):
        continue_splitting = True
        while continue_splitting:
            candidate_splits = [self.get_new_set_candidates(p, regions) for p in self.hidden_set]
            best_split = max(candidate_splits, key = len)
            best_split_idx = candidate_splits.index(best_split)
            if len(best_split)>1:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Hidden point found to split into', len(best_split))
                hidden_point_old = self.hidden_set[best_split_idx]
                p_hidden_old  = self.points[hidden_point_old]
                for s_key in self.sample_set.keys():
                    vis_points = self.sample_set[s_key][1] 
                    eq_list = [np.array_equal(v, p_hidden_old) for v in vis_points]
                    idx_eq = eq_list.index(True) if True in eq_list else None
                    if idx_eq is not None:
                        vis_points.pop(idx_eq)
                    p_sample = self.sample_set[s_key][0]
                    for idnr, p_split in enumerate(best_split):
                        if self.is_los(p_split, p_sample, regions):
                            vis_points.append(p_split)
                #remove old hidden point from hidden set
                self.hidden_set.pop(best_split_idx)
                for c in best_split:
                    self.hidden_set.append([np.array_equal(v, c) for v in self.points].index(True))
            else:
                continue_splitting = False
        return [self.points[ih] for ih in self.hidden_set]