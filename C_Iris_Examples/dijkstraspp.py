from scipy.sparse.csgraph import dijkstra
from scipy.sparse import coo_matrix
import numpy as np
from pydrake.all import (MathematicalProgram, Variable, HPolyhedron, le, SnoptSolver, Solve, eq) 

class node:
    def __init__(self, loc, regs = None):
        self.loc = loc
        if regs is None:
            self.regions = []
        else:
            self.regions = regs

class DijkstraSPPsolver:
    def __init__(self,
                 regions, 
                 point_to_region_space_conversion,
                 verbose = True) :
        self.regions = regions
        self.verbose = verbose
        self.base_ad_mat, self.node_intersections = self.build_base_adjacency_matrix()
        self.point_conversion = point_to_region_space_conversion
        

    def solve(self, 
              start_q,
              target_q,
              refine_path = True):
        ad_mat = self.extend_adjacency_mat(start_q, target_q)
        if ad_mat is not None:
            wps, dist = self.dijkstra_in_configspace(adj_mat=ad_mat)
            if dist<0:
                print('[DijkstraSPP] Points not reachable')
                return [], -1
            if refine_path:
                location_wps_optimized_t, dist_optimized = self.refine_path_SOCP(wps, 
                                                                        self.point_conversion(start_q), 
                                                                        self.point_conversion(target_q), 
                                                                        )
                return location_wps_optimized_t, dist_optimized
            else:
                intermediate_nodes = [self.node_intersections[idx].loc for idx in wps[1:-1]]
                waypoints = [self.point_conversion(start_q)] + intermediate_nodes + [self.point_conversion(target_q)]
                return waypoints, dist
        else:
            print('[DijkstraSPP] Points not in regions')
            return [], -1

    def build_base_adjacency_matrix(self):    
        nodes_intersections = []
        for idx, r in enumerate(self.regions[:-1]):
            if (idx%10) == 0:
                if self.verbose: print('[DijkstraSPP] Pre-Building adjacency matrix ', idx,'/', len(self.regions))
            for r2 in self.regions[idx+1:]:
                if r.IntersectsWith(r2):
                    try:
                        loc = r.Intersection(r2).ChebyshevCenter() #MaximumVolumeInscribedEllipsoid().center()
                        nodes_intersections.append(node(loc, [r, r2]))
                    except:
                        if self.verbose: print('[DijkstraSPP] Failed ellispe prog', idx)

        node_locations = [node.loc for node in nodes_intersections]
        adjacency_list = []
        adjacency_dist = []
        for n in nodes_intersections:
            edges = []
            edge_dist = []
            for n2 in nodes_intersections:
                if nodes_intersections.index(n) != nodes_intersections.index(n2):
                    n_regs = set([str(r) for r in n.regions])
                    n2_regs = set([str(r) for r in n2.regions])
                    if len(list(n_regs&n2_regs)):
                        edges.append(nodes_intersections.index(n2))
                        edge_dist.append(np.linalg.norm(n.loc-n2.loc))
            adjacency_list.append(edges)
            adjacency_dist.append(edge_dist)
        
        N = len(node_locations)
        data = []
        rows = []
        cols = []

        ad_mat = coo_matrix((N, N), np.float32)

        for idx in range(N):
            nei_idx = 0
            for nei in adjacency_list[idx]:
                if not nei == idx:
                    data.append(adjacency_dist[idx][nei_idx])
                    rows.append(idx)
                    cols.append(nei)
                nei_idx += 1

        ad_mat = coo_matrix((data, (rows, cols)), shape=(N, N))
        return ad_mat, nodes_intersections

    def dijkstra_in_configspace(self, adj_mat):
        # convention for start and target: source point is second to last and target is last point
        src = adj_mat.shape[0] -2
        target = adj_mat.shape[0] -1
        dist, pred = dijkstra(adj_mat, directed=False, indices=src, return_predecessors=True)
        #print(f'{len(np.argwhere(pred == -9999))} disconnected nodes'), #np.argwhere(pred == -9999))
        idxs = (pred == -9999)
        pred[idxs] = -1000000
        dist[idxs] = -1000000
        sp_list = []
        sp_length = dist[target]
        if sp_length<0:
            return [], sp_length
        current_idx = target
        sp_list.append(current_idx)
        while not current_idx == adj_mat.shape[0] - 2:
            current_idx = pred[current_idx]
            sp_list.append(current_idx)
            if current_idx==src: break
        return [idx for idx in sp_list[::-1]], sp_length

    def extend_adjacency_mat(self, start_q, target_q):
        #first check point memberships
        start_idx = []
        target_idx = []
        start_conv = self.point_conversion(start_q)
        target_conv = self.point_conversion(target_q)
        for idx, r in enumerate(self.regions):
            if r.PointInSet(start_conv):
                start_idx.append(idx)
            if r.PointInSet(target_conv):
                target_idx.append(idx)
        if len(start_idx)==0 or len(target_idx)==0:
            print('[DijkstraSPP] Points not in set, idxs', start_idx,', ', target_idx)
            return None
        N = len(self.node_intersections) + 2
        data = list(self.base_ad_mat.data)
        rows = list(self.base_ad_mat.row)
        cols = list(self.base_ad_mat.col)
        #get idstances of all nodes  
        start_regions = [self.regions[idx] for idx in start_idx]
        target_regions = [self.regions[idx] for idx in target_idx]
        start_adj_idx = N-2
        target_adj_idx = N-1
        for node_idx, node in enumerate(self.node_intersections):
            if len(list(set(start_regions) & set(node.regions))):
                dist = np.linalg.norm(start_conv-node.loc)
                data.append(dist)
                rows.append(start_adj_idx)
                cols.append(node_idx)
                data.append(dist)
                rows.append(node_idx)
                cols.append(start_adj_idx)
            if len(list(set(target_regions) & set(node.regions))):
                dist = np.linalg.norm(target_conv-node.loc)
                data.append(dist)
                rows.append(target_adj_idx)
                cols.append(node_idx)
                data.append(dist)
                rows.append(node_idx)
                cols.append(target_adj_idx)
        if len(list(set(start_regions) & set(target_regions))):
            dist = np.linalg.norm(target_conv-start_conv)
            data.append(dist)
            rows.append(target_adj_idx)
            cols.append(start_adj_idx)
            data.append(dist)
            rows.append(start_adj_idx)
            cols.append(target_adj_idx)
        
        ad_mat_extend = coo_matrix((data, (rows, cols)), shape=(N, N))
        return ad_mat_extend

    # def refine_path_QP(self, wps, start_t, target_t):
    #     intermediate_nodes = [self.node_intersections[idx] for idx in wps[1:-1]]
    #     dim = len(self.node_intersections[0].loc)
    #     prog = MathematicalProgram()
    #     intermediates = []
    #     for idx, wpnode in enumerate(intermediate_nodes):
    #         x = prog.NewContinuousVariables(dim, 'x'+str(idx))
    #         intermediates.append(x)
    #         prog.SetInitialGuess(x, wpnode.loc)
    #         for r in wpnode.regions:
    #             prog.AddConstraint(le(r.A()@x, r.b())) 

    #     cost = 0
    #     prev = start_t
    #     for pt in intermediates + [target_t]:
    #         a = (prev-pt) #* np.array([4.0,3.5,3,2.5,2,2.5,1]) 
    #         cost += a.T@a
    #         prev = pt
    #     prog.AddCost(cost)

    #     res = Solve(prog)
    #     if res.is_success():
    #         path = [start_t]
    #         for i in intermediates:
    #             path.append(res.GetSolution(i))
    #         path.append(target_t)
    #         wps_start = [self.node_intersections[idx].loc for idx in wps[1:-1]]
    #         dist_start = 0
    #         prev = start_t
    #         for wp in wps_start + [target_t]:
    #             #dist_start += np.linalg.norm()#* np.array([4.0,3.5,3,2.5,2,2.5,1])
    #             a = prev-wp
    #             dist_start += a.T@a
    #             prev = wp
    #         if self.verbose: print("[DijkstraSPP] optimized distance/ start-distance = {opt:.2f} / {start:.2f} = {res:.2f}".format(opt = res.get_optimal_cost(), start = dist_start, res = res.get_optimal_cost()/dist_start))
    #         return path, res.get_optimal_cost()
    #     else:
    #         print("[DijkstraSPP] Refine path QP failed")
    #         return None, None
        
    def refine_path_SOCP(self, wps, start_t, target_t):
            intermediate_nodes = [self.node_intersections[idx] for idx in wps[1:-1]]
            dim = len(self.node_intersections[0].loc)
            prog = MathematicalProgram()
            intermediates = []
            for idx, wpnode in enumerate(intermediate_nodes):
                x = prog.NewContinuousVariables(dim, 'x'+str(idx))
                intermediates.append(x)
                prog.SetInitialGuess(x, wpnode.loc)
                for r in wpnode.regions:
                    prog.AddConstraint(le(r.A()@x, r.b())) 

            prev = start_t
            cost = 0 
            for idx in range(len(intermediate_nodes)+1):
                t = prog.NewContinuousVariables(dim+1, 't'+str(idx))
                prog.AddConstraint(eq(t[1:], prev-(intermediates + [target_t])[idx]))
                prev = (intermediates + [target_t])[idx]
                prog.AddLorentzConeConstraint(t)
                cost += t[0]
    
            prog.AddCost(cost)

            res = Solve(prog)
            if res.is_success():
                path = [start_t]
                for i in intermediates:
                    path.append(res.GetSolution(i))
                path.append(target_t)
                wps_start = [self.node_intersections[idx].loc for idx in wps[1:-1]]
                dist_start = 0
                prev = start_t
                for wp in wps_start + [target_t]:
                    #dist_start += np.linalg.norm()#* np.array([4.0,3.5,3,2.5,2,2.5,1])
                    a = prev-wp
                    dist_start += np.sqrt(a.T@a)
                    prev = wp
                if self.verbose: print("[DijkstraSPP] optimized distance/ start-distance = {opt:.2f} / {start:.2f} = {res:.2f}".format(opt = res.get_optimal_cost(), start = dist_start, res = res.get_optimal_cost()/dist_start))
                return path, res.get_optimal_cost()
            else:
                print("[DijkstraSPP] Refine path SCOP failed")
                return None, None