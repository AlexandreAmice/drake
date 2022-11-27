import numpy as np

#visibility PRM
class Node:
    def __init__(self, position, nodetype = '', connections = []):
        self.type = nodetype
        self.position = position
        self.connections = connections                

class SetNode:
    def __init__(self, position, region, nodetype = '', connections = []):
        self.type = nodetype
        self.region = region
        self.position = position
        self.connections = connections  

#Muller , Marsaglia (‘Normalised Gaussians’) approach to sampling surface of L2-ball
def sample_unit_ball(dim):
    u = np.random.normal(0,1,dim).reshape(2,1)  
    val = np.linalg.norm(u)
    return u/val

#warp sampling - clean up later
def sample_in_set(ell):
    B = np.linalg.inv(ell.A())
    center = ell.center().reshape(2,1)
    return B@sample_unit_ball(center.shape[0]) + center

class VPRM:
    def __init__(self,
                 limits = None,
                 M = 10,
                 collision_handle = None,
                 is_in_line_of_sight = None,
                 plot_node = None,
                 plot_edge = None,
                 Verbose = False):
        
        self.M = M
        self.los_handle = is_in_line_of_sight
        self.col_handle = collision_handle
        self.plot_node = plot_node
        self.plot_edge = plot_edge
        self.limits = limits
        self.min_pos = self.limits[0]
        self.max_pos = self.limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 
        self.dim = len(self.min_pos)
        self.Verbose = Verbose
        self.guard_domains = []
        self.guards = []
        self.connectors = []
    
    def build(self,):
        ntry = 0
        q = self.sample_node_pos()
        self.guards.append(Node(q, 'guard'))
        self.guard_domains.append([self.guards[-1]])
            
        while ntry < self.M:
            q = self.sample_node_pos()
            vis_guards = []
            vis_guard_domains = []
            break_flag = False
            for domain in self.guard_domains:
                is_found = False
                #break_flag = False
                for guard in domain:
                    #print(q.reshape(2,1))
                    #print(guard.position.reshape(2,1))
                    is_los = self.los_handle(q.reshape(2,1), guard.position.reshape(2,1))
                    if is_los[0]:
                        is_found = True
                        if len(vis_guards) == 0:
                            vis_guards.append(guard)
                            vis_guard_domains.append(self.guard_domains.index(domain))
                        else:
                            #is connector
                            vis_guards.append(guard)
                            self.connectors.append(Node(q, 'connector', connections = vis_guards))
                            print('connector found, ntry =', ntry)  if self.Verbose else None 
                            #merge visible domains
                            other_domain = self.guard_domains[vis_guard_domains[0]]
                            cur_domain_idx = self.guard_domains.index(domain)
                            self.guard_domains[vis_guard_domains[0]] = other_domain + domain
                            self.guard_domains = \
                            [dom for idx, dom in enumerate(self.guard_domains) if idx!= cur_domain_idx]
                            break_flag = True
                        break
                if break_flag:
                    break
            if len(vis_guards)==0:
                self.guards.append(Node(q, 'guard'))
                self.guard_domains.append([self.guards[-1]])
                print('guard found, ntry =', ntry) if self.Verbose else None
                ntry = 0 
            else:
                ntry +=1
        if self.plot_edge is not None:
            self.plot()
        
    def plot(self,):
        for g in self.guards:
            self.plot_node(g.position, is_guard = True)
        for c in self.connectors:
            self.plot_node(c.position, is_guard = False)
            for vis in c.connections:
                self.plot_edge(c.position, vis.position)
                
    def sample_node_pos(self, MAXIT = 1e4):  
        rand = np.random.rand(self.dim)
        pos_samp = self.min_pos + rand*self.min_max_diff 
        good_sample = not self.col_handle(pos_samp)

        it = 0
        while not good_sample and it < MAXIT:
            rand = np.random.rand(self.dim)
            pos_samp = self.min_pos + rand*self.min_max_diff 
            good_sample = not self.col_handle(pos_samp)
            it+=1

        if not good_sample:
            raise ValueError("[VPRM ERROR] Could not find collision free point in MAXIT")
        return pos_samp


class SetVPRM:
    def __init__(self,
                 limits = None,
                 M = 10,
                 collision_handle = None,
                 is_in_line_of_sight = None,
                 iris_handle = None,
                 connector_iris_handle = None,
                 plot_node = None,
                 plot_edge = None,
                 plot_region = None,
                 Verbose = False):
        
        self.M = M
        self.los_handle = is_in_line_of_sight
        self.col_handle = collision_handle
        self.iris_handle = iris_handle
        self.connector_iris_handle = connector_iris_handle
        self.plot_node = plot_node
        self.plot_edge = plot_edge
        self.plot_region = plot_region
        self.limits = limits
        self.min_pos = self.limits[0]
        self.max_pos = self.limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 
        self.dim = len(self.min_pos)
        self.Verbose = Verbose
        
        self.guard_domains = []
        self.guards = []
        self.connectors = []
        self.guard_regions = []
        self.guard_seeds = []
        self.connector_regions = []

    def make_guard(self, q):
        region = self.iris_handle(q)
        ellipse = region.MaximumVolumeInscribedEllipsoid()        
        self.guards.append(SetNode(ellipse.center(), region, 'guard'))
        self.guard_domains.append([self.guards[-1]])
        self.guard_regions.append(region)

    def point_in_guard_regions(self, q):
        for r in self.guard_regions:
            if r.PointInSet(q):
                    return True
        return False

    def build(self,):
        ntry = 0
        q = self.sample_node_pos()
            
        while ntry < self.M:
            q = self.sample_node_pos()
            vis_guards = []
            vis_guard_domains = []
            vis_guards_pt_in_set = []
            break_flag = False
            for domain in self.guard_domains:
                #is_found = False
                #break_flag = False
                for guard in domain:
                    #print(q.reshape(2,1))
                    #print(guard.position.reshape(2,1))
                    is_los, pt_in_set = self.los_handle(q.reshape(2,1), guard.region)
                    #print(is_los)
                    if is_los:
                        #is_found = True
                        if len(vis_guards) == 0:
                            vis_guards.append(guard)
                            vis_guard_domains.append(self.guard_domains.index(domain))
                            vis_guards_pt_in_set.append(pt_in_set)
                        else:
                            #is connector
                            vis_guards.append(guard)
                            vis_guards_pt_in_set.append(pt_in_set)
                            self.connectors.append(SetNode(q, None, 'connector', connections = [vis_guards, vis_guards_pt_in_set]))
                            print('connector found, ntry =', ntry)  if self.Verbose else None 
                            #merge visible domains
                            other_domain = self.guard_domains[vis_guard_domains[0]]
                            cur_domain_idx = self.guard_domains.index(domain)
                            self.guard_domains[vis_guard_domains[0]] = other_domain + domain
                            self.guard_domains = \
                            [dom for idx, dom in enumerate(self.guard_domains) if idx!= cur_domain_idx]
                            break_flag = True
                        break
                if break_flag:
                    break
            if len(vis_guards)==0:
                self.guard_seeds.append(q)
                self.make_guard(q)
                print('guard found, ntry =', ntry) if self.Verbose else None
                ntry = 0 
            else:
                ntry +=1
        print('building connector regions')
        #check if connector point in region


        if self.plot_edge is not None:
            self.plot()
        
    def plot(self,):
        for g in self.guard_seeds:
            self.plot_node(g, is_guard = True)
        for c in self.connectors:
            self.plot_node(c.position, is_guard = False)
            for vis_pt_in_set in c.connections[1]:
                self.plot_edge(c.position, vis_pt_in_set)
        for r in self.guard_regions:
            self.plot_region(r)

    def sample_node_pos(self, MAXIT = 1e4):  
        rand = np.random.rand(self.dim)
        pos_samp = self.min_pos + rand*self.min_max_diff 
        good_sample = (not self.col_handle(pos_samp)) and (not self.point_in_guard_regions(pos_samp))

        it = 0
        while not good_sample and it < MAXIT:
            rand = np.random.rand(self.dim)
            pos_samp = self.min_pos + rand*self.min_max_diff 
            good_sample = not self.col_handle(pos_samp)
            it+=1

        if not good_sample:
            raise ValueError("[VPRM ERROR] Could not find collision free point in MAXIT")
        return pos_samp

import networkx as nx
import matplotlib.pyplot as plt
from time import gmtime, strftime
from pydrake.geometry.optimization import HPolyhedron

def default_sample_ranking(vs):
    samples_outside_regions = vs.samples_outside_regions
    regions = vs.regions
    connectivity_graph = vs.connectivity_graph
    key_max = ''
    max_vis_components = -1
    #get connected components
    components = [list(a) for a in nx.connected_components(connectivity_graph)]
    #check if all nodes to connect are part of a single connected component
    for c in components:
        if vs.nodes_to_connect & set(c) == vs.nodes_to_connect:
            return None, True

    for key, s_list in samples_outside_regions.items():
        vis_components = 0
        for component in components:
            vis_regions = s_list[1]
            vis_regions_idx = [regions.index(r) for r in vis_regions]
            if len(list(set(vis_regions_idx) & set(component))):
                vis_components +=1
        if vis_components > max_vis_components:
                max_vis_components = vis_components
                key_max = key
    print(strftime("[%H:%M:%S] ", gmtime()) + '[VPRMSeeding] Num connected Components Vis:', max_vis_components)

    return key_max, not (len(samples_outside_regions.keys()) > 0)

class VPRMSeeding:
    def __init__(self,
                 samples_to_connect,
                 limits = None,
                 alpha = 0.05,
                 eps = 0.05,
                 collision_handle = None,
                 is_in_line_of_sight = None,
                 iris_handle = None,
                 iris_handle_with_obstacles = None,
                 ranking_samples_handle = default_sample_ranking,
                 point_to_region_conversion = None,
                 plot_node = None,
                 plot_edge = None,
                 plot_region = None,
                 Verbose = True):

        self.verbose = Verbose
        self.limits = limits
        self.min_pos = self.limits[0]
        self.max_pos = self.limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 
        self.dim = len(self.min_pos)
        self.alpha = alpha
        self.eps = eps
        self.M = int(np.log(alpha)/np.log(1-eps))
        if self.verbose: 
            print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] Expecting points of interest in q')
            print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] GuardInsertion attempts M:', str(self.M))
            print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] {} probability that unseen region is less than {} "%" of Cfree '.format(1-self.alpha, 100*eps))


        self.col_handle = collision_handle
        self.is_in_line_of_sight = is_in_line_of_sight
        self.grow_region_at = iris_handle
        self.grow_region_at_with_obstacles = iris_handle_with_obstacles 
        self.guard_regions = []
        self.regions = []
        self.seed_points = []
        self.samples_outside_regions = {}
        self.samples_to_connect = samples_to_connect
        self.sample_rank_handle = ranking_samples_handle
        self.nodes_to_connect = set([idx for idx, s in enumerate(self.samples_to_connect)])
        self.point_to_region_space = point_to_region_conversion
        self.need_to_convert_samples = True if self.point_to_region_space is not None else False

    def set_guard_regions(self, regions = None):
        if regions is None:
            if len(self.guard_regions)==0:
                self.regions = [self.grow_region_at(r) for r in self.samples_to_connect]
                self.seed_points = [s for s in self.samples_to_connect]
                for idx in range(len(self.regions)): self.guard_regions.append(idx) 
            else:
                raise ValueError(strftime("[%H:%M:%S] ", gmtime()) +"[VPRMSeeding] guard_regions must be an empty list")
        else:
            for idx, r in enumerate(regions):
                seed, reg = r
                self.regions.append(reg)
                self.seed_points.append(seed)
                self.guard_regions.append(idx)

    def point_in_guard_regions(self, q):
        if self.need_to_convert_samples:
            pt = self.point_to_region_space(q)
        else:
            pt = q
        for r in self.guard_regions:
            if self.regions[r].PointInSet(pt):
                    return True
        return False

    def sample_node_pos(self, MAXIT = 1e4):  
        rand = np.random.rand(self.dim)
        pos_samp = self.min_pos + rand*self.min_max_diff 
        good_sample = (not self.col_handle(pos_samp)) and (not self.point_in_guard_regions(pos_samp))
        it = 0
        while not good_sample and it < MAXIT:
            rand = np.random.rand(self.dim)
            pos_samp = self.min_pos + rand*self.min_max_diff 
            col = False
            for _ in range(10):
                r  = 0.01*(np.random.rand(self.dim)-0.5)
                col |= (self.col_handle(pos_samp+r) > 0)
            good_sample = (not col) and (not self.point_in_guard_regions(pos_samp))
            it+=1
        if not good_sample:
            raise ValueError(strftime("[%H:%M:%S] ", gmtime()) +"[VPRMSeeding] ERROR: Could not find collision free point in MAXIT %d".format(MAXIT))
        return pos_samp

    def load_checkpoint(self, checkpoint):
        self.seed_points = checkpoint['seedpoints']
        A = checkpoint['regionsA']
        B = checkpoint['regionsB']
        self.regions = [HPolyhedron(a,b) for a,b, in zip(A,B)]
        self.guard_regions = [idx for idx in range(len(self.regions))]
        self.samples_outside_regions = {}
        vis_reg = [[self.regions[idx] for idx in vis] for vis in checkpoint['sample_set_vis_regions']]
        for i ,pt in enumerate(checkpoint['sample_set_points']):
            self.samples_outside_regions[str(pt)] = [pt, vis_reg[i]]
        
        self.connectivity_graph = nx.Graph()
        for idx in range(len(self.guard_regions)):
            self.connectivity_graph.add_node(idx)
            
        for idx1 in range(len(self.guard_regions)):
            for idx2 in range(idx1 +1, len(self.guard_regions)):
                r1 = self.regions[idx1]
                r2 = self.regions[idx2]
                if r1.IntersectsWith(r2):
                    self.connectivity_graph.add_edge(idx1,idx2)
        print(strftime("[%H:%M:%S] ", gmtime()) +"[VPRMSeeding] Checkpoint loaded successfully, current state is at end of guard phase")

    def run(self):
        self.set_guard_regions()
        self.guard_phase()        
        done_connecting = self.connectivity_phase()
            
    def draw_connectivity_graph(self,):
        fig = plt.figure()
        colors = []
        for idx in range(len(self.regions)):
            if idx<len(self.samples_to_connect):
                colors.append('c')
            elif idx<len(self.guard_regions):
                colors.append('b')
            else:
                colors.append('m')
        nx.draw_spring(self.connectivity_graph, 
                              with_labels = True, 
                              node_color = colors)

    def guard_phase(self,):
        it = 0
        while it < self.M:
            p = self.sample_node_pos()
            add_to_sample_set = False
            visible_regions = []
            for idx_guard in self.guard_regions: # zip(self.guard_regions, self.guard_region_seed_points):
                guard_seed_point = self.seed_points[idx_guard]
                guard_region = self.regions[idx_guard]
                if self.is_in_line_of_sight(p.reshape(-1,1), guard_seed_point.reshape(-1,1))[0]:
                    add_to_sample_set = True
                    visible_regions.append(guard_region)
            if add_to_sample_set:
                self.samples_outside_regions[str(p)] = [p, visible_regions]
            else:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +"[VPRMSeeding] New guard placed N = ", str(len(self.guard_regions)), "it = ", it) 
                try:
                    rnew = self.grow_region_at(p)
                    self.regions.append(rnew)
                    self.seed_points.append(p)
                    self.guard_regions.append(len(self.regions)-1)
                    it = 0
                    #update visibility and cull points
                    keys_to_del = []
                    for s_key in self.samples_outside_regions.keys():
                        s = self.samples_outside_regions[s_key][0]
                        if self.need_to_convert_samples:
                            s = self.point_to_region_space(s)
                        if rnew.PointInSet(s.reshape(-1,1)):
                            keys_to_del.append(s_key)
                        elif self.is_in_line_of_sight(s.reshape(-1,1), p.reshape(-1,1))[0]:
                            self.samples_outside_regions[s_key][1].append(rnew)
                    for k in keys_to_del:
                        del self.samples_outside_regions[k]
                    if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] Sample set size',len(self.samples_outside_regions.keys()))
                except:
                    print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] Mosek failed, deleting point')
            it+=1

        self.connectivity_graph = nx.Graph()
        for idx in range(len(self.guard_regions)):
            self.connectivity_graph.add_node(idx)
            
        for idx1 in range(len(self.guard_regions)):
            for idx2 in range(idx1 +1, len(self.guard_regions)):
                r1 = self.regions[idx1]
                r2 = self.regions[idx2]
                if r1.IntersectsWith(r2):
                    self.connectivity_graph.add_edge(idx1,idx2)
        #if self.verbose: print("[VPRMSeeding] Connectivity phase")

    def connectivity_phase(self,):
        done_connecting = False
        while not done_connecting:
            best_sample, done_connecting = self.sample_rank_handle(self)
            if done_connecting:
                break
            loc_best_sample = self.samples_outside_regions[best_sample][0]
            try:
                nr = self.grow_region_at_with_obstacles(loc_best_sample.reshape(-1,1), self.regions)
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] New region added', loc_best_sample.reshape(-1))
                self.regions.append(nr)
                self.seed_points.append(loc_best_sample.copy())
                idx_new_region = len(self.regions)-1
                self.connectivity_graph.add_node(idx_new_region)
                keys_to_del = []
                for s_key in self.samples_outside_regions.keys():
                    s = self.samples_outside_regions[s_key][0]
                    if self.need_to_convert_samples:
                            s = self.point_to_region_space(s)
                    if nr.PointInSet(s.reshape(-1,1)):
                        keys_to_del.append(s_key)
                    #elif is_LOS(s, loc_max)[0]:
                    #    vs.samples_outside_regions[s_key][1].append(nr)
                #numerics
                if best_sample not in keys_to_del:
                    keys_to_del.append(best_sample)
                for k in keys_to_del:
                    del self.samples_outside_regions[k]
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] Sample set size',len(self.samples_outside_regions.keys()), 'num keys to del ', len(keys_to_del))
                #update connectivity graph
                for idx, r in enumerate(self.regions[:-1]):
                    if r.IntersectsWith(nr):
                        self.connectivity_graph.add_edge(idx, idx_new_region)
            except:
                print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] Failed, deleting point')
                del self.samples_outside_regions[best_sample]
        return done_connecting


class RandSeeding:
    def __init__(self,
                 samples_to_connect,
                 limits = None,
                 alpha = 0.05,
                 eps = 0.05,
                 collision_handle = None,
                 iris_handle = None,
                 iris_handle_with_obstacles = None,
                 point_to_region_conversion = None,
                 plot_node = None,
                 plot_edge = None,
                 plot_region = None,
                 Verbose = True):

        self.verbose = Verbose
        self.limits = limits
        self.min_pos = self.limits[0]
        self.max_pos = self.limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 
        self.dim = len(self.min_pos)
        self.alpha = alpha
        self.eps = eps
        self.M = int(np.log(alpha)/np.log(1-eps))
        if self.verbose: 
            print(strftime("[%H:%M:%S] ", gmtime()) +'[RandSeeding] Expecting points of interest in q')
            print(strftime("[%H:%M:%S] ", gmtime()) +'[RandSeeding] GuardInsertion attempts M:', str(self.M))
            print(strftime("[%H:%M:%S] ", gmtime()) +'[RandSeeding] {} probability that unseen region is less than {} "%" of Cfree '.format(1-self.alpha, 100*eps))


        self.col_handle = collision_handle
        self.grow_region_at = iris_handle
        self.grow_region_at_with_obstacles = iris_handle_with_obstacles 
        self.regions = []
        self.seed_points = []
        self.samples_to_connect = samples_to_connect
        self.nodes_to_connect = set([idx for idx, s in enumerate(self.samples_to_connect)])
        self.point_to_region_space = point_to_region_conversion
        self.need_to_convert_samples = True if self.point_to_region_space is not None else False

    def set_init_regions(self, regions = None):
        if regions is None:
            if len(self.regions)==0:
                self.regions = [self.grow_region_at(r) for r in self.samples_to_connect]
                self.seed_points = [s for s in self.samples_to_connect]
            else:
                raise ValueError(strftime("[%H:%M:%S] ", gmtime()) +"[RandSeeding] regions must be an empty list")
        else:
            for idx, r in enumerate(regions):
                seed, reg = r
                self.regions.append(reg)
                self.seed_points.append(seed)
                
    def point_in_regions(self, q):
        if self.need_to_convert_samples:
            pt = self.point_to_region_space(q)
        else:
            pt = q
        for r in self.regions:
            if r.PointInSet(pt):
                    return True
        return False

    def sample_node_pos(self, MAXIT = 1e4):  
        rand = np.random.rand(self.dim)
        pos_samp = self.min_pos + rand*self.min_max_diff 
        good_sample = (not self.col_handle(pos_samp)) 
        it = 0
        while not good_sample and it < MAXIT:
            rand = np.random.rand(self.dim)
            pos_samp = self.min_pos + rand*self.min_max_diff 
            col = False
            for _ in range(10):
                r  = 0.01*(np.random.rand(self.dim)-0.5)
                col |= (self.col_handle(pos_samp+r) > 0)
            good_sample = not col
            it+=1
        if not good_sample:
            raise ValueError(strftime("[%H:%M:%S] ", gmtime()) +"[RandSeeding] ERROR: Could not find collision free point in MAXIT %d".format(MAXIT))
        return pos_samp

    def load_checkpoint(self, checkpoint):
        self.seed_points = checkpoint['seedpoints']
        A = checkpoint['regionsA']
        B = checkpoint['regionsB']
        self.regions = [HPolyhedron(a,b) for a,b, in zip(A,B)]
        #vis_reg = [[self.regions[idx] for idx in vis] for vis in checkpoint['sample_set_vis_regions']]
        #for i ,pt in enumerate(checkpoint['sample_set_points']):
        #    self.samples_outside_regions[str(pt)] = [pt, vis_reg[i]]
        
        self.connectivity_graph = nx.Graph()
        for idx in range(len(self.regions)):
            self.connectivity_graph.add_node(idx)
            
        for idx1 in range(len(self.regions)):
            for idx2 in range(idx1 +1, len(self.regions)):
                r1 = self.regions[idx1]
                r2 = self.regions[idx2]
                if r1.IntersectsWith(r2):
                    self.connectivity_graph.add_edge(idx1,idx2)
        print(strftime("[%H:%M:%S] ", gmtime()) +"[RandSeeding] Checkpoint loaded successfully, current state is at end of guard phase")

    def run(self):
        self.set_init_regions()
        self.sample_regions_phase()        
        done_connecting = self.connectivity_phase()
            
    def draw_connectivity_graph(self,):
        fig = plt.figure()
        colors = []
        for idx in range(len(self.regions)):
            if idx<len(self.samples_to_connect):
                colors.append('c')
            elif idx<len(self.regions):
                colors.append('b')
            else:
                colors.append('m')
        nx.draw_spring(self.connectivity_graph, 
                              with_labels = True, 
                              node_color = colors)

    def sample_regions_phase(self,):
        it = 0
        self.connectivity_graph = nx.Graph()
        for idx in range(len(self.regions)):
            self.connectivity_graph.add_node(idx)

        for idx1 in range(len(self.regions)):
            for idx2 in range(idx1 +1, len(self.regions)):
                r1 = self.regions[idx1]
                r2 = self.regions[idx2]
                if r1.IntersectsWith(r2):
                    self.connectivity_graph.add_edge(idx1,idx2)

        while it < self.M:
            p = self.sample_node_pos()
            if self.point_in_regions(p):
                it+=1
            else:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +"[RandSeeding] New Region placed N = ", str(len(self.regions)), ", it = ", str(it)) 
                try:
                    rnew = self.grow_region_at_with_obstacles(p.reshape(-1, 1), self.regions)
                    self.regions.append(rnew)
                    it = 0
                    idx_new_region = len(self.regions)-1
                    self.connectivity_graph.add_node(idx_new_region)
                    #update connectivity graph
                    for idx, r in enumerate(self.regions[:-1]):
                        if r.IntersectsWith(rnew):
                            self.connectivity_graph.add_edge(idx, idx_new_region)
                    #check if all points in one connected component
                    #get connected components
                    components = [list(a) for a in nx.connected_components(self.connectivity_graph)]
                    #check if all nodes to connect are part of a single connected component
                    for c in components:
                        if self.nodes_to_connect & set(c) == self.nodes_to_connect:
                            return True
                except:
                    print(strftime("[%H:%M:%S] ", gmtime()) +'[RandSeeding] Mosek failed, deleting point')
        return False    
        
            
        
        #if self.verbose: print("[VPRMSeeding] Connectivity phase")

   