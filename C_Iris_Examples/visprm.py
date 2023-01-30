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
    nodes_to_connect = vs.nodes_to_connect if len(vs.nodes_to_connect) else vs.guard_regions
    for c in components:
        if set(nodes_to_connect) & set(c) == set(nodes_to_connect):
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

def sample_ranking_connected_components_weight(vs):
    is_done = False
    samples_outside_regions = vs.samples_outside_regions
    regions = vs.regions
    connectivity_graph = vs.connectivity_graph
    key_max = ''
    max_vis_components = -1
    #get connected components
    components = [list(a) for a in nx.connected_components(connectivity_graph)]
    #check if all nodes to connect are part of a single connected component
    nodes_to_connect = vs.nodes_to_connect if len(vs.nodes_to_connect) else vs.guard_regions
    for c in components:
        if (set(nodes_to_connect) & set(c) == set(nodes_to_connect)) :
            is_done = True

    for key, s_list in samples_outside_regions.items():
        vis_components = 0
        for component in components:
            vis_regions = s_list[1]
            vis_regions_idx = [regions.index(r) for r in vis_regions]
            if len(list(set(vis_regions_idx) & set(component))):
                overlap = 1.0 if len(list(set(vis_regions_idx)&set(nodes_to_connect))) else 0
                vis_components += 5 + overlap
        if vis_components > max_vis_components:
                max_vis_components = vis_components
                key_max = key
    print(strftime("[%H:%M:%S] ", gmtime()) + '[VPRMSeeding] Num weighted connected components Vis:', max_vis_components)

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
    
    def point_in_regions(self, q):
        if self.need_to_convert_samples:
            pt = self.point_to_region_space(q)
        else:
            pt = q
        for r in self.regions:
            if r.PointInSet(pt):
                    return True
        return False

    def sample_node_pos(self, outside_regions = True, MAXIT = 1e4):  
        it = 0
        good_sample = False
        while not good_sample and it < MAXIT:
            rand = np.random.rand(self.dim)
            pos_samp = self.min_pos + rand*self.min_max_diff 
            col = False
            for _ in range(10):
                r  = 0.01*(np.random.rand(self.dim)-0.5)
                col |= (self.col_handle(pos_samp+r) > 0)
            if outside_regions: 
                good_sample = (not col) and (not self.point_in_guard_regions(pos_samp))
            else:
                good_sample = (not col)
            it+=1
        if not good_sample:
            raise ValueError(strftime("[%H:%M:%S] ", gmtime()) +"[VPRMSeeding] ERROR: Could not find collision free point in MAXIT %d".format(MAXIT))
        return pos_samp

    def sample_in_regions(self, MAXIT = 1e3):  
        it = 0
        good_sample = False
        while not good_sample and it < MAXIT:
            rand = np.random.rand(self.dim)
            pos_samp = self.min_pos + rand*self.min_max_diff 
            good_sample = self.point_in_regions(pos_samp)
            it+=1
        if not good_sample:
            raise ValueError(strftime("[%H:%M:%S] ", gmtime()) +"[VPRMSeeding] ERROR: Could not find point in regions MAXIT %d".format(MAXIT))
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
    
    def load_full_checkpoint(self, checkpoint):
        self.seed_points = checkpoint['seedpoints']
        A = checkpoint['regionsA']
        B = checkpoint['regionsB']
        self.regions = [HPolyhedron(a,b) for a,b, in zip(A,B)]
        self.guard_regions = checkpoint['guard_regions'] 
        self.samples_outside_regions = {}
        vis_reg = [[self.regions[idx] for idx in vis] for vis in checkpoint['sample_set_vis_regions']]
        for i ,pt in enumerate(checkpoint['sample_set_points']):
            self.samples_outside_regions[str(pt)] = [pt, vis_reg[i]]
        
        self.connectivity_graph = nx.Graph()
        for idx in range(len(self.regions)):
            self.connectivity_graph.add_node(idx)
        self.connectivity_edges = []
        for idx1 in range(len(self.regions)):
            if idx1%10 == 0: print(idx1/len(self.regions))
            for idx2 in range(idx1 +1, len(self.regions)):
                r1 = self.regions[idx1]
                r2 = self.regions[idx2]
                if r1.IntersectsWith(r2):
                    self.connectivity_edges.append([idx1, idx2])
                    #self.connectivity_graph.add_edge(idx1,idx2)
        print(strftime("[%H:%M:%S] ", gmtime()) +"[VPRMSeeding] Checkpoint loaded successfully")

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
            try:
                p = self.sample_node_pos(outside_regions=False)
            except:
                print(strftime("[%H:%M:%S] ", gmtime()) +"[VPRMSeeding] No sample found outside of regions ")
                break
            add_to_sample_set = False
            visible_regions = []
            p_region_space = self.point_to_region_space(p) if self.need_to_convert_samples else p
            for idx_guard in self.guard_regions: 
                guard_seed_point_q = self.seed_points[idx_guard]
                guard_seed_point_region_space = self.point_to_region_space(guard_seed_point_q) if self.need_to_convert_samples else guard_seed_point_q
                guard_region = self.regions[idx_guard]
                #check visibility in t
                if self.is_in_line_of_sight(p_region_space.reshape(-1,1), guard_seed_point_region_space.reshape(-1,1))[0]:
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
                        s_Q = self.samples_outside_regions[s_key][0]
                        if self.need_to_convert_samples:
                            s_conv = self.point_to_region_space(s_Q)
                        else:
                            s_conv = s_Q
                        # if rnew.PointInSet(s_conv.reshape(-1,1)):
                        #     keys_to_del.append(s_key)
                        if self.is_in_line_of_sight(s_conv.reshape(-1,1), p_region_space.reshape(-1,1))[0]:
                            self.samples_outside_regions[s_key][1].append(rnew)
                    # for k in keys_to_del:
                    #     del self.samples_outside_regions[k]
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
    
    def get_new_seed_candidates(self, guard):
        vertices = self.compute_kernel_of_guard(guard)
        vertices += [self.seed_points[guard]]
        print(len(vertices))
        graph = nx.Graph()
        if len(vertices)>1:
            for i1, v1 in enumerate(vertices):
                for i2, v2 in enumerate(vertices):
                    if self.need_to_convert_samples:
                        v1_c = self.point_to_region_space(v1)
                        v2_c = self.point_to_region_space(v2)
                    else:
                        v1_c = v1
                        v2_c = v2
                    if i1!=i2 and self.is_in_line_of_sight(v1_c.reshape(-1,1), v2_c.reshape(-1,1))[0]:
                        graph.add_edge(i1,i2)
            # print(len(vertices))
            # print(len(graph.edges))
            new_cands = nx.maximal_independent_set(graph)
            return [vertices[c] for c in new_cands]
        else:
            return [self.seed_points[guard]]

    def refine_guards_greedy(self):
        continue_splitting = True
        while continue_splitting:
            candidate_splits = [self.get_new_seed_candidates(g) for g in self.guard_regions]
            best_split = max(candidate_splits, key = len)
            best_split_idx = candidate_splits.index(best_split) 
            if len(best_split)>1:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] Guard found to split into', len(best_split))
                #generate new regions
                nr = [self.grow_region_at(s) for s in best_split]
                self.regions += nr
                self.seed_points += best_split
                r_old = self.regions[best_split_idx]
                #keys_to_del = []
                gs_conv = [self.point_to_region_space(s) for s in best_split] if self.need_to_convert_samples else best_split
                
                for s_key in self.samples_outside_regions.keys():
                    vis_regs = self.samples_outside_regions[s_key][1] 
                    if r_old in vis_regs:
                        vis_regs.remove(r_old)
                    s_Q = self.samples_outside_regions[s_key][0]
                    if self.need_to_convert_samples:
                        s_conv = self.point_to_region_space(s_Q)
                    else:
                        s_conv = s_Q 
                    for idnr, gs in enumerate(gs_conv):
                        if self.is_in_line_of_sight(s_conv.reshape(-1,1), gs.reshape(-1,1))[0]:
                            vis_regs.append(nr[idnr])
                
                del self.seed_points[best_split_idx]
                del self.regions[best_split_idx]
                self.guard_regions = [self.regions.index(r) for r in self.regions]
            else:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] No guard to split')
                continue_splitting = False
        #rebuild connectivity graph
        self.connectivity_graph = nx.Graph()
        for idx in range(len(self.guard_regions)):
            self.connectivity_graph.add_node(idx)

        for idx1 in range(len(self.guard_regions)):
            for idx2 in range(idx1 +1, len(self.guard_regions)):
                r1 = self.regions[idx1]
                r2 = self.regions[idx2]
                if r1.IntersectsWith(r2):
                    self.connectivity_graph.add_edge(idx1,idx2)
        return

    # def find_guard_to_refine(self,):
    #     to_split = []
    #     for goi in self.guard_regions:
    #         g1,g2 = None, None
                
    #         if goi >= len(self.samples_to_connect): 

    #             ker = self.compute_kernel_of_guard(goi)
    #             targ_seed = self.seed_points[goi]
                
    #             if len(ker) >10:
    #                 found = False
    #                 for idx, k1 in enumerate(ker[:-1]):
    #                     for k2 in ker[idx:]:
    #                         #print('a')
    #                         if not self.is_in_line_of_sight(k1, k2)[0]:
    #                             g1 = k1
    #                             g2 = k2
    #                             found = True
    #                             break
    #                     if found:
    #                         break
    #             ker = np.array(ker)
    #         #sample_set = np.array(sample_set)
    #         #print(len(ker))
    #         #print(g1,g2)
    #         if g1 is not None:
    #             to_split.append([goi, targ_seed, ker, g1, g2])
    #             if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] Guard found to split')
    #             return to_split
    #     if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] No guard to split')
    #     return to_split

    # def split_refineable_guard(self, to_split):
    #     if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] N = ', len(to_split), ' guards to split')
    #     if len(to_split):
    #         goi, targ_seed, ker, g1, g2 = to_split[0]
    #         #generate new regions
    #         r1 = self.grow_region_at(g1)
    #         r2 = self.grow_region_at(g2)
    #         self.regions += [r1, r2]
    #         #vs.guard_regions += [vs.regions.index(r) for r in [r1,r2]]
    #         self.seed_points += [g1, g2]
    #         r_old = self.regions[goi]
    #         keys_to_del = []
    #         g1_conv = self.point_to_region_space(g1) if self.need_to_convert_samples else g1
    #         g2_conv = self.point_to_region_space(g2) if self.need_to_convert_samples else g2

    #         for s_key in self.samples_outside_regions.keys():
    #             vis_regs = self.samples_outside_regions[s_key][1] 
    #             if  r_old in vis_regs:
    #                 vis_regs.remove(r_old)
    #             s_Q = self.samples_outside_regions[s_key][0]
    #             if self.need_to_convert_samples:
    #                 s_conv = self.point_to_region_space(s_Q)
    #             else:
    #                 s_conv = s_Q 
    #             if self.is_in_line_of_sight(s_conv.reshape(-1,1), g1_conv.reshape(-1,1))[0]:
    #                 vis_regs.append(r1)
    #             if self.is_in_line_of_sight(s_conv.reshape(-1,1), g2_conv.reshape(-1,1))[0]:
    #                 vis_regs.append(r2)
    #         for k in keys_to_del:
    #             del self.samples_outside_regions[k]
            
    #         del self.seed_points[goi]
    #         del self.regions[goi]
    #     self.guard_regions = [self.regions.index(r) for r in self.regions]
    
    # def refine_guards(self,):
    #     keep_splitting = True
    #     while keep_splitting:
    #         to_split = self.find_guard_to_refine()
    #         if len(to_split):
    #             #goi, targ_seed, ker, g1, g2 = to_split[0]
    #             self.split_refineable_guard(to_split)
    #             self.guard_phase()
    #             keep_splitting = True
    #         else:
    #             keep_splitting = False

    #     #rebuild connectivity graph
    #     self.connectivity_graph = nx.Graph()
    #     for idx in range(len(self.guard_regions)):
    #         self.connectivity_graph.add_node(idx)

    #     for idx1 in range(len(self.guard_regions)):
    #         for idx2 in range(idx1 +1, len(self.guard_regions)):
    #             r1 = self.regions[idx1]
    #             r2 = self.regions[idx2]
    #             if r1.IntersectsWith(r2):
    #                 self.connectivity_graph.add_edge(idx1,idx2)

    def compute_kernel_of_guard(self, guard):
        ker = []
        for sampdat in self.samples_outside_regions.values():
            pos = sampdat[0]
            vis = [self.regions.index(rviz) for rviz in sampdat[1]]
            if len(vis)==1 and vis[0] == guard:
                ker.append(pos)
        return ker 
    
    def connectivity_phase(self,):
        #delete points in regions
        keys_to_del = []
        for s_key in self.samples_outside_regions.keys():
            s_Q = self.samples_outside_regions[s_key][0]
            if self.need_to_convert_samples:
                s_conv = self.point_to_region_space(s_Q)
            else:
                s_conv = s_Q
            for r in self.regions:
                if r.PointInSet(s_conv.reshape(-1,1)):
                    keys_to_del.append(s_key)
                    break
        for k in keys_to_del:
            del self.samples_outside_regions[k]
        #begin connectivity phase    
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

    def fill_remaining_space_phase(self,):
        it = 0
        # self.connectivity_graph = nx.Graph()
        # for idx in range(len(self.regions)):
        #     self.connectivity_graph.add_node(idx)

        # for idx1 in range(len(self.regions)):
        #     for idx2 in range(idx1 +1, len(self.regions)):
        #         r1 = self.regions[idx1]
        #         r2 = self.regions[idx2]
        #         if r1.IntersectsWith(r2):
        #             self.connectivity_graph.add_edge(idx1,idx2)

        while it < self.M:
            p = self.sample_node_pos(outside_regions=False)
            if self.point_in_regions(p):
                it+=1
            else:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +"[VPRMSeeding] New Region placed N = ", str(len(self.regions)), ", it = ", str(it)) 
                try:
                    rnew = self.grow_region_at_with_obstacles(p.reshape(-1, 1), self.regions)
                    self.regions.append(rnew)
                    self.seed_points.append(p)
                    it = 0
                    idx_new_region = len(self.regions)-1
                    self.connectivity_graph.add_node(idx_new_region)
                    #update connectivity graph
                    for idx, r in enumerate(self.regions[:-1]):
                        if r.IntersectsWith(rnew):
                            self.connectivity_graph.add_edge(idx, idx_new_region)
                except:
                    print(strftime("[%H:%M:%S] ", gmtime()) +'[VPRMSeeding] Mosek failed, deleting point')
        #get connected components
        components = [list(a) for a in nx.connected_components(self.connectivity_graph)]
        nodes_to_connect = self.nodes_to_connect if len(self.nodes_to_connect) else self.guard_regions
        for c in components:
            if set(nodes_to_connect) & set(c) == set(nodes_to_connect):
                return  True
        return False    

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
                 terminate_early = True,
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
        self.terminate_early = terminate_early
        if len(self.samples_to_connect) == 0:
            self.terminate_early = False
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
                    self.seed_points.append(p)
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
                    if self.terminate_early:
                        for c in components:
                            if set(self.nodes_to_connect) & set(c) == set(self.nodes_to_connect):
                                return True
                except:
                    print(strftime("[%H:%M:%S] ", gmtime()) +'[RandSeeding] Mosek failed, deleting point')
        
        components = [list(a) for a in nx.connected_components(self.connectivity_graph)]
        for c in components:
            if set(self.nodes_to_connect) & set(c) == set(self.nodes_to_connect):
                return True
        return False    
        
            
        
        #if self.verbose: print("[VPRMSeeding] Connectivity phase")

   