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