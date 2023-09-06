import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra, csgraph_from_dense
from scipy.sparse import coo_matrix


class Node:
    def __init__(self, pos, cost, parent=None):
        self.pos = pos
        self.parent = parent
        self.cost = cost
        self.children = []
        self.id = None


class PRM:
    def __init__(self,
                 limits,
                 num_points,
                 col_func_handle,
                 num_neighbours=5,
                 dist_thresh=0.1,
                 num_col_checks=10,
                 max_it = 1e4,
                 initial_points = None,
                 verbose=False,
                 plotcallback=None,
                 sample_via_gaussian = False,
                 gaussian_sample_var = 0.1,
                 node_sampling_function = None
                 ):

        # col_check(pos) == True -> in collision!
        self.in_collision = col_func_handle
        self.check_col = False if self.in_collision == None else True
        self.dim = len(limits[0])
        self.min_pos = limits[0]
        self.max_pos = limits[1]
        self.min_max_diff = self.max_pos - self.min_pos
        self.num_neighbours = num_neighbours
        self.dist_thresh = dist_thresh
        self.t_check = np.linspace(0, 1, num_col_checks)
        self.plotcallback = plotcallback
        self.verbose = verbose

        self.node_sampling_function = self.sample_node_pos if node_sampling_function is None else node_sampling_function


        # generate n samples using rejection sampling
        nodes = [] if initial_points is None else initial_points
        for idx in range(num_points - len(nodes)):
            if sample_via_gaussian:
                nodes.append(self.sample_via_gaussian(nodes, var = gaussian_sample_var))
            else:
                nodes.append(self.node_sampling_function(MAXIT=max_it))
            if self.verbose and idx % 30 == 0:
                print('[PRM] Samples', idx)
        self.nodes_list = nodes
        self.nodes = np.array(nodes)
        self.nodes_kd = cKDTree(self.nodes)


        # generate edges
        self.adjacency_list, self.dist_adj = self.connect_nodes()
        self.make_start_end_pairs()
        self.plot()

    def sample_via_gaussian(self, cur_nodes, var = 0.1, collision_free = True, MAXIT=1e4):
        """
        randomly select a node and sample from a gaussian around that node
        """
        ind = np.random.randint(0, len(cur_nodes))
        mean = cur_nodes[ind]
        ctr = 0
        while ctr < MAXIT:
            pos_samp = mean + var*np.random.randn(len(self.min_pos))
            good_sample = not self.check_col or not self.in_collision(pos_samp)
            if good_sample:
                return pos_samp 
            ctr += 1
        return self.node_sampling_function(collision_free=collision_free, MAXIT=MAXIT)
             


    def sample_node_pos(self, collision_free=True, MAXIT=1e4):

        # rand = np.random.rand(self.dim)
        # pos_samp = self.min_pos + rand * self.min_max_diff
        pos_samp = np.random.uniform(self.min_pos, self.max_pos)

        if self.check_col:
            good_sample = not self.in_collision(pos_samp) if collision_free else True
        else:
            good_sample = True

        it = 0
        while not good_sample and it < MAXIT:
            # rand = np.random.rand(self.dim)
            # pos_samp = self.min_pos + rand * self.min_max_diff
            pos_samp = np.random.uniform(self.min_pos, self.max_pos)
            good_sample = not self.in_collision(pos_samp)

            it += 1
        # good_sample = True
        if not good_sample:
            print("[PRM ERROR] Could not find collision free point in MAXIT")
            raise NotImplementedError
        return pos_samp

    def connect_nodes(self, ):
        adjacency_list = []
        dist_adj = []
        for node_idx in range(self.nodes.shape[0]):
            if self.verbose and node_idx % 20 == 0:
                print('[PRM] Nodes connected:', node_idx)
            edges = []
            edge_dist = []
            dists, idxs = self.nodes_kd.query(self.nodes[node_idx, :], k=self.num_neighbours, p=2,
                                              distance_upper_bound=self.dist_thresh)
            # linesearch connection for collision
            for step in range(len(idxs)):
                nearest_idx = idxs[step]
                if not dists[step] == np.inf:
                    add = True
                    for t in self.t_check:
                        pos = (1 - t) * self.nodes[node_idx, :] + t * self.nodes[nearest_idx, :]
                        if self.in_collision(pos):
                            add = False
                            break
                    if add:
                        edges.append(nearest_idx)
                        edge_dist.append(dists[step])
            adjacency_list.append(edges)
            dist_adj.append(edge_dist)
        return adjacency_list, dist_adj

    def add_start_end(self, start, end):
        self.nodes_list.append(start)
        self.nodes_list.append(end)
        self.nodes = np.array(self.nodes_list)
        self.nodes_kd = cKDTree(self.nodes)
        for node_idx in [-2, -1]:
            edges = []
            edge_dist = []
            dists, idxs = self.nodes_kd.query(self.nodes[node_idx, :], k=self.num_neighbours * 5, p=2,
                                              distance_upper_bound=self.dist_thresh * 2)
            # linesearch connection for collision
            for step in range(len(idxs)):
                nearest_idx = idxs[step]
                if not dists[step] == np.inf:
                    add = True
                    for t in self.t_check:
                        pos = (1 - t) * self.nodes[node_idx, :] + t * self.nodes[nearest_idx, :]
                        if self.in_collision(pos):
                            add = False
                            break
                    if add:
                        edges.append(nearest_idx)
                        edge_dist.append(dists[step])
            self.adjacency_list.append(edges)
            self.dist_adj.append(edge_dist)

    def build_adjacency_mat(self, ):
        N = len(self.nodes)
        data = []
        rows = []
        cols = []

        ad_mat = coo_matrix((N, N), np.float32)

        for idx in range(N):
            nei_idx = 0
            for nei in self.adjacency_list[idx]:
                if not nei == idx:
                    data.append(self.dist_adj[idx][nei_idx])
                    rows.append(idx)
                    cols.append(nei)
                    data.append(self.dist_adj[idx][nei_idx])
                    rows.append(nei)
                    cols.append(idx)
                    # ad_mat[idx, nei] = self.dist_adj[idx][nei_idx]
                    # ad_mat[nei, idx] = self.dist_adj[idx][nei_idx]
                nei_idx += 1

        ad_mat = coo_matrix((data, (rows, cols)), shape=(N, N))
        return ad_mat

    def find_shortest_path(self):
        ad_mat = self.build_adjacency_mat()
        dist, pred = dijkstra(ad_mat, directed=False, indices=-2, return_predecessors=True)
        print(f'{len(np.argwhere(pred == -9999))} disconnected nodes', np.argwhere(pred == -9999))
        pred[pred == -9999] = -100000000

        sp_list = []
        sp_length = dist[-1]
        current_idx = -1
        sp_list.append(self.nodes[current_idx])
        while not current_idx == ad_mat.shape[0] - 2:
            current_idx = pred[current_idx]
            sp_list.append(self.nodes[current_idx])
        return sp_list, sp_length

    def plot(self):
        if self.plotcallback:
            self.plotcallback(self.nodes, self.adjacency_list)
        else:
            pass

    def make_start_end_pairs(self):
        endpoint_index_set = set()
        self.prm_pairs = []
        for neighbors in self.adjacency_list:
            for n in neighbors[1:]:
                endpoint_index_set.add((neighbors[0], n))
        for i, (idx0, idx1) in enumerate(endpoint_index_set):
            self.prm_pairs.append((self.nodes[idx0], self.nodes[idx1]))
    def __deepcopy__(self):
        limits = np.array([self.min_pos, self.max_pos])
        tmp = PRM(limits, len(self.nodes_list),
                  self.in_collision, self.num_neighbours,
                  self.dist_thresh, len(self.t_check),
                  self.verbose, None)
        tmp.nodes_list = self.nodes_list
        tmp.nodes = self.nodes
        tmp.nodes_kd = self.nodes_kd
        tmp.adjacency_list = self.adjacency_list
        tmp.dist_adj = self.dist_adj
        tmp.plotcallback = self.plotcallback
        return tmp


