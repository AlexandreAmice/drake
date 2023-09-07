from scipy.spatial import KDTree
import networkx as nx
from pydrake.all import Sphere, Rgba, RigidBody
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from visualization_utils import visualize_body_at_s, VisualizationBundle
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
import visualization_utils as vis_utils


class StraightLineCollisionChecker:
    def __init__(self, in_collision_handle, query_density=100):
        self.in_collision_handle = in_collision_handle
        self.query_density = query_density

    def straight_line_has_collision(self, start, end):
        for pos in np.linspace(start, end, self.query_density):
            if self.in_collision_handle(pos):
                return True
        return False


@dataclass
class DrawTreeOptions:
    edge_color: Rgba = Rgba(0, 0, 1, 0.5)
    start_color: Rgba = Rgba(0, 1, 0, 0.5)
    end_color: Rgba = Rgba(1, 0, 0, 0.5)
    path_size: float = 0.01
    num_points: int = 100
    start_end_radius: float = 0.05


class PRM:
    def __init__(
        self,
        node_sampling_fun,  # generate a random sample. It does not need to be collision free.
        num_points,
        straight_line_col_checker,
        dist_thresh=0.1,
        num_neighbors=5,
        max_it=int(1e4),
        initial_points=None,
    ):
        self.max_it = max_it

        self.prm = nx.Graph()
        if initial_points is not None:
            for p in initial_points:
                self.prm.add_node(p)

        self._build(
            node_sampling_fun,
            straight_line_col_checker,
            num_points,
            max_it,
            num_neighbors,
            dist_thresh,
        )

        print(f"PRM has {len(self.prm.nodes)} nodes")
        print(f"PRM has {len(self.prm.edges)} edges")

    def _build(
        self,
        node_sampling_fun,
        num_points,
        straight_line_col_checker,
        max_it,
        num_neighbors,
        dist_thresh,
    ):
        self._sample_nodes(
            node_sampling_fun, straight_line_col_checker, num_points, max_it
        )
        self.node_kd_tree = cKDTree(self.prm.nodes)
        self._connect_nodes(num_neighbors, straight_line_col_checker, dist_thresh)

    def _sample_nodes(
        self, node_sampling_fun, straight_line_col_checker, num_points, max_it
    ):
        add_node_attempt_ctr = 0
        node_added_ctr = 0
        while node_added_ctr < num_points:
            x = node_sampling_fun()
            if not straight_line_col_checker.in_collision_handle(x):
                add_node_attempt_ctr = 0
                self.prm.add_node(tuple(x))
                node_added_ctr += 1

            add_node_attempt_ctr += 1
            if add_node_attempt_ctr > max_it:
                import warnings

                txt = (
                    f"[PRM] failed to find a collision free point after {max_it}. "
                    f"Building PRM with {len(self.prm.nodes)}/{num_points}"
                )
                warnings.warn(txt)
                print(txt)
                print()
                break

    def _connect_nodes(self, num_neighbors, straight_line_col_checker, dist_thresh):
        for n in self.prm.nodes:
            dists, inds = self.node_kd_tree.query(
                n, k=num_neighbors, p=2, distance_upper_bound=dist_thresh
            )
            # the first nearest neighbor is always the point itself
            for i, neighbor_ind in enumerate(inds[1:]):
                dist_is_inf = dists[i + 1] == np.inf
                if dists[i + 1] != np.inf:
                    has_collision = (
                        straight_line_col_checker.straight_line_has_collision(
                            np.array(n), self.node_kd_tree.data[neighbor_ind]
                        )
                    )
                    if not has_collision:
                        self.prm.add_edge(
                            n, tuple(self.node_kd_tree.data[neighbor_ind])
                        )

    def draw_tree(
        self,
        vis_bundle: VisualizationBundle,
        body: RigidBody,
        prefix="prm",
        options=vis_utils.TrajectoryVisualizationOptions(),
    ):
        vis_bundle.meshcat_instance.Delete(prefix)
        for idx, (s0, s1) in enumerate(self.prm.edges()):
            vis_utils.visualize_s_space_segment(
                vis_bundle,
                np.array(s0),
                np.array(s1),
                body,
                f"{prefix}/seg_{idx}",
                options,
            )


class PRMFixedEdges(PRM):
    def __init__(
        self,
        node_sampling_fun,  # generate a random sample. It does not need to be collision free.
        num_edges,
        straight_line_col_checker,
        dist_thresh=0.1,
        num_neighbors=5,
        max_it=int(1e4),
        initial_points=None,
    ):
        super().__init__(
            node_sampling_fun,  # generate a random sample. It does not need to be collision free.
            num_edges,
            straight_line_col_checker,
            dist_thresh,
            num_neighbors,
            max_it,
            initial_points,
        )

    def _build(
        self,
        node_sampling_fun,
        straight_line_col_checker,
        num_edges,
        max_it,
        num_neighbors,
        dist_thresh,
    ):
        self.add_k_edges(
            num_edges,
            node_sampling_fun,
            straight_line_col_checker,
            max_it,
            num_neighbors,
            dist_thresh,
        )

    def add_k_edges(
        self,
        k,
        node_sampling_fun,
        straight_line_col_checker,
        max_it,
        num_neighbors,
        dist_thresh,
    ):
        node_sample_stride = max(1, int(np.sqrt(k)))
        edges_added = 0
        while edges_added < k:
            cur_num_edges = len(self.prm.edges)
            self._sample_nodes(
                node_sampling_fun, straight_line_col_checker, node_sample_stride, max_it
            )
            if len(self.prm.nodes) < 2:
                while len(self.prm.nodes) < 2:
                    self._sample_nodes(
                        node_sampling_fun,
                        straight_line_col_checker,
                        node_sample_stride,
                        max_it,
                    )
            self.node_kd_tree = cKDTree(self.prm.nodes)
            self._connect_nodes(
                num_neighbors,
                straight_line_col_checker,
                dist_thresh,
                num_edges=len(self.prm.edges) + k - edges_added,
            )
            edges_added += len(self.prm.edges) - cur_num_edges
            print(f"PRM has {len(self.prm.nodes)} nodes")
            print(f"PRM has {len(self.prm.edges)} edges")

    def _connect_nodes(
        self, num_neighbors, straight_line_col_checker, dist_thresh, num_edges=0
    ):
        for n in self.prm.nodes:
            dists, inds = self.node_kd_tree.query(
                n, k=num_neighbors, p=2, distance_upper_bound=dist_thresh
            )
            # the first nearest neighbor is always the point itself
            for i, neighbor_ind in enumerate(inds[1:]):
                dist_is_inf = dists[i + 1] == np.inf
                if dists[i + 1] != np.inf:
                    has_collision = (
                        straight_line_col_checker.straight_line_has_collision(
                            np.array(n), self.node_kd_tree.data[neighbor_ind]
                        )
                    )
                    if not has_collision:
                        self.prm.add_edge(
                            n, tuple(self.node_kd_tree.data[neighbor_ind])
                        )
                        if num_edges > 0 and len(self.prm.edges()) >= num_edges:
                            return


class Node:
    def __init__(self, pos, cost, parent=None):
        self.pos = pos
        self.parent = parent
        self.cost = cost
        self.children = []
        self.id = None


class PRM_old:
    def __init__(
        self,
        limits,
        num_points,
        col_func_handle,
        num_neighbours=5,
        dist_thresh=0.1,
        num_col_checks=10,
        max_it=1e4,
        initial_points=None,
        verbose=False,
        plotcallback=None,
        sample_via_gaussian=False,
        gaussian_sample_var=0.1,
        node_sampling_function=None,
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

        self.node_sampling_function = (
            self.sample_node_pos
            if node_sampling_function is None
            else node_sampling_function
        )

        # generate n samples using rejection sampling
        nodes = [] if initial_points is None else initial_points
        for idx in range(num_points - len(nodes)):
            if sample_via_gaussian:
                nodes.append(self.sample_via_gaussian(nodes, var=gaussian_sample_var))
            else:
                nodes.append(self.node_sampling_function(MAXIT=max_it))
            if self.verbose and idx % 30 == 0:
                print("[PRM] Samples", idx)
        self.nodes_list = nodes
        self.nodes = np.array(nodes)
        self.nodes_kd = cKDTree(self.nodes)

        # generate edges
        self.adjacency_list, self.dist_adj = self.connect_nodes()
        self.make_start_end_pairs()
        self.plot()

    def sample_via_gaussian(self, cur_nodes, var=0.1, collision_free=True, MAXIT=1e4):
        """
        randomly select a node and sample from a gaussian around that node
        """
        ind = np.random.randint(0, len(cur_nodes))
        mean = cur_nodes[ind]
        ctr = 0
        while ctr < MAXIT:
            pos_samp = mean + var * np.random.randn(len(self.min_pos))
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

    def connect_nodes(
        self,
    ):
        adjacency_list = []
        dist_adj = []
        for node_idx in range(self.nodes.shape[0]):
            if self.verbose and node_idx % 20 == 0:
                print("[PRM] Nodes connected:", node_idx)
            edges = []
            edge_dist = []
            dists, idxs = self.nodes_kd.query(
                self.nodes[node_idx, :],
                k=self.num_neighbours,
                p=2,
                distance_upper_bound=self.dist_thresh,
            )
            # linesearch connection for collision
            for step in range(len(idxs)):
                nearest_idx = idxs[step]
                if not dists[step] == np.inf:
                    add = True
                    for t in self.t_check:
                        pos = (1 - t) * self.nodes[node_idx, :] + t * self.nodes[
                            nearest_idx, :
                        ]
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
            dists, idxs = self.nodes_kd.query(
                self.nodes[node_idx, :],
                k=self.num_neighbours * 5,
                p=2,
                distance_upper_bound=self.dist_thresh * 2,
            )
            # linesearch connection for collision
            for step in range(len(idxs)):
                nearest_idx = idxs[step]
                if not dists[step] == np.inf:
                    add = True
                    for t in self.t_check:
                        pos = (1 - t) * self.nodes[node_idx, :] + t * self.nodes[
                            nearest_idx, :
                        ]
                        if self.in_collision(pos):
                            add = False
                            break
                    if add:
                        edges.append(nearest_idx)
                        edge_dist.append(dists[step])
            self.adjacency_list.append(edges)
            self.dist_adj.append(edge_dist)

    def build_adjacency_mat(
        self,
    ):
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
        dist, pred = dijkstra(
            ad_mat, directed=False, indices=-2, return_predecessors=True
        )
        print(
            f"{len(np.argwhere(pred == -9999))} disconnected nodes",
            np.argwhere(pred == -9999),
        )
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
        tmp = PRM(
            limits,
            len(self.nodes_list),
            self.in_collision,
            self.num_neighbours,
            self.dist_thresh,
            len(self.t_check),
            self.verbose,
            None,
        )
        tmp.nodes_list = self.nodes_list
        tmp.nodes = self.nodes
        tmp.nodes_kd = self.nodes_kd
        tmp.adjacency_list = self.adjacency_list
        tmp.dist_adj = self.dist_adj
        tmp.plotcallback = self.plotcallback
        return tmp


class RRT:
    def __init__(
        self,
        start_pos,
        end_pos,
        lower_limits,
        upper_limits,
        straight_line_col_checker: StraightLineCollisionChecker,
        do_build_max_iter=-1,
    ):
        """
        start_pos: start of the rrt
        end_pos: end of the rrt
        lower_limits]upper limits: search in the box [lower_limits, upper_limits]
        collision_check_handle: return True if the configuration is in collision, false otherwise.
        """
        self.tree = nx.Graph()
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_node = self.tree.add_node(start_pos)

        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

        self.straight_line_col_checker = straight_line_col_checker
        if do_build_max_iter > 0:
            self.build_tree(do_build_max_iter)

    def get_nearest_node(self, pos):
        nearest_node = None
        nearest_distance = np.inf
        for node in self.tree.nodes():
            if (dist := np.linalg.norm(node - pos)) < nearest_distance:
                nearest_node = node
                nearest_distance = dist
        return nearest_node, nearest_distance

    def get_random_node(self):
        if np.random.rand() > 0.1:
            pos = np.random.uniform(self.lower_limits, self.upper_limits)
            while self.straight_line_col_checker.in_collision_handle(pos):
                pos = np.random.uniform(self.lower_limits, self.upper_limits)
        else:
            pos = np.array(self.end_pos)
        return pos

    def add_node(self, pos, bisection_tol=1e-5):
        nearest_node, nearest_neighbor_dist = self.get_nearest_node(pos)
        nearest_nod_arr = np.array(nearest_node)
        # run bisection search to extend as far as possible in this direction
        t_upper_bound = 1
        t_lower_bound = 0
        max_extend = False
        if self.straight_line_col_checker.straight_line_has_collision(
            pos, nearest_nod_arr
        ):
            while t_upper_bound - t_lower_bound > bisection_tol:
                t = (t_upper_bound + t_lower_bound) / 2
                cur_end = (1 - t) * nearest_nod_arr + t * pos
                if self.straight_line_col_checker.straight_line_has_collision(
                    cur_end, nearest_nod_arr
                ):
                    t_upper_bound = t
                else:
                    t_lower_bound = t
        else:
            cur_end = pos
            max_extend = True
        new_node = tuple(cur_end)
        self.tree.add_node(new_node)
        self.tree.add_edge(
            nearest_node, new_node, weight=np.linalg.norm(nearest_nod_arr - cur_end)
        )
        return max_extend

    def add_new_random_node(self, bisection_tol=1e-5):
        pos = self.get_random_node()
        return self.add_node(pos, bisection_tol)

    def build_tree(self, max_iter=int(1e4), bisection_tol=1e-5):
        for i in tqdm(range(max_iter)):
            self.add_new_random_node(bisection_tol)
            if self.end_pos in self.tree.nodes():
                return True
        return False

    def draw_start_and_end(
        self,
        vis_bundle: VisualizationBundle,
        body,
        prefix="rrt",
        options=DrawTreeOptions(),
    ):
        start_name = f"{prefix}/start"
        s = self.start_pos
        visualize_body_at_s(
            vis_bundle,
            body,
            s,
            start_name,
            options.start_end_radius,
            options.start_color,
        )

        end_name = f"{prefix}/end"
        s = self.end_pos
        visualize_body_at_s(
            vis_bundle, body, s, end_name, options.start_end_radius, options.end_color
        )

    def draw_tree(self, vis_bundle, body, prefix="rrt", options=DrawTreeOptions()):
        vis_bundle.meshcat_instance.Delete(prefix)
        traj_options = vis_utils.TrajectoryVisualizationOptions(
            start_size=options.start_end_radius,
            start_color=options.edge_color,
            end_size=options.start_end_radius,
            end_color=options.edge_color,
            path_color=options.edge_color,
            path_size=options.path_size,
            num_points=options.num_points,
        )
        for idx, (s0, s1) in enumerate(self.tree.edges()):
            vis_utils.visualize_s_space_segment(
                vis_bundle,
                np.array(s0),
                np.array(s1),
                body,
                f"{prefix}/seg_{idx}",
                traj_options,
            )
        self.draw_start_and_end(vis_bundle, body, prefix, options)


class BiRRT:
    def __init__(
        self,
        start_pos,
        end_pos,
        lower_limits,
        upper_limits,
        straight_line_col_checker: StraightLineCollisionChecker,
    ):
        """
        start_pos: start of the rrt
        end_pos: end of the rrt
        lower_limits]upper limits: search in the box [lower_limits, upper_limits]
        collision_check_handle: return True if the configuration is in collision, false otherwise.
        """
        self.tree_to_start = RRT(
            start_pos, end_pos, lower_limits, upper_limits, straight_line_col_checker
        )
        self.tree_to_end = RRT(
            end_pos, start_pos, lower_limits, upper_limits, straight_line_col_checker
        )
        self.start_pos = start_pos
        self.end_pos = end_pos

        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

        self.straight_line_col_checker = straight_line_col_checker

        self.connected_tree = None

    def add_node(self, pos, bisection_tol=1e-5):
        if np.random.rand() > 0.1:
            return self.tree_to_start.add_node(
                pos, bisection_tol
            ) and self.tree_to_end.add_node(pos, bisection_tol)
        else:
            return self.tree_to_start.add_node(
                np.array(self.end_pos), bisection_tol
            ) or self.tree_to_end.add_node(np.array(self.start_pos), bisection_tol)

    def get_random_node(self):
        pos = np.random.uniform(self.lower_limits, self.upper_limits)
        while self.straight_line_col_checker.in_collision_handle(pos):
            pos = np.random.uniform(self.lower_limits, self.upper_limits)
        return pos

    def build_tree(self, max_iter=int(1e4), bisection_tol=1e-5, verbose=True):
        for i in tqdm(range(max_iter)):
            pos = self.get_random_node()
            trees_connected = self.add_node(pos, bisection_tol)
            if trees_connected:
                self.connected_tree = nx.compose(
                    self.tree_to_start.tree, self.tree_to_end.tree
                )
                return True
        return False

    def draw_tree(
        self,
        vis_bundle: VisualizationBundle,
        body,
        prefix="bi_rrt",
        start_tree_options=DrawTreeOptions(
            edge_color=Rgba(0, 1, 0, 0.5), start_color=Rgba(0, 1, 0, 0.5)
        ),
        end_tree_options=DrawTreeOptions(
            edge_color=Rgba(1, 0, 0, 0.5), start_color=Rgba(1, 0, 0, 0.5)
        ),
        shortest_path_options=DrawTreeOptions(edge_color=Rgba(0, 0, 1, 0.5)),
    ):
        self.tree_to_start.draw_tree(
            vis_bundle, body, prefix + "/start_tree", start_tree_options
        )
        self.tree_to_end.draw_tree(
            vis_bundle, body, prefix + "/end_tree", end_tree_options
        )
        self.tree_to_start.draw_start_and_end(
            vis_bundle, body, prefix + "/shortest_path", start_tree_options
        )
        if self.connected_tree is not None:
            self.draw_start_target_path(vis_bundle, body, prefix, shortest_path_options)

    def draw_start_target_path(
        self,
        vis_bundle: VisualizationBundle,
        body,
        prefix="bi_rrt",
        options=DrawTreeOptions(),
    ):
        if self.connected_tree is None:
            raise ValueError("This Bi-RRT is not connected")
        path = nx.dijkstra_path(self.connected_tree, self.start_pos, self.end_pos)
        edges = zip(path[:-1], path[1:])
        traj_options = vis_utils.TrajectoryVisualizationOptions(
            start_size=options.start_end_radius,
            start_color=options.edge_color,
            end_size=options.start_end_radius,
            end_color=options.edge_color,
            path_color=options.edge_color,
            path_size=options.path_size,
            num_points=options.num_points,
        )
        for idx, (s0, s1) in enumerate(edges):
            vis_utils.visualize_s_space_segment(
                vis_bundle,
                np.array(s0),
                np.array(s1),
                body,
                f"{prefix}/path_seg_{idx}",
                traj_options,
            )
        self.tree_to_start.draw_start_and_end(vis_bundle, body, prefix, options)
