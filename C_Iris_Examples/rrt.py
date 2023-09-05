from scipy.spatial import KDTree
import networkx as nx
from pydrake.all import Sphere, Rgba
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from visualization_utils import visualize_body_at_s, visualize_task_space_trajectory, VisualizationBundle


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


class RRT:
    def __init__(self, start_pos, end_pos, lower_limits,
                 upper_limits, straight_line_col_checker: StraightLineCollisionChecker,
                 do_build_max_iter=-1):
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

    def get_random_node(self, bisection_tol=1e-5):
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
        v = pos - nearest_node
        t_upper_bound = 1
        t_lower_bound = 0
        max_extend = False
        if self.straight_line_col_checker.straight_line_has_collision(pos, nearest_nod_arr):
            while t_upper_bound - t_lower_bound > bisection_tol:
                t = (t_upper_bound + t_lower_bound) / 2
                cur_end = (1 - t) * nearest_nod_arr + t * pos
                if self.straight_line_col_checker.straight_line_has_collision(cur_end, nearest_nod_arr):
                    t_upper_bound = t
                else:
                    t_lower_bound = t
        else:
            cur_end = pos
            max_extend = True
        new_node = tuple(cur_end)
        self.tree.add_node(new_node)
        self.tree.add_edge(nearest_node, new_node, weight=np.linalg.norm(nearest_nod_arr - cur_end))
        return max_extend

    def add_new_random_node(self, bisection_tol=1e-5):
        pos = self.get_random_node(bisection_tol)
        return add_node(pos, bisection_tol)

    def build_tree(self, max_iter=int(1e4), bisection_tol=1e-5, verbose=True):
        for i in tqdm(range(max_iter)):
            self.add_new_random_node(bisection_tol)
            if self.end_pos in self.tree.nodes():
                return True
        return False

    def draw_start_and_end(self, vis_bundle: VisualizationBundle, body, prefix="rrt", options=DrawTreeOptions()):
        start_name = f"{prefix}/start"
        s = self.start_pos
        visualize_body_at_s(vis_bundle, body, s, start_name, options.start_end_radius, options.start_color)

        end_name = f"{prefix}/end"
        s = self.end_pos
        visualize_body_at_s(vis_bundle, body, s, end_name, options.start_end_radius, options.end_color)

    def draw_tree(self, vis_bundle, body, prefix="rrt", options=DrawTreeOptions()):
        vis_bundle.meshcat_instance.Delete(prefix)
        for idx, (s0, s1) in enumerate(self.tree.edges()):
            visualize_task_space_trajectory(vis_bundle, s0, s1, body, color=options.edge_color,
                                            path_size=options.path_size, num_points=options.num_points,
                                            path=f"{prefix}/seg_{idx}")
        self.draw_start_and_end(vis_bundle, body, prefix, options)


class BiRRT:
    def __init__(self, start_pos, end_pos, lower_limits,
                 upper_limits, straight_line_col_checker: StraightLineCollisionChecker):
        """
        start_pos: start of the rrt
        end_pos: end of the rrt
        lower_limits]upper limits: search in the box [lower_limits, upper_limits]
        collision_check_handle: return True if the configuration is in collision, false otherwise.
        """
        self.tree_to_start = RRT(start_pos, end_pos, lower_limits,
                                 upper_limits, straight_line_col_checker)
        self.tree_to_end = RRT(end_pos, start_pos, lower_limits,
                               upper_limits, straight_line_col_checker)
        self.start_pos = start_pos
        self.end_pos = end_pos

        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

        self.straight_line_col_checker = straight_line_col_checker

        self.connected_tree = None

    def add_node(self, pos, bisection_tol=1e-5):
        if np.random.rand() > 0.1:
            return (self.tree_to_start.add_node(pos, bisection_tol) and
                    self.tree_to_end.add_node(pos, bisection_tol))
        else:
            return (self.tree_to_start.add_node(np.array(self.end_pos), bisection_tol) or
                    self.tree_to_end.add_node(np.array(self.start_pos), bisection_tol))

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
                self.connected_tree = nx.compose(self.tree_to_start.tree, self.tree_to_end.tree)
                return True
        return False

    def draw_tree(self, vis_bundle: VisualizationBundle, body, prefix="bi_rrt",
                  start_tree_options=DrawTreeOptions(edge_color=Rgba(0, 1, 0, 0.5),
                                                     start_color=Rgba(0, 1, 0, 0.5)),
                  end_tree_options=DrawTreeOptions(edge_color=Rgba(1, 0, 0, 0.5),
                                                   start_color=Rgba(1, 0, 0, 0.5)),
                  shortest_path_options=DrawTreeOptions(edge_color=Rgba(0, 0, 1, 0.5))):
        self.tree_to_start.draw_tree(vis_bundle, body,
                                     prefix + "/start_tree",
                                     start_tree_options)
        self.tree_to_end.draw_tree(vis_bundle, body,
                                   prefix + "/end_tree",
                                   end_tree_options)
        self.tree_to_start.draw_start_and_end(vis_bundle, body, prefix + "/shortest_path", start_tree_options)
        if self.connected_tree is not None:
            self.draw_start_target_path(vis_bundle, body, prefix, shortest_path_options)

    def draw_start_target_path(self, vis_bundle: VisualizationBundle, body, prefix="bi_rrt",
                               options=DrawTreeOptions()):
        if self.connected_tree is None:
            raise ValueError("This Bi-RRT is not connected")
        path = nx.dijkstra_path(self.connected_tree, self.start_pos, self.end_pos)
        edges = zip(path[:-1], path[1:])
        for idx, (s0, s1) in enumerate(edges):
            visualize_task_space_trajectory(vis_bundle, s0, s1, body, color=options.edge_color,
                                            path_size=options.path_size, num_points=options.num_points,
                                            path=f"{prefix}/path_seg_{idx}")
        self.tree_to_start.draw_start_and_end(vis_bundle, body, prefix, options)
