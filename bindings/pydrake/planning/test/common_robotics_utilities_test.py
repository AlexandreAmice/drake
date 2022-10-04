import unittest
import numpy as np

from pydrake.common import RandomGenerator
from pydrake.planning.common_robotics_utilities import (
    MakeKinematicLinearRRTNearestNeighborsFunction,
    MakeKinematicLinearBiRRTNearestNeighborsFunction,
    MakeRRTTimeoutTerminationFunction,
    MakeBiRRTTimeoutTerminationFunction,
    PropagatedState,
    RRTPlanSinglePath,
    BiRRTPlanSinglePath,
    SimpleRRTPlannerState,
    SimpleRRTPlannerTree,
    Graph,
    GrowRoadMap,
    UpdateRoadMapEdges,
    QueryPath,
    LazyQueryPath,
    # ResamplePath,
)


class TestCommonRoboticsUtilities(unittest.TestCase):
    def test_rrt(self):
        start = np.array([0.5, 0.5])
        goal = np.array([2.5, 0.5])
        goal_bias = 0.05
        step_size = 0.1
        check_step = 0.01
        solve_timeout = 2

        rrt_tree = SimpleRRTPlannerTree([SimpleRRTPlannerState(start)])

        def sampling_fn():
            if (np.random.rand() < goal_bias):
                return goal
            return np.random.rand(2) * 3

        def distance_fn(point1, point2):
            return np.linalg.norm(point2 - point1)

        def check_goal_fn(sample):
            return np.linalg.norm(sample - goal) < 1e-6

        def extend_fn(nearest, sample):
            if (sample[0] >= 1 and sample[0] <= 2 and sample[1] <= 2):
                return []

            extend = None
            extend_dist = distance_fn(nearest, sample)
            if extend_dist <= step_size:
                extend = sample
            else:
                extend = nearest + step_size/extend_dist * (sample - nearest)

            check_dist = distance_fn(nearest, extend)
            for ii in range(1, int(check_dist/check_step)):
                check_point = nearest \
                    + ii * check_step / check_dist * (extend - nearest)
                if (check_point[0] >= 1 and check_point[0] <= 2
                        and check_point[1] <= 2):
                    return []
            return [PropagatedState(state=extend, relative_parent_index=-1)]

        nearest_neighbor_fn = MakeKinematicLinearRRTNearestNeighborsFunction(
            distance_fn=distance_fn, use_parallel=False)

        termination_fn = MakeRRTTimeoutTerminationFunction(solve_timeout)

        single_result = RRTPlanSinglePath(
            tree=rrt_tree, sampling_fn=sampling_fn,
            nearest_neighbor_fn=nearest_neighbor_fn,
            forward_propagation_fn=extend_fn, state_added_callback_fn=None,
            check_goal_reached_fn=check_goal_fn, goal_reached_callback_fn=None,
            termination_check_fn=termination_fn)

        print(single_result.Path())

    def test_birrt(self):
        seed = 0
        start = np.array([0.5, 0.5])
        goal = np.array([2.5, 0.5])
        goal_bias = 0.05
        step_size = 0.1
        check_step = 0.01
        solve_timeout = 2

        start_tree = SimpleRRTPlannerTree([SimpleRRTPlannerState(start)])
        goal_tree = SimpleRRTPlannerTree([SimpleRRTPlannerState(goal)])

        def sampling_fn():
            return np.random.rand(2) * 3

        def distance_fn(point1, point2):
            return np.linalg.norm(point2 - point1)

        def extend_fn(nearest, sample, is_start_tree):
            if (sample[0] >= 1 and sample[0] <= 2 and sample[1] <= 2):
                return []

            extend = None
            extend_dist = distance_fn(nearest, sample)
            if extend_dist <= step_size:
                extend = sample
            else:
                extend = nearest + step_size/extend_dist * (sample - nearest)

            check_dist = distance_fn(nearest, extend)
            for ii in range(1, int(check_dist/check_step)):
                check_point = nearest \
                    + ii * check_step / check_dist * (extend - nearest)
                if (check_point[0] >= 1 and check_point[0] <= 2
                        and check_point[1] <= 2):
                    return []
            return [PropagatedState(state=extend, relative_parent_index=-1)]

        def connect_fn(nearest, sample, is_start_tree):
            if (sample[0] >= 1 and sample[0] <= 2 and sample[1] <= 2):
                return []

            total_dist = distance_fn(nearest, sample)
            total_steps = int(np.ceil(total_dist / step_size))

            propagated_states = []
            parent_offset = -1
            current = nearest
            for steps in range(total_steps):
                current_target = None
                target_dist = distance_fn(current, sample)
                if (target_dist > step_size):
                    current_target = current \
                        + step_size/target_dist * (sample - current)
                elif (target_dist < 1e-6):
                    break
                else:
                    current_target = sample

                check_dist = distance_fn(current, current_target)
                for ii in range(1, int(check_dist/check_step)):
                    check_point = current + ii * check_step / check_dist \
                        * (current_target - current)
                    if (check_point[0] >= 1 and check_point[0] <= 2
                            and check_point[1] <= 2):
                        return propagated_states
                propagated_states.append(PropagatedState(
                    state=current_target, relative_parent_index=parent_offset))
                parent_offset += 1
                current = current_target

            return propagated_states

        def states_connected_fn(source, target, is_start_tree):
            return np.linalg.norm(source - target) < 1e-6

        nearest_neighbor_fn = MakeKinematicLinearBiRRTNearestNeighborsFunction(
            distance_fn=distance_fn, use_parallel=False)

        termination_fn = MakeBiRRTTimeoutTerminationFunction(solve_timeout)

        extend_result = BiRRTPlanSinglePath(
            start_tree=start_tree, goal_tree=goal_tree,
            state_sampling_fn=sampling_fn,
            nearest_neighbor_fn=nearest_neighbor_fn,
            propagation_fn=extend_fn,
            state_added_callback_fn=None,
            states_connected_fn=states_connected_fn,
            goal_bridge_callback_fn=None,
            tree_sampling_bias=0.5, p_switch_tree=0.25,
            termination_check_fn=termination_fn,
            uniform_unit_real_fn=np.random.rand)

        print(extend_result.Path())

        connect_result = BiRRTPlanSinglePath(
            start_tree=start_tree, goal_tree=goal_tree,
            state_sampling_fn=sampling_fn,
            nearest_neighbor_fn=nearest_neighbor_fn,
            propagation_fn=connect_fn,
            state_added_callback_fn=None,
            states_connected_fn=states_connected_fn,
            goal_bridge_callback_fn=None,
            tree_sampling_bias=0.5, p_switch_tree=0.25,
            termination_check_fn=termination_fn,
            uniform_unit_real_fn=np.random.rand)

        # print(connect_result.Path())

    def test_prm(self):
        np.random.seed(42)
        test_env = np.array(["####################",
                             "#                  #",
                             "#  ####            #",
                             "#  ####    #####   #",
                             "#  ####    #####   #",
                             "#          #####   #",
                             "#          #####   #",
                             "#                  #",
                             "#      #########   #",
                             "#     ##########   #",
                             "#    ###########   #",
                             "#   ############   #",
                             "#                  #",
                             "#                  #",
                             "#    ##            #",
                             "#    ##   ######## #",
                             "#    ##   ######## #",
                             "#    ##   ######## #",
                             "#                  #",
                             "####################"])

        test_env_shape = [len(test_env[0]), len(test_env)]

        K = 5
        roadmap_size = 100

        def roadmap_termination_fn(current_roadmap_size):
            return current_roadmap_size >= roadmap_size

        def state_sampling_fn():
            x = np.random.randint(test_env_shape[0])
            y = np.random.randint(test_env_shape[1])

            return np.array([x, y])

        def distance_fn(start, end):
            return np.linalg.norm(end - start)

        def check_state_validity_fn(point):
            x, y = point
            return test_env[int(y)][int(x)] != '#'

        def check_edge_validity_fn(start, end):
            def checkEdgeCollisionFree(start, end, stepsize):
                num_steps = np.ceil(distance_fn(start, end)/stepsize)

                for step in range(int(num_steps)+1):
                    interpolation_ratio = step / num_steps
                    interpolated_point = start + np.round(
                        interpolation_ratio*(end-start))

                    if not check_state_validity_fn(interpolated_point):
                        return False

                return True

            return (checkEdgeCollisionFree(start, end, 0.5)
                    and checkEdgeCollisionFree(end, start, 0.5))

        # for plan checking
        def setCell(env, point, char):
            x, y = point
            x, y = int(x), int(y)
            env[y] = env[y][:x] + char + env[y][x+1:]

        def getCell(env, point):
            x, y = point
            return env[int(y)][int(x)]

        # def ResamplePoints(points):
        #     return ResamplePath(points, 0.5, distance_fn, interpolate_point)

        def drawEnvironment(env):
            print("".join(list(map(lambda row: row + "\n", env))))

        # def drawRoadmap(roadmap):
        #     tmp_env = test_env.copy()

        #     roadmap_nodes = roadmap.GetNodesImmutable()

        #     for roadmap_node in roadmap_nodes:
        #         out_edges = roadmap_node.GetOutEdgesImmutable()
        #         for edge in out_edges:
        #             self_point = roadmap.GetNodeImmutable(
        #                 edge.GetFromIndex()).GetValueImmutable()
        #             other_point = roadmap.GetNodeImmutable(
        #                 edge.GetToIndex()).GetValueImmutable()
        #             edge_path = ResamplePoints([self_point, other_point])

        #             for point in edge_path:
        #                 current_val = getCell(tmp_env, point)
        #                 if current_val != '+':
        #                     setCell(tmp_env, point, '-')
        #             setCell(tmp_env, self_point, '+')
        #             setCell(tmp_env, other_point, '+')

        #     drawEnvironment(tmp_env)

        def drawPath(env, starts, goals, path):
            tmp_env = env.copy()
            # if len(path) > 0:
            #     setCell(tmp_env, path[0], '+')

            #     for previous, current in zip(path[:-1], path[1:]):
            #         edge_path = ResamplePoints([previous, current])

            #         for point in edge_path:
            #             current_val = getCell(tmp_env, point)
            #             if current_val != '+':
            #                 setCell(tmp_env, point, '-')

            #         setCell(tmp_env, current, '+')

            for p in path:
                setCell(tmp_env, p, '+')
            for start in starts:
                setCell(tmp_env, start, 'S')
            for goal in goals:
                setCell(tmp_env, goal, 'G')

            drawEnvironment(tmp_env)

        def check_path(path):
            self.assertTrue(len(path) >= 2)

            for idx in range(1, len(path)):
                # We check both forward and backward because rounding in the
                # waypoint interpolation can create edges that are valid in
                # only one direction.

                forward_valid = check_edge_validity_fn(path[idx-1], path[idx])
                backward_valid = check_edge_validity_fn(path[idx], path[idx-1])

                edge_valid = forward_valid and backward_valid

                self.assertTrue(edge_valid)

        def checkPlan(starts, goals, path):
            drawPath(test_env, starts, goals, path)

            print("Checking raw path")
            check_path(path)

            # TODO
            """
            smoothed_path = SmoothWaypoints(path, check_edge_validity_fn, prng)
            print("Checking smoothed path")
            check_path(smoothed_path)

            ResampleWaypoints(smoothed_path);
            print("Checking resampled path")
            check_path(resampled_path);
            """

        roadmap = Graph()

        GrowRoadMap(roadmap, state_sampling_fn, distance_fn,
                    check_state_validity_fn, check_edge_validity_fn,
                    roadmap_termination_fn, K, False, True, False)
        self.assertTrue(roadmap.CheckGraphLinkage())

        UpdateRoadMapEdges(roadmap, check_edge_validity_fn, distance_fn, False)
        self.assertTrue(roadmap.CheckGraphLinkage())

        nodes_to_prune = {10, 20, 30, 40, 50, 60}
        serial_pruned_roadmap = roadmap.MakePrunedCopy(nodes_to_prune, False)
        self.assertTrue(serial_pruned_roadmap.CheckGraphLinkage())

        parallel_pruned_roadmap = roadmap.MakePrunedCopy(nodes_to_prune, True)
        self.assertTrue(parallel_pruned_roadmap.CheckGraphLinkage())

        # test planning
        keypoints = [np.array([1, 1]), np.array([18, 18]),
                     np.array([7, 13]), np.array([9, 5])]

        for start in keypoints:
            for goal in keypoints:
                if np.array_equal(start, goal):
                    continue

                print(f"PRM Path ({start} to {goal})")
                path = QueryPath([start], [goal], roadmap, distance_fn,
                                 check_edge_validity_fn, K,
                                 use_parallel=False,
                                 connection_is_symmetric=True,
                                 add_duplicate_states=False,
                                 limit_astar_pqueue_duplicates=True).Path()
                checkPlan([start], [goal], path)

                print(f"Lazy-PRM Path ({start} to {goal})")

                lazy_path = LazyQueryPath(
                    [start], [goal], roadmap, distance_fn,
                    check_edge_validity_fn, K, use_parallel=False,
                    connection_is_symmetric=True, add_duplicate_states=False,
                    limit_astar_pqueue_duplicates=True).Path()

                checkPlan([start], [goal], lazy_path)

        starts = [keypoints[0], keypoints[1]]
        goals = [keypoints[2], keypoints[3]]
        print(f"Multi start/goal PRM Path ({starts} to {goals})")
        multi_path = QueryPath(starts, goals, roadmap, distance_fn,
                               check_edge_validity_fn, K, use_parallel=False,
                               connection_is_symmetric=True,
                               add_duplicate_states=False,
                               limit_astar_pqueue_duplicates=True).Path()

        checkPlan(starts, goals, multi_path)
