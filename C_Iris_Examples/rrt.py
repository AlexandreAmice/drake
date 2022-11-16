import numpy as np

class Node:
    def __init__(self, pos, parent = None):
        self.pos = pos
        self.parent = parent
        self.children = []
        self.id = None


class RRT:
    def __init__(self, 
                 start, 
                 goal, 
                 limits, 
                 col_func_handle = None, 
                 max_extend_length = 1e-1,
                 extend_steps = 1e-2, 
                 init_goal_sample_rate = 0.05,
                 goal_sample_rate_scaler = 0.1,
                 verbose = False,
                 plotcallback = None,
                 sample_collision_free = True
                 ):

        #col_check(pos) == True -> in collision!
        self.in_collision = col_func_handle
        self.check_col = False if self.in_collision == None else True
        
        if self.check_col and self.in_collision(start):
            print("[RRT ERROR] Start position is in collision")
            raise NotImplementedError

        if self.check_col and self.in_collision(goal):
            print("[RRT ERROR] Goal position is in collision")
            raise NotImplementedError

        root = Node(start)
        root.id = 0
        self.nodes = [root]
        self.node_pos = [self.nodes[0].pos] 
        self.start = start
        self.goal = goal
        self.limits = limits
        self.min_pos = self.limits[0]
        self.max_pos = self.limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 

        self.init_goal_sample_rate = init_goal_sample_rate
        self.goal_sample_rate = init_goal_sample_rate
        self.goal_sample_rate_scaler = goal_sample_rate_scaler
        self.max_extend_length = max_extend_length
        self.extend_step_size = extend_steps
        self.max_extend_steps = int(max_extend_length/self.extend_step_size)
        self.verbose = verbose
        self.plotcallback = plotcallback
        self.do_plot = True if self.plotcallback is not None else False
        self.sample_collision_free =sample_collision_free
        if not self.check_col:
            self.sample_collision_free = False
        self.dim = len(start)
        self.distance_to_go = 1e9

       

    def get_closest_node(self, pos):
        dists = np.linalg.norm(pos.reshape(-1, self.dim) - np.array(self.node_pos), axis=1)
        id_min = np.argmin(dists)
        return id_min

    def sample_node_pos(self, collision_free = True, MAXIT = 1e4):
        rand = np.random.rand()
        if rand < self.goal_sample_rate:
            return self.goal
        
        rand = np.random.rand(self.dim)
        pos_samp = self.min_pos + rand*self.min_max_diff 
        if self.sample_collision_free:
            good_sample = not self.in_collision(pos_samp) if collision_free else True
        else: 
            good_sample = True 

        it = 0
        while not good_sample and it < MAXIT:
            rand = np.random.rand(self.dim)
            pos_samp = self.min_pos + rand*self.min_max_diff 
            good_sample = not self.in_collision(pos_samp)
            it+=1

        if not good_sample:
            print("[RRT ERROR] Could not find collision free point in MAXIT")
            raise NotImplementedError
        return pos_samp

    def grow_naive(self, parent, pos_samp):
        dir_raw = pos_samp - parent.pos
        dir_norm = np.linalg.norm(dir_raw)
        dir = np.divide(dir_raw, dir_norm + 1e-9)

        steps = 1
        child_pos = parent.pos

        if self.check_col:
            while steps < self.max_extend_steps:
                #check point one step ahead
                extend = steps * self.extend_step_size
                is_past_samplepoint = extend > dir_norm
                check = parent.pos + dir * extend

                if self.in_collision(check) or is_past_samplepoint:
                   # if is in collision return last position that wasnt 
                   child = Node(child_pos, parent)
                   return child

                child_pos = check
                steps +=1

            child = Node(child_pos, parent)
            parent.children.append(child)
            return child
            
        else:
            child = Node(child_pos + np.min([self.max_extend_length, dir_norm])*dir, parent)
            parent.children.append(child)
            return child

    def run(self, n_it):

        for it in range(n_it):
            if it%1000 == 0:
                print(it)
            pos_samp = self.sample_node_pos()
            nearest_id = self.get_closest_node(pos_samp) 
            parent_node = self.nodes[nearest_id]
            child_node = self.grow_naive(parent_node, pos_samp)
            child_node.id = it + 1
            self.nodes.append(child_node)
            self.node_pos.append(child_node.pos)
            dist_to_target = np.linalg.norm(self.goal - self.node_pos[-1])

            if self.do_plot:
                self.plotcallback(parent_node, child_node, pos_samp, it)

            if dist_to_target<self.distance_to_go:
                self.distance_to_go = dist_to_target
                self.closest_id = child_node.id
                self.goal_sample_rate = np.clip(0.8 - self.distance_to_go*self.goal_sample_rate_scaler, a_min = self.init_goal_sample_rate, a_max = 1)
                if self.verbose:
                    print("it: {iter} distance to target: {dist: .3f} goalsample prob: {prob: .3f}".format(iter =it, dist = self.distance_to_go, prob = self.goal_sample_rate))

            if self.distance_to_go< self.extend_step_size:
                break

        #walk back through tree to get closest node    
        path = []
        current_node = self.nodes[self.closest_id]
        while current_node.parent is not None:
            current_node = current_node.parent
            path.append(current_node.pos)
        
        if self.distance_to_go <= self.extend_step_size:
            print('[RRT] Collision free path found in ', it,' steps')
            return True, path[::-1]
        else:
            print('[RRT] Could not find path in ', it,' steps')
            return False, path[::-1]