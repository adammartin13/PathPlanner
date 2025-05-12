import numpy as np
import heapq
import math
import random
import time
from functools import lru_cache
from collections import deque
from typing import List, Tuple, Optional


directions = [  # Precomputed, reusable
    (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]

def get_legal_actions(world, player):
    rows, cols = world.shape
    x, y = player
    actions = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            if world[nx][ny] == 0:  # Valid move
                actions.append((dx, dy))

    return [np.array((dx, dy)) for dx, dy in actions]

# Calculate heuristic using the diagonal distance technique
def heuristic(start, end):
    # H = max(|x - end_x|, |y - end_y|)
    dx = abs(start[0] - end[0])
    dy = abs(start[1] - end[1])
    return max(dx, dy)

def apply_position(action, player):
    position = player.copy()
    position[0] += action[0]
    position[1] += action[1]
    return position

# Simulate pursuers actions for greater avoidance
def pursuer_positions(world, pursuer, depth):
    reachable = set()
    frontier = {tuple(pursuer)}

    for it in range(depth):
        next_frontier = set()
        for position in frontier:
            actions = get_legal_actions(world, position)
            for action in actions:
                new_position = tuple(apply_position(position, action))
                if new_position not in reachable:
                    reachable.add(new_position)
                    next_frontier.add(new_position)
        frontier = next_frontier

    return reachable


def check_winner(player, pursued):
    # Check if the provided player/target condition is a winning one
    return player[0] == pursued[0] and player[1] == pursued[1]


# If the target is never reachable, hit an obstacle so they don't get points
def crash_direction(world: np.ndarray, player: np.ndarray) -> np.ndarray:
    rows, cols = world.shape
    visited = set()
    queue = deque()
    start = tuple(player)
    queue.append((start, 0))
    visited.add(start)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        (x, y), dist = queue.popleft()

        # Check for in-bounds obstacles
        if 0 <= x < rows and 0 <= y < cols:
            if world[x][y] == 1:
                # return np.array([x, y])
                obstacle = np.array([x, y])
                direction = obstacle - player
                if direction[0] != 0: direction[0] = np.sign(direction[0])
                if direction[1] != 0: direction[1] = np.sign(direction[1])
                return direction

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))

    # If no obstacle is found return current position
    return np.array(player)

# If we're close to capture and there is an obstacle nearby, crash into it
def crash(world, player):
    directions = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                           [-1, -1], [-1, 1], [1, -1], [1, 1]])
    actions = get_legal_actions(world, player)
    action_tuples = [tuple(action) for action in actions]

    for direction in directions:
        if tuple(direction) not in action_tuples:
            return direction

    return np.array([0,0])  # no wall found

# Penalty for approaching obstacles
def obstacle_penalty(world: np.ndarray, position: Tuple[int, int], penalty_weight: float = 1.0) -> float:
    x, y = position
    rows, cols = world.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            if world[nx][ny] == 1:
                return penalty_weight  # Fixed penalty for adjacency

    return 0.0  # No adjacent obstacles

# A* Search
def a_star(world, current, pursued, pursuer):
    start = tuple(current)
    end = tuple(pursued)

    # f = g + h
    # open_set priority queue
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start))
    came_from = {start: None}  # backtracking for path reconstruction
    g = {start: 0}

    while open_set:
        _, current_g, current_node = heapq.heappop(open_set)  # f, g, node

        if current_node == end:  # goal reached, backtrack to determine step taken
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()  # list of nodes traveled
            if len(path) > 1:
                dx = path[1][0] - start[0]
                dy = path[1][1] - start[1]
                return np.array([dx, dy])  # return next step
            else:
                return np.array([0, 0])  # Otherwise we're at the goal, stand still

        # check where we can go next
        for action in get_legal_actions(world, current_node):
            neighbor = (current_node[0] + action[0], current_node[1] + action[1])
            temp_g = current_g + 1

            if neighbor not in g or temp_g < g[neighbor]:  # best way to reach neighbor
                danger_penalty = 5 / (heuristic(neighbor, pursuer) + 1)
                obstacle = obstacle_penalty(world, neighbor, penalty_weight=2)
                g[neighbor] = temp_g
                f = temp_g + heuristic(neighbor, end) + danger_penalty + obstacle
                heapq.heappush(open_set, (f, temp_g, neighbor))
                came_from[neighbor] = current_node

    return crash_direction(world, current)  # No path found

@lru_cache(maxsize=5000)  # Wrap A* with caching
def cached_astar(player_x, player_y, target_x, target_y, aggressor_x, aggressor_y, state_bytes):
    state = np.frombuffer(state_bytes, dtype=np.int32).reshape((30, 30))
    return a_star(state, (player_x, player_y), (target_x, target_y), (aggressor_x, aggressor_y))

# Helper function to call cached A*
def get_astar(player, target, aggressor, state):
    state_bytes = state.astype(np.int32).tobytes()
    return cached_astar(player[0], player[1], target[0], target[1], aggressor[0], aggressor[1], state_bytes)

# Returns the top k A* paths
def a_star_top(world, current, goal, maximize=False):
    k = 3
    start = tuple(current)
    end = tuple(goal)

    # f = g + h
    # open_set priority queue
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start))
    came_from = {start: None}  # backtracking for path reconstruction
    g = {start: 0}

    candidates = []

    while open_set:
        f_val, current_g, current_node = heapq.heappop(open_set)  # f, g, node

        # If this node has a previous position, we can determine the first step from start
        if came_from[current_node] is not None:
            step_from_start = (current_node[0] - start[0], current_node[1] - start[1])
            candidates.append((f_val, step_from_start))

        # check where we can go next
        for action in get_legal_actions(world, current_node):
            neighbor = (current_node[0] + action[0], current_node[1] + action[1])
            temp_g = current_g + 1
            if neighbor not in g or temp_g < g[neighbor]:
                g[neighbor] = temp_g
                f = temp_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f, temp_g, neighbor))
                came_from[neighbor] = current_node

    # Sort based on heuristic: minimize if chasing, maximize if evading
    sorted_candidates = sorted(candidates, key=lambda x: x[0], reverse=maximize)

    return [step for _, step in sorted_candidates[:k]] if sorted_candidates else [np.array([0, 0])]

@lru_cache(maxsize=5000)
def cached_k_astar(start_x, start_y, goal_x, goal_y, world_bytes, maximize):
    world = np.frombuffer(world_bytes, dtype=np.int32).reshape((30, 30))
    return tuple(map(tuple, a_star_top(world, (start_x, start_y), (goal_x, goal_y), maximize)))

def get_k_astar(world, start, goal, maximize=False):
    world_bytes = world.astype(np.int32).tobytes()
    return [np.array(step) for step in cached_k_astar(start[0], start[1], goal[0], goal[1], world_bytes, maximize=False)]


def apply_pursuer(state, player, pursuer):
    # Calculate the best move the pursuer could make
    # For all directions they could take, which is legal and minimizes distance to the player
    actions = get_legal_actions(state, pursuer)
    if not actions: return pursuer

    def score(action):
        new_pos = apply_position(action, pursuer)
        return heuristic(new_pos, player)  # minimize distance from player

    # Sort actions by score ascending (closer to player)
    sorted_actions = sorted(actions, key=score)

    # Pick randomly from top 4 (or fewer)
    top_k = sorted_actions[:min(4, len(sorted_actions))]
    chosen_action = random.choice(top_k)

    return apply_position(chosen_action, pursuer)


def apply_pursued(state, player, pursued):
    # Calculate the best move the pursued could make
    # For all directions they could take, which is best for evading us while reaching its target
    actions = get_legal_actions(state, pursued)
    if not actions: return pursued

    def score(action):
        new_pos = apply_position(action, pursued)
        return heuristic(new_pos, player)  # maximize distance from player

    # Sort actions by score descending
    sorted_actions = sorted(actions, key=score, reverse=True)

    # Pick randomly from top 4 actions (or fewer if < 4 available)
    top_k = sorted_actions[:min(4, len(sorted_actions))]
    chosen_action = random.choice(top_k)

    return apply_position(chosen_action, pursued)

# DFS search the tree for a matching state
def find_matching_node(root, state_key):
    stack = [root]
    while stack:
        node = stack.pop()
        if node.state_key == state_key:
            return node
        stack.extend(node.children)
    return None


def simulate(node, tree_node):
    # Simulate a single round with A*
    # Assume target is greedy, select randomly from k-highest A* outcomes
    # Player selects best option for selected target location
    # Aggressor selects the best option provided player location
    state_key = (tuple(node.player), tuple(node.pursued), tuple(node.pursuer))
    tree_node = find_matching_node(tree_node, state_key)

    # Check if this state has been simulated before
    if tree_node is not None:
        if tree_node.k_astar is None:  # Expected cached kA* but DNE
            tree_node.k_astar = get_k_astar(node.state, node.pursued, node.pursuer, maximize=False)

        new_pursued = random.choice(tree_node.k_astar)
        for child in tree_node.children:  # Check if child exists for this simulated node
            if np.array_equal(child.pursued, new_pursued):
                return child  # Outcome already simulated

        # Child DNE for chosen move
        new_player = apply_position(get_astar(node.player, new_pursued, node.pursuer, node.state), node.player)
        new_pursuer = apply_position(get_astar(node.pursuer, new_player, new_pursued, node.state), node.pursuer)
        return Node(node.state, new_player, new_pursued, new_pursuer, parent=tree_node)

    # Fresh node
    k_astar = get_k_astar(node.state, node.pursued, node.pursuer, maximize=False)
    new_pursued = random.choice(k_astar)
    new_player = apply_position(get_astar(node.player, new_pursued, node.pursuer, node.state), node.player)
    new_pursuer = apply_position(get_astar(node.pursuer, new_player, new_pursued, node.state), node.pursuer)
    new_node = Node(node.state, new_player, new_pursued, new_pursuer, parent=node)
    new_node.k_astar = k_astar

    return new_node


def UCT(child):
    Q = child.value
    N = child.visits + 1e-9
    c = 1.41
    Np = child.parent.visits
    return Q / N + c * math.sqrt(math.log(Np + 1) / N)

# Monte Carlo Tree Search
def mcts(root):
    simulations = 50  # SIMULATIONS = 1 FOR TESTING ONLY
    if heuristic(root.player, root.pursued) <= 5:
        simulations = max(10 * heuristic(root.player, root.pursued), 10)

    for it in range(simulations):
        node = root

        # Selection
        # print('selection')
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # Expansion
        # print('expansion')
        if not check_winner(node.player, node.pursued):
            expanded = node.expand()
            if expanded is not None:  # no expansion possible
                node = expanded

        # Simulation
        # print('simulation')
        result = node.rollout(root)

        # Backpropagation
        # print('backpropagation')
        # depth = 0
        while node is not None:
            node.visits += 1
            # node.value += result * (0.99 ** depth)  # Gamma = 0.99
            node.value += result
            node = node.parent
            # depth += 1

    chosen = root.best_child()
    return chosen.player - root.player, chosen


class Node:
    def __init__(self, state, player, pursued, pursuer, parent=None):
        self.state = state
        self.player = player
        self.pursued = pursued
        self.pursuer = pursuer
        self.parent = parent
        self.children = []
        self.state_key = (tuple(player), tuple(pursued), tuple(pursuer))
        self.k_astar = None  # cached list of possible target positions
        self.visits = 0
        self.value = 0.0

    directions = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                           [-1, -1], [-1, 1], [1, -1], [1, 1]])

    def is_fully_expanded(self):
        # Checks if every action that can be taken has been accounted for in children
        actions = get_legal_actions(self.state, self.player)
        children = []
        for child in self.children:
            children.append(tuple(child.player))

        for action in actions:
            next_player = apply_position(action, self.player)
            if tuple(next_player) not in children:
                return False
        return True

    def is_terminal(self):
        return self.player == self.pursued or self.player == self.pursuer

    def best_child(self):
        # Upper Confidence Bound for Trees
        # The sqrt has been changed as a safety against unvisited parents
        # UCT = Q / N + c * sqrt(ln(N_p + 1) / (N + 1e-6))
        # Q = total reward
        # N = number of visits
        # c = constant, typically sqrt(2)
        # N_p = number of parent visits
        UCTs = []
        for child in self.children:
            Q = child.value
            N = child.visits + 1e-9
            c = 1.41
            Np = child.parent.visits

            if N == 0:
                UCT = math.inf
            else:
                UCT = Q / N + c * math.sqrt(math.log(Np + 1) / N)
            UCTs.append((UCT, child))

        highest_UCT = max(UCTs, key=lambda x: x[0])[0]
        best_children = [child for UCT, child in UCTs if UCT == highest_UCT]
        return random.choice(best_children)

    def expand(self):
        # Try a new child that hasn't been pursued yet
        children = []
        for child in self.children:
            children.append(tuple(child.player))

        # Rather than picking child at random, pick the one with the best heuristic value
        actions = sorted(get_legal_actions(self.state, self.player),
                         key=lambda a: heuristic(apply_position(a, self.player), self.pursued))

        for action in actions:
            next_player = apply_position(action, self.player)
            if tuple(next_player) not in children:
                next_pursuer = apply_pursuer(self.state, next_player, self.pursuer)
                next_pursued = apply_pursued(self.state, next_player, self.pursued)
                player = self
                child = Node(self.state, next_player, next_pursued, next_pursuer, parent=player)
                player.children.append(child)
                return child

    def rollout(self, root, max_depth=10, min_depth=5) -> float:  # DEPTH = 1 FOR TESTING ONLY
        # We simulate the game X number of turns from the current node or until we win or lose
        # We track values from these simulations so that when we backpropagate we can determine the best route

        # Longer rollouts increase simulation time and decrease value of unique paths
        scaled_depth = int(min(max_depth, max(min_depth, (
                heuristic(self.player, self.pursued) + heuristic(self.player, self.pursuer)) // 2)))

        simulation = self

        for step in range(scaled_depth):
            if simulation.player[0] == simulation.pursued[0] and simulation.player[1] == simulation.pursued[1]:
                return 100 - step
            if simulation.player[0] == simulation.pursuer[0] and simulation.player[1] == simulation.pursuer[1]:
                return -100 + step

            simulation = simulate(simulation, root)

        # smooth reward
        dist_pursued = heuristic(simulation.player, simulation.pursued)
        dist_pursuer = heuristic(simulation.player, simulation.pursuer)
        reward = 100 * math.exp(-dist_pursued / 6)
        risk = -50 / ((dist_pursuer + 1) ** 0.5)

        return reward + risk

class PlannerAgent:
    def __init__(self):
        self.root = None
        self.rotation = {'left':1, 'safe':1, 'right':1}  # Laplace smoothing
        self.last_position = None
        self.last_intended_action = None

    def plan_action(self, world: np.ndarray, current: np.ndarray, pursued: np.ndarray, pursuer: np.ndarray) -> Optional[
        np.ndarray]:
        """
        Parameters:
        - world (np.ndarray): A 2D numpy array representing the grid environment.
        - 0 represents a walkable cell.
        - 1 represents an obstacle.
        - current (np.ndarray): The (row, column) coordinates of the current position.
        - pursued (np.ndarray): The (row, column) coordinates of the agent to be pursued.
        - pursuer (np.ndarray): The (row, column) coordinates of the agent to evade from.
        """
        # p-value estimation
        # Online Bayesian Estimation with a Dirichlet Prior
        def observe_actual_move(new_position: np.ndarray, weight=1):
            '''
            Combines MLE with Bayesian Estimation and prior known knowledge, called Dirichlet Prior.
            Pi = Ni/sum_j(Nj)
            Pi = Estimated probability for category i
            Ni = Number of times category i was observed
            sum_j(Nj) = The total number of actions observed

            Because we combine Maximum Likelihood Estimation with laplace smoothing, we have a
            Bayesian Estimator with a Dirichlet prior belief.
            This means rather than just using observed frequencies, we assume p ~ Dirichlet (a1, a2, ..., ak)
            where a_k represents the number of potential actions (in our case left, right, or no rotation).
            '''
            if self.last_position is None or self.last_intended_action is None:
                return

            actual_action = new_position - self.last_position
            intended = self.last_intended_action

            # Normalize for comparison: intended direction must not be (0,0)
            if np.all(intended == 0):
                self.last_position = None
                self.last_intended_action = None
                return

            # Left rotation: (-y, x)
            left = np.array([-intended[1], intended[0]])
            right = np.array([intended[1], -intended[0]])

            if np.array_equal(actual_action, intended):
                self.rotation["safe"] += weight
            elif np.array_equal(actual_action, left):
                self.rotation["left"] += weight
            elif np.array_equal(actual_action, right):
                self.rotation["right"] += weight

            self.last_position = None
            self.last_intended_action = None
        observe_actual_move(current)

        # Continue from previous tree if possible
        state_key = (tuple(current), tuple(pursued), tuple(pursuer))
        if self.root is not None:
            # Find a matching child of the previous root
            match = [child for child in self.root.children if child.state_key == state_key]
            if match:
                self.root = match[0]
            else:
                self.root = None

        # Create a new root if no continuation is found
        if self.root is None: self.root = Node(world, current, pursued, pursuer)

        # p-value estimation helper function
        def get_estimated_probabilities():
            total = sum(self.rotation.values())
            # Compute raw probabilities
            raw_probs = {
                "left": self.rotation["left"],
                "safe": self.rotation.get("safe", self.rotation.get("safe", 0)),  # compatibility
                "right": self.rotation["right"]
            }

            # Normalize to ensure they sum to 1
            # The sum of all prob cannot be greater than 1
            normalized = {k: v / total for k, v in raw_probs.items()}
            return normalized

        if heuristic(current, pursued) > 5 and heuristic(current, pursuer) > 5:
            action = get_astar(current, pursued, pursuer, world)
        else:
            action, new_root = mcts(self.root)
            self.root = new_root

        '''
        CRASH CONDITION
        If the target is not reachable OR
        the target is not within range
        AND the next action is within range of aggressor
        AND an obstacle is within range.
        '''
        '''crash_location = crash(world, current)
        if (heuristic(current, pursued) > 1 >= heuristic(action - current, pursuer) and
                not np.array_equal(crash_location, np.array([0,0]))):
            return crash_location'''

        '''
        DODGE CONDITION
        If the aggressor is next to you
        AND the target is not within range
        AND the action is within range of the aggressor
        '''
        if (heuristic(current, pursuer) == 1 and heuristic(current, pursued) > 1 >=
                heuristic(apply_position(action, current), pursuer)):
            return pursuer - current

        self.last_position = current.copy()
        self.last_intended_action = action.copy()
        prob = get_estimated_probabilities()
        print('Tom prob accuracy: ' +
              'Left: ' + str(round(abs(0.3 - prob['left']), 2)) +
              ', Safe: ' + str(round(abs(0.3 - prob['safe']), 2)) +
              ', Right: ' + str(round(abs(0.4 - prob['right']), 2)))
        return action
        # return np.array([0, 0])
        # return crash_direction(world, current)

