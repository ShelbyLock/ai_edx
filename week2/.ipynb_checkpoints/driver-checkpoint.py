"""
Skeleton code for Project 1 of Columbia University's AI EdX course (8-puzzle).
Python 3
"""
import queue as Q
from collections import deque
import timeit
import resource
import sys
import math
# -------------------------------------
## The Class that Represents the Puzzle
# -------------------------------------
class PuzzleState(object):
    """docstring for PuzzleState"""
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        if n*n != len(config) or n < 2:
            raise Exception("the length of config is not correct!")
        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.dimension = n
        self.config = config
        self.children = []
        for i, item in enumerate(self.config):
            if item == 0:
                self.blank_row = i // self.n
                self.blank_col = i % self.n
                break

    def display(self):
        for i in range(self.n):
            line = []
            offset = i * self.n
            for j in range(self.n):
                line.append(self.config[offset + j])
            print(line)

    def move_left(self):
        if self.blank_col == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):
        if self.blank_col == self.n - 1:
            return None

        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):
        if self.blank_row == 0:
            return None

        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
        if self.blank_row == self.n - 1:
            return None

        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):
        """expand the node"""
        # add child nodes in order of UDLR
        if len(self.children) == 0:
            up_child = self.move_up()
            if up_child is not None:
                self.children.append(up_child)
            down_child = self.move_down()
            if down_child is not None:
                self.children.append(down_child)
            left_child = self.move_left()
            if left_child is not None:
                self.children.append(left_child)
            right_child = self.move_right()
            if right_child is not None:
                self.children.append(right_child)
        return self.children
    
# -------------------------------------
## Helper Functions
# -------------------------------------
def output(actions_traced, cost, nodes_expanded, max_search_depth,running_time, max_men_use):
    file1 = open("output.txt","w") 
    strpath = "path_to_goal: "+str(actions_traced) + '\n'
    strcost = "cost_of_path: "+str(cost) + '\n'
    strnode = "nodes_expanded: "+str(nodes_expanded) + '\n'
    strdepth = "search_depth: "+str(cost) + '\n'
    strmax = "max_search_depth: "+str(max_search_depth) + '\n'
    strrun = "running_time: "+str(running_time) + '\n'
    strmen = "max_ram_usage: "+str(max_men_use) + '\n'
    out = strpath + strcost + strnode + strdepth + strmax + strrun + strmen
    print(out)
    file1.write(out) 
    
def tracePathToGoal(explored, initial, goal):
    actions = list()
    while goal.config != initial.config:
        actions.append(goal.action)
        goal = goal.parent
    actions.reverse()
    return actions

def calculate_manhattan_dist(n, current_config, goal_config):
    distance = 0
    for v in range(1,n*n):
        current_inx = current_config.index(v)
        goal_inx = goal_config.index(v)
        current_row = current_inx // n
        current_col = current_inx % n
        goal_row = goal_inx // n
        goal_col = goal_inx % n
        distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance

# -------------------------------------
## BFS search
# -------------------------------------
def bfs_search(initial_state):
    start = timeit.default_timer()
    memory_usage = list()
    
    max_search_depth = 0
    nodes_expanded = 0
    goal_state_config = "0,1,2,3,4,5,6,7,8".split(",")
    goal_state_config = tuple(map(int, goal_state_config))
    
    frontier = deque()
    frontier.append(initial_state)
    explored = deque()
    explored_stored = set()
    frontier_stored = set()
    frontier_stored.add(initial_state.config)
    
    while frontier_stored:
        state = frontier.popleft()
        frontier_stored.remove(state.config)
        explored.append(state)
        explored_stored.add(state.config)
        
        if state.config == goal_state_config:
            stop = timeit.default_timer()
            actions_traced = tracePathToGoal(explored, initial_state, state)
            cost = state.cost
            running_time = stop - start
            max_men_use = max(memory_usage)
            output(actions_traced, cost, nodes_expanded, max_search_depth, running_time, max_men_use)
            return True
        
        for neighbor in state.expand():
            
            if (neighbor.config not in frontier_stored) and (neighbor.config not in explored_stored):
                
                memory_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                
                frontier.append(neighbor)
                frontier_stored.add(neighbor.config)
                if neighbor.cost > max_search_depth:
                    max_search_depth = neighbor.cost
                    
        nodes_expanded += 1
    return False

# -------------------------------------
## DFS search
# -------------------------------------
def dfs_search(initial_state):
    start = timeit.default_timer()
    memory_usage = list()
    
    max_search_depth = 0
    nodes_expanded = 0
    goal_state_config = "0,1,2,3,4,5,6,7,8".split(",")
    goal_state_config = tuple(map(int, goal_state_config))

    frontier = deque()
    frontier.append(initial_state)
    explored = deque()
    explored_stored = set()
    frontier_stored = set()
    frontier_stored.add(initial_state.config)
    
    while frontier_stored:
        state = frontier.pop()
        frontier_stored.remove(state.config)
        explored.append(state)
        explored_stored.add(state.config)

        if state.config == goal_state_config:
            stop = timeit.default_timer()
            actions_traced = tracePathToGoal(explored, initial_state, state)
            cost = state.cost
            running_time = stop - start
            max_men_use = max(memory_usage)
            output(actions_traced, cost, nodes_expanded, max_search_depth, running_time, max_men_use)
            return True

        state.expand()
        length_children = len(state.children)
        for i in range(length_children):
            neighbor = state.children[length_children-i-1]
            if (neighbor.config not in frontier_stored) and (neighbor.config not in explored_stored):
                
                memory_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                
                frontier.append(neighbor)
                frontier_stored.add(neighbor.config)
                if neighbor.cost > max_search_depth:
                    max_search_depth = neighbor.cost 
                    
        nodes_expanded += 1
    return False

# -------------------------------------
## AST search
# -------------------------------------
def A_star_search(initial_state):
    start = timeit.default_timer()
    memory_usage = list()
    
    max_search_depth = 0
    nodes_expanded = 0
    goal_state_config = "0,1,2,3,4,5,6,7,8".split(",")
    goal_state_config = tuple(map(int, goal_state_config))
    
    frontier = list()
    explored = deque()    
    explored_stored = set()
    frontier_stored = set()    
    h = calculate_manhattan_dist(initial_state.n, initial_state.config, goal_state_config)
    frontier.append((initial_state, h))
    frontier_stored.add(initial_state.config)
    
    while frontier_stored:
        state, cost = frontier.pop(0)
        frontier_stored.remove(state.config)
        explored.append((state, cost))
        explored_stored.add(state.config)

        if state.config == goal_state_config:
            stop = timeit.default_timer()
            actions_traced = tracePathToGoal(explored, initial_state, state)
            cost = state.cost
            running_time = stop - start
            max_men_use = max(memory_usage)
            output(actions_traced, cost, nodes_expanded, max_search_depth, running_time, max_men_use)
            return True
        
        for neighbor in state.expand():
            g = neighbor.cost
            h = calculate_manhattan_dist(initial_state.n, neighbor.config, goal_state_config)
            f = g+h 
            if (neighbor.config not in frontier_stored) and (neighbor.config not in explored_stored):
                
                memory_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                
                frontier.append((neighbor, f))
                frontier_stored.add(neighbor.config)
                if neighbor.cost > max_search_depth:
                    max_search_depth = neighbor.cost
                    
            elif (neighbor.config in frontier_stored):
                frontier = list([(s,f) if (s.config == neighbor.config) and (c > f) else (s,c) for s,c in frontier]) 
                memory_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                
        frontier = sorted(frontier, key=lambda tup: tup[1])       
        nodes_expanded += 1
    return False

# -------------------------------------
## Main Function 
#1. reads in Input
#2. Runs corresponding Algorithm
# -------------------------------------
def main():
    sm = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    #begin_state = cf.split(",")
    begin_state = tuple(map(int, begin_state))
    size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, size)
    if sm == "bfs":
        bfs_search(hard_state)
    elif sm == "dfs":
        dfs_search(hard_state)
    elif sm == "ast":
        A_star_search(hard_state)
    else:
        print("Enter valid command arguments !")
        
if __name__ == '__main__':

    main()