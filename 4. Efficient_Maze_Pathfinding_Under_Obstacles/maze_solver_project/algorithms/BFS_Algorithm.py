from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

def bfs_search(m, goal):
    start = (m.rows - 1, m.cols - 1)  # Start at the bottom-right corner
    queue = deque([start])
    visited = set([start])
    parent_map = {}
    exploration_order = []
    
    while queue:
        current = queue.popleft()
        exploration_order.append(current)
        
        if current == goal:
            break
        
        for direction in 'ESNW':
            if m.maze_map[current][direction] == 1:  # If the direction is open
                next_cell = get_next_cell(current, direction)
                if next_cell not in visited:
                    visited.add(next_cell)
                    queue.append(next_cell)
                    parent_map[next_cell] = current
    
    path_to_goal = reconstruct_path(parent_map, goal, start)
    return exploration_order, visited, path_to_goal

def get_next_cell(current, direction):
    x, y = current
    if direction == 'E':
        return (x, y + 1)
    elif direction == 'W':
        return (x, y - 1)
    elif direction == 'N':
        return (x - 1, y)
    elif direction == 'S':
        return (x + 1, y)

def reconstruct_path(parent_map, goal, start):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = parent_map[current]
    path.reverse()
    return path
