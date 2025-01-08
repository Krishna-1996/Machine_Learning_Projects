from collections import deque
from pyamaze import maze, agent, COLOR, textLabel

def bfs_search(m, start=None, goal=None):
    if start is None:
        start = (m.rows - 1, m.cols - 1)  # Default start position
    if goal is None:
        goal = (1, 1)  # Default goal position

    # Initialize queue for BFS
    queue = deque([start])
    visited = {start: None}  # Store visited cells and their parents
    exploration_order = []  # Keep track of the exploration order

    while queue:
        current = queue.popleft()
        exploration_order.append(current)

        if current == goal:
            break

        for direction in 'ESNW':  # Check all four directions
            if m.mazeMap[current][direction] == 1:  # If the direction is open
                next_cell = get_next_cell(current, direction)
                if next_cell not in visited:
                    visited[next_cell] = current
                    queue.append(next_cell)

    # Reconstruct the path
    path_to_goal = {}
    if goal in visited:
        cell = goal
        while cell != start:
            path_to_goal[visited[cell]] = cell
            cell = visited[cell]

    return exploration_order, visited, path_to_goal

def get_next_cell(current, direction):
    x, y = current
    if direction == 'E':
        return (x, y + 1)  # Move East
    elif direction == 'W':
        return (x, y - 1)  # Move West
    elif direction == 'N':
        return (x - 1, y)  # Move North
    elif direction == 'S':
        return (x + 1, y)  # Move South
    return current
