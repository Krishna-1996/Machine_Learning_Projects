from pyamaze import maze, agent, COLOR, textLabel
from collections import deque
from algorithms.BFS_Algorithm import bfs_search

# Function to search the maze using BFS
def bfs_search(maze_obj, start=None, goal=None):
    start = (m.rows - 1, m.cols - 1)  # Start at the bottom-right corner
    queue = deque([start])  # Initialize queue with the start position
    visited = set([start])  # Set of visited cells
    parent_map = {}  # Map to store the parent of each cell
    exploration_order = []  # List to store the order of exploration

    # Perform BFS to find the path
    while queue:
        current = queue.popleft()
        exploration_order.append(current)  # Add the current cell to the exploration order

        if current == goal:  # If we reached the goal, stop
            break
        
        # Explore all four directions (East, West, North, South)
        for direction in 'ESNW':
            if m.maze_map[current][direction] == 1:  # If the direction is open
                next_cell = get_next_cell(current, direction)
                if next_cell not in visited:
                    visited.add(next_cell)
                    queue.append(next_cell)
                    parent_map[next_cell] = current  # Store the parent of the next cell

    # If goal is not in visited, there's no path to the goal
    if goal not in parent_map:
        return exploration_order, visited, []  # No path found, return empty path

    # Reconstruct the path from goal to start
    path_to_goal = reconstruct_path(parent_map, goal, start)
    return exploration_order, visited, path_to_goal

# Function to get the next cell based on the current cell and direction
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

# Function to reconstruct the path from the goal to the start using parent_map
def reconstruct_path(parent_map, goal, start):
    path = []
    current = goal
    while current != start:  # Keep going until we reach the start
        path.append(current)  # Add current cell to the path
        current = parent_map.get(current)  # Get the parent of the current cell
        if current is None:  # If no parent exists, stop the reconstruction (in case of no valid path)
            return []
    path.append(start)  # Add the start to the path
    path.reverse()  # Reverse the path to get it from start to goal
    return path
