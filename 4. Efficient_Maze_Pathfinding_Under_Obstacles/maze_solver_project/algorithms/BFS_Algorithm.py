from pyamaze import maze
from collections import deque

def bfs_search(m, goal):
    start = (0, 0)
    open_list = deque([start])  # Queue to explore the maze
    visited = set([start])  # Set of visited cells
    parent_map = {start: None}  # To reconstruct the path
    
    # Directions: (row offset, col offset)
    directions = {
        'N': (-1, 0),  # North
        'S': (1, 0),   # South
        'E': (0, 1),   # East
        'W': (0, -1),  # West
    }

    exploration_order = []  # To track exploration order
    visited_cells = set()   # To track visited cells

    while open_list:
        current = open_list.popleft()  # Get the next cell to explore
        exploration_order.append(current)
        visited_cells.add(current)

        if current == goal:
            break  # If we reached the goal, stop the search

        # Check all four possible directions (N, S, E, W)
        for direction, (dr, dc) in directions.items():
            next_cell = (current[0] + dr, current[1] + dc)

            # Ensure the next cell is within bounds of the maze
            if 0 <= next_cell[0] < m.rows and 0 <= next_cell[1] < m.cols:
                # Check if the move is valid (i.e., the wall is not blocking the path)
                if direction == 'N' and m.maze_map[current]["N"] == 1:  # Move North is possible
                    if next_cell not in visited:
                        visited.add(next_cell)
                        open_list.append(next_cell)
                        parent_map[next_cell] = current
                elif direction == 'S' and m.maze_map[current]["S"] == 1:  # Move South is possible
                    if next_cell not in visited:
                        visited.add(next_cell)
                        open_list.append(next_cell)
                        parent_map[next_cell] = current
                elif direction == 'E' and m.maze_map[current]["E"] == 1:  # Move East is possible
                    if next_cell not in visited:
                        visited.add(next_cell)
                        open_list.append(next_cell)
                        parent_map[next_cell] = current
                elif direction == 'W' and m.maze_map[current]["W"] == 1:  # Move West is possible
                    if next_cell not in visited:
                        visited.add(next_cell)
                        open_list.append(next_cell)
                        parent_map[next_cell] = current

    # Reconstruct the path to the goal from the parent map
    path_to_goal = []
    current = goal
    while current != start:
        path_to_goal.append(current)
        current = parent_map[current]
    path_to_goal.append(start)
    path_to_goal.reverse()  # Reverse the path to get it from start to goal

    return exploration_order, visited_cells, path_to_goal
