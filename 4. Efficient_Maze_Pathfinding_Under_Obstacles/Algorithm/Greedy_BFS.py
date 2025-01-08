import heapq
from pyamaze import maze

def greedy_bfs_search(m, goal):
    start = (0, 0)
    open_list = []
    heapq.heappush(open_list, (manhattan_distance(start, goal), start))  # (f_score, position)
    came_from = {}  # To reconstruct path
    visited_cells = set()
    exploration_order = []

    # Directions: North, South, East, West (relative movement)
    directions = {
        'N': (-1, 0),  # North (row - 1)
        'S': (1, 0),   # South (row + 1)
        'E': (0, 1),   # East (col + 1)
        'W': (0, -1)   # West (col - 1)
    }

    while open_list:
        _, current = heapq.heappop(open_list)
        exploration_order.append(current)

        # If the goal is reached, stop
        if current == goal:
            break
        
        visited_cells.add(current)

        # Check all possible directions (North, South, East, West)
        for direction, (d_row, d_col) in directions.items():
            next_cell = (current[0] + d_row, current[1] + d_col)

            # Ensure next_cell is within bounds and has an open wall
            if 0 <= next_cell[0] < m.rows and 0 <= next_cell[1] < m.cols:
                if direction == 'N' and m.maze_map[current]["N"] == 1:  # Check if North wall is open
                    heapq.heappush(open_list, (manhattan_distance(next_cell, goal), next_cell))
                    came_from[next_cell] = current
                elif direction == 'S' and m.maze_map[current]["S"] == 1:  # Check if South wall is open
                    heapq.heappush(open_list, (manhattan_distance(next_cell, goal), next_cell))
                    came_from[next_cell] = current
                elif direction == 'E' and m.maze_map[current]["E"] == 1:  # Check if East wall is open
                    heapq.heappush(open_list, (manhattan_distance(next_cell, goal), next_cell))
                    came_from[next_cell] = current
                elif direction == 'W' and m.maze_map[current]["W"] == 1:  # Check if West wall is open
                    heapq.heappush(open_list, (manhattan_distance(next_cell, goal), next_cell))
                    came_from[next_cell] = current

    # Reconstruct the path to the goal
    path_to_goal = []
    current = goal
    while current != start:
        path_to_goal.append(current)
        current = came_from[current]
    path_to_goal.append(start)
    path_to_goal.reverse()

    return exploration_order, visited_cells, path_to_goal
