import heapq
from pyamaze import maze

def dijkstra_search(m, goal):
    start = (0, 0)
    open_list = []
    heapq.heappush(open_list, (0, start))  # (distance, position)
    
    distances = {start: 0}  # Distance from the start node
    came_from = {}  # To reconstruct the path
    visited_cells = set()
    exploration_order = []
    
    directions = {
        'N': (-1, 0),
        'S': (1, 0),
        'E': (0, 1),
        'W': (0, -1)
    }

    while open_list:
        current_distance, current = heapq.heappop(open_list)
        exploration_order.append(current)
        visited_cells.add(current)

        if current == goal:
            break
        
        # Check all neighbors
        for direction, (d_row, d_col) in directions.items():
            next_cell = (current[0] + d_row, current[1] + d_col)

            # Ensure next_cell is within bounds and open
            if 0 <= next_cell[0] < m.rows and 0 <= next_cell[1] < m.cols:
                if direction == 'N' and m.maze_map[current]["N"] == 1:
                    new_distance = current_distance + 1
                    if next_cell not in distances or new_distance < distances[next_cell]:
                        distances[next_cell] = new_distance
                        heapq.heappush(open_list, (new_distance, next_cell))
                        came_from[next_cell] = current

        # If goal is reached, we stop the search
    # Reconstruct the path
    path_to_goal = []
    current = goal
    while current != start:
        path_to_goal.append(current)
        current = came_from[current]
    path_to_goal.append(start)
    path_to_goal.reverse()

    return exploration_order, visited_cells, path_to_goal
