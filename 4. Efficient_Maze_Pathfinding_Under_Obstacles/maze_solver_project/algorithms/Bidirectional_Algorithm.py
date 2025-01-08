from pyamaze import maze
from collections import deque

def bidirectional_search(m, goal):
    start = (0, 0)
    open_list_start = deque([start])
    open_list_goal = deque([goal])
    
    visited_from_start = {start: None}
    visited_from_goal = {goal: None}
    
    directions = {
        'N': (-1, 0),
        'S': (1, 0),
        'E': (0, 1),
        'W': (0, -1)
    }
    
    exploration_order = []
    visited_cells = set()

    while open_list_start and open_list_goal:
        # Expand from the start
        current_start = open_list_start.popleft()
        exploration_order.append(current_start)
        visited_cells.add(current_start)

        if current_start in visited_from_goal:
            # Path found from start to goal
            return reconstruct_path(visited_from_start, visited_from_goal, current_start, goal), exploration_order, visited_cells
        
        for direction, (d_row, d_col) in directions.items():
            next_cell = (current_start[0] + d_row, current_start[1] + d_col)

            if 0 <= next_cell[0] < m.rows and 0 <= next_cell[1] < m.cols:
                if direction == 'N' and m.maze_map[current_start]["N"] == 1:
                    if next_cell not in visited_from_start:
                        visited_from_start[next_cell] = current_start
                        open_list_start.append(next_cell)

        # Expand from the goal
        current_goal = open_list_goal.popleft()
        exploration_order.append(current_goal)
        visited_cells.add(current_goal)

        if current_goal in visited_from_start:
            # Path found from goal to start
            return reconstruct_path(visited_from_start, visited_from_goal, current_goal, goal), exploration_order, visited_cells
        
        for direction, (d_row, d_col) in directions.items():
            next_cell = (current_goal[0] + d_row, current_goal[1] + d_col)

            if 0 <= next_cell[0] < m.rows and 0 <= next_cell[1] < m.cols:
                if direction == 'N' and m.maze_map[current_goal]["N"] == 1:
                    if next_cell not in visited_from_goal:
                        visited_from_goal[next_cell] = current_goal
                        open_list_goal.append(next_cell)

    return [], exploration_order, visited_cells

def reconstruct_path(visited_from_start, visited_from_goal, meet_point, goal):
    path = []
    current = meet_point

    while current != (0, 0):
        path.append(current)
        current = visited_from_start[current]
    path.reverse()

    current = meet_point
    while current != goal:
        current = visited_from_goal[current]
        path.append(current)
    
    return path
