import heapq
from pyamaze import maze, agent, COLOR, textLabel

# Heuristic function to calculate Manhattan distance between two points
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Get the next cell based on the current cell and the direction
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

def A_star_search(m, start=None, goal=None):
    if start is None:
        start = (m.rows, m.cols)  # Default start position
    if goal is None:
        goal = (m.rows // 2, m.cols // 2)  # Default goal position
    if not (0 <= goal[0] < m.rows and 0 <= goal[1] < m.cols):
        raise ValueError(f"Invalid goal position: {goal}. It must be within the bounds of the maze.")
    
    frontier = []  # Priority queue for frontier cells
    heapq.heappush(frontier, (0 + heuristic(start, goal), start))  # Add start cell to frontier
    visited = {}  # Dictionary to store visited cells
    exploration_order = []  # Order of exploration
    explored = set([start])  # Set of explored cells
    g_costs = {start: 0}  # Cost to reach each cell
    
    while frontier:
        _, current = heapq.heappop(frontier)  # Get the cell with lowest cost
        if current == goal:
            break  # Stop if goal is reached
        for direction in 'ESNW':  # Check all four directions
            if m.mazeMap[current][direction] == 1:  # If the direction is open
                next_cell = get_next_cell(current, direction)  # Get next cell in that direction
                new_g_cost = g_costs[current] + 1  # Increment cost
                if next_cell not in explored or new_g_cost < g_costs.get(next_cell, float('inf')):
                    g_costs[next_cell] = new_g_cost  # Update cost
                    f_cost = new_g_cost + heuristic(next_cell, goal)  # Total cost (g + h)
                    heapq.heappush(frontier, (f_cost, next_cell))  # Add to frontier
                    visited[next_cell] = current  # Store the path
                    exploration_order.append(next_cell)  # Add to exploration order
                    explored.add(next_cell)  # Mark as explored
    
    if goal not in visited:  # If no path to goal
        print("Goal is unreachable!")
        return [], {}, {}
    
    path_to_goal = {}  # Reconstruct path from goal to start
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]
    
    return exploration_order, visited, path_to_goal
