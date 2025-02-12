from pyamaze import maze, agent, COLOR, textLabel
import heapq  # for priority queue implementation
import csv  # for reading CSV file to load maze data
import random  # for generating random obstacles

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

# Load a maze from a CSV file and map its walls
def load_maze_from_csv(file_path, maze_obj):
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            coords = eval(row[0])  # Convert string to tuple
            E, W, N, S = map(int, row[1:])  # Extract wall data
            maze_obj.maze_map[coords] = {"E": E, "W": W, "N": N, "S": S}

# Add obstacles to the maze at random positions
def add_obstacles(maze_obj, obstacle_percentage=20):
    total_cells = maze_obj.rows * maze_obj.cols  # Total number of cells
    num_obstacles = int(total_cells * (obstacle_percentage / 100))  # Number of obstacles to add
    valid_cells = [(row, col) for row in range(maze_obj.rows) for col in range(maze_obj.cols)]  # All valid cell positions
    blocked_cells = random.sample(valid_cells, num_obstacles)  # Randomly select blocked cells
    obstacle_locations = []
    
    # Set walls in the selected cells to be blocked
    for (row, col) in blocked_cells:
        if (row, col) in maze_obj.maze_map:
            if random.choice([True, False]):
                maze_obj.maze_map[(row, col)]["E"] = 0  # Block East wall
                maze_obj.maze_map[(row, col)]["W"] = 0  # Block West wall
            if random.choice([True, False]):
                maze_obj.maze_map[(row, col)]["N"] = 0  # Block North wall
                maze_obj.maze_map[(row, col)]["S"] = 0  # Block South wall
            obstacle_locations.append((row, col))
    return obstacle_locations

# A* search algorithm to find the shortest path from start to goal
def A_star_search(maze_obj, start=None, goal=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)  # Default start position
    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)  # Default goal position
    if not (0 <= goal[0] < maze_obj.rows and 0 <= goal[1] < maze_obj.cols):
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
            if maze_obj.maze_map[current][direction] == 1:  # If the direction is open
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

if __name__ == '__main__':
    m = maze(50, 100)  # Maze size 50x100
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA//maze_with_obstacles.csv')  # Load maze from CSV
    # D:/Machine_Learning_Projects/3. Efficient_Maze_Pathfinding_Under_Obstacles/maze_csvs
    goal_position = (1, 1)  # Set the goal position
    exploration_order, visited_cells, path_to_goal = A_star_search(m, goal=goal_position)
    
    if path_to_goal:
        # Create agents for path visualization
        agent_astar = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
        agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)
        agent_goal = agent(m, goal_position[0], goal_position[1], 
                           footprints=True, color=COLOR.green, shape='square', filled=True)
        
        # Trace paths for agent movements
        m.tracePath({agent_astar: exploration_order}, delay=1)
        m.tracePath({agent_trace: path_to_goal}, delay=1)
        m.tracePath({agent_goal: visited_cells}, delay=1)
        
        # Display text labels for information
        textLabel(m, 'Goal Position', str(goal_position))
        textLabel(m, 'A* Path Length', len(path_to_goal) + 1)
        textLabel(m, 'A* Search Length', len(exploration_order))
    else:
        print("No path found to the goal!")
    m.run()  # Run the maze visualization
