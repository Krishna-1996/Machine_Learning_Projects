from pyamaze import maze, agent, COLOR, textLabel
import csv  # for reading the maze data from a CSV file
import random  # for generating random obstacles in the maze
import heapq  # for implementing a priority queue (min-heap)

# Heuristic function: Calculates Manhattan distance between two points (used by Greedy BFS)
def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Function to determine the next cell based on the current cell and direction
def get_next_cell(current, direction):
    """Calculate the next cell based on the current cell and direction."""
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

# Function to load the maze from a CSV file
def load_maze_from_csv(file_path, maze_obj):
    """Load maze from CSV."""
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            coords = eval(row[0])  # Convert the string to a tuple
            E, W, N, S = map(int, row[1:])  # Extract wall data
            maze_obj.maze_map[coords] = {"E": E, "W": W, "N": N, "S": S}

# Function to add random obstacles to the maze
def add_obstacles(maze_obj, obstacle_percentage=20):
    """Add random obstacles."""
    total_cells = maze_obj.rows * maze_obj.cols  # Calculate total number of cells in the maze
    num_obstacles = int(total_cells * (obstacle_percentage / 100))  # Number of obstacles to add
    valid_cells = [(row, col) for row in range(maze_obj.rows) for col in range(maze_obj.cols)]  # List of valid cells
    blocked_cells = random.sample(valid_cells, num_obstacles)  # Randomly select blocked cells
    obstacle_locations = []
    
    # Set the walls for blocked cells
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

# Greedy BFS algorithm to find the path from start to goal
def greedy_bfs_search(maze_obj, start=None, goal=None):
    """Greedy BFS algorithm."""
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)  # Default start position at bottom-right corner
    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)  # Default goal position at center
    
    # Min-heap priority queue based on heuristic (Greedy BFS uses heuristic only)
    frontier = []  # The frontier will store cells to explore
    heapq.heappush(frontier, (heuristic(start, goal), start))  # Push start position to the queue (heuristic, start)
    visited = {}  # Dictionary to store visited cells and their predecessors
    exploration_order = []  # Order of exploration (for visualization)
    explored = set([start])  # Set to track the explored cells
    
    while frontier:
        _, current = heapq.heappop(frontier)  # Pop the cell with the lowest heuristic cost
        
        if current == goal:
            break  # Stop if the goal is reached

        for direction in 'ESNW':  # Explore all four directions (East, West, North, South)
            if maze_obj.maze_map[current][direction] == 1:  # If the direction is open (no wall)
                next_cell = get_next_cell(current, direction)  # Get the next cell based on the direction
                if next_cell not in explored:
                    heapq.heappush(frontier, (heuristic(next_cell, goal), next_cell))  # Add to frontier based on heuristic
                    visited[next_cell] = current  # Mark the predecessor of the next cell
                    exploration_order.append(next_cell)  # Add to exploration order
                    explored.add(next_cell)  # Mark as explored

    # If the goal is not reached, return empty results
    if goal not in visited:
        print("Goal is unreachable!")
        return [], {}, {}

    # Reconstruct the path from goal to start
    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

if __name__ == '__main__':
    # Create a maze of size 50x100 and load from the specified CSV file
    m = maze(50, 100)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA//maze_with_obstacles.csv')
    goal_position = (1, 1)  # Set the goal position (top-left corner)
    
    # Run Greedy BFS search algorithm to find the path to the goal
    exploration_order, visited_cells, path_to_goal = greedy_bfs_search(m, goal=goal_position)

    # If a path is found, visualize the search process and the path
    if path_to_goal:
        # Create agents for visualization of the exploration and the path
        agent_gbfs = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
        agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)
        agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)
        
        # Trace the path for the Greedy BFS search and the actual path to the goal
        m.tracePath({agent_gbfs: exploration_order}, delay=1)  # Trace exploration order
        m.tracePath({agent_trace: path_to_goal}, delay=1)  # Trace the path from start to goal
        m.tracePath({agent_goal: visited_cells}, delay=1)  # Visualize visited cells

        # Display the goal position, path length, and search length
        textLabel(m, 'Goal Position', str(goal_position))
        textLabel(m, 'Greedy BFS Path Length', len(path_to_goal) + 1)
        textLabel(m, 'Greedy BFS Search Length', len(exploration_order))
    else:
        print("No path found to the goal!")  # In case no path is found

    # Run the maze visualization
    m.run()
