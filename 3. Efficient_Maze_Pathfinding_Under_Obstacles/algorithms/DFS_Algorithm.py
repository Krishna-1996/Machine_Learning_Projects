from pyamaze import maze, agent, COLOR, textLabel
import csv
import random

def get_next_cell(current, direction):
    """Calculate the next cell based on the current cell and direction."""
    x, y = current  # Deconstruct current position into x, y coordinates
    if direction == 'E':  # If direction is East, move right
        return (x, y + 1)
    elif direction == 'W':  # If direction is West, move left
        return (x, y - 1)
    elif direction == 'N':  # If direction is North, move up
        return (x - 1, y)
    elif direction == 'S':  # If direction is South, move down
        return (x + 1, y)
    return current  # Return current if no valid direction

def load_maze_from_csv(file_path, maze_obj):
    """Load maze from CSV."""
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)  # Read the CSV file
        next(reader)  # Skip header row
        for row in reader:  # Iterate through each row in the CSV
            coords = eval(row[0])  # Convert coordinate string to a tuple
            E, W, N, S = map(int, row[1:])  # Parse the direction data
            maze_obj.maze_map[coords] = {"E": E, "W": W, "N": N, "S": S}  # Update maze with walls data

def dfs_search(maze_obj, start=None, goal=None):
    """Depth-First Search algorithm."""
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)  # Default start position (bottom-right)
    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)  # Default goal position (center)

    frontier = [start]  # Use a stack for DFS
    visited = {}  # Dictionary to store visited cells
    exploration_order = []  # List to store the order of exploration
    explored = set([start])  # Set to track all explored cells

    while frontier:  # Continue exploring while there are cells to visit
        current = frontier.pop()  # Pop from stack to get the current cell

        if current == goal:  # Stop if we reach the goal
            break

        for direction in 'ESNW':  # Check each direction (East, South, North, West)
            if maze_obj.maze_map[current][direction] == 1:  # If the path is open
                next_cell = get_next_cell(current, direction)  # Get the next cell in the direction
                if next_cell not in explored:  # If the next cell hasn't been explored yet
                    frontier.append(next_cell)  # Add the cell to the stack
                    visited[next_cell] = current  # Mark current cell as the predecessor of the next cell
                    exploration_order.append(next_cell)  # Record the exploration order
                    explored.add(next_cell)  # Mark the cell as explored

    if goal not in visited:  # If goal is unreachable, return empty results
        print("Goal is unreachable!")
        return [], {}, {}

    # Reconstruct the path to the goal by tracing back from goal to start
    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell  # Record the cell's predecessor
        cell = visited[cell]  # Move backwards along the path

    return exploration_order, visited, path_to_goal  # Return the exploration order, visited cells, and the path to goal

if __name__ == '__main__':
    m = maze(50, 100)  # Create a maze of size 50x100
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA//maze_with_obstacles.csv')  # Load maze from CSV
    goal_position = (1, 1)  # Set goal position (top-left corner)
    exploration_order, visited_cells, path_to_goal = dfs_search(m, goal=goal_position)  # Perform DFS to find the path

    if path_to_goal:  # If a path to the goal is found
        # Create agents for visualization
        agent_dfs = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)  # Red agent for exploring the maze
        agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)  # Yellow agent for tracing the path
        agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)  # Green agent at goal

        # Trace the paths in the maze
        m.tracePath({agent_dfs: exploration_order}, delay=1)  # Visualize DFS exploration
        m.tracePath({agent_trace: path_to_goal}, delay=1)  # Visualize path from start to goal
        m.tracePath({agent_goal: visited_cells}, delay=1)  # Visualize visited cells

        # Display relevant information about the search
        textLabel(m, 'Goal Position', str(goal_position))  # Show goal position
        textLabel(m, 'DFS Path Length', len(path_to_goal) + 1)  # Show length of the path
        textLabel(m, 'DFS Search Length', len(exploration_order))  # Show number of cells explored
    else:
        print("No path found to the goal!")  # Print message if no path was found
    m.run()  # Run the maze visualization
