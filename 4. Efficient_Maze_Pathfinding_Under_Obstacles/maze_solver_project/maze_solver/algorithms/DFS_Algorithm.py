from pyamaze import maze, agent, COLOR, textLabel
import csv
import random

def get_next_cell(current, direction):
    """Calculate the next cell based on the current cell and direction."""
    x, y = current
    if direction == 'E':
        return (x, y + 1)
    elif direction == 'W':
        return (x, y - 1)
    elif direction == 'N':
        return (x - 1, y)
    elif direction == 'S':
        return (x + 1, y)
    return current

def load_maze_from_csv(file_path, maze_obj):
    """Load maze from CSV into the maze object."""
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            coords = eval(row[0])  # Converts string to tuple
            E, W, N, S = map(int, row[1:])  # Parse the directions
            maze_obj[coords] = {"E": E, "W": W, "N": N, "S": S}  # Update maze map with directions

def is_valid_move(current, direction, maze_obj):
    """Check if moving in the given direction is valid (i.e., no wall)."""
    return maze_obj.get(current, {}).get(direction, 0) == 1


def dfs_search(maze_obj, start=None, goal=None):
    """Depth-First Search (DFS) algorithm."""
    if start is None:
        start = (1, 1)  # Default start position
    if goal is None:
        goal = (50, 100)  # Default goal position

    frontier = [start]  # Stack for DFS
    visited = {}  # Stores the visited cells
    exploration_order = []  # The order of exploration
    explored = set([start])  # Set of already explored cells

    while frontier:
        current = frontier.pop()  # Pop the next cell from the stack

        if current == goal:
            break  # Stop if we reached the goal

        for direction in 'ESNW':  # Check all possible directions (East, West, North, South)
            if is_valid_move(current, direction, maze_obj):  # If a wall is not blocking
                next_cell = get_next_cell(current, direction)  # Get the next cell in that direction
                if next_cell not in explored:  # If the next cell is unexplored
                    frontier.append(next_cell)  # Add it to the frontier
                    visited[next_cell] = current  # Mark the current cell as visited from 'next_cell'
                    exploration_order.append(next_cell)  # Add it to the exploration order
                    explored.add(next_cell)  # Add to explored set

    if goal not in visited:
        print("Goal is unreachable!")
        return [], {}, {}  # Return empty if the goal is unreachable

    path_to_goal = {}  # To store the path from goal to start
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell  # Trace path backwards from goal to start
        cell = visited[cell]

    return exploration_order, visited, path_to_goal  # Return exploration order, visited cells, and path to goal

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
