from pyamaze import maze, agent, COLOR, textLabel
import csv
import random
from collections import deque

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
    """Load maze from CSV."""
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            coords = eval(row[0])  # Converts string to tuple
            E, W, N, S = map(int, row[1:])  # Parse the directions
            maze_obj.maze_map[coords] = {"E": E, "W": W, "N": N, "S": S}  # Update maze map with directions

def bfs_search(maze_obj, start=None, goal=None):
    """Breadth-First Search algorithm."""
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)  # Default start position
    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)  # Default goal position

    frontier = deque([start])  # Queue for BFS
    visited = {}  # Stores the visited cells
    exploration_order = []  # The order of exploration
    explored = set([start])  # Set of already explored cells

    while frontier:
        current = frontier.popleft()  # Dequeue the next cell

        if current == goal:
            break  # Stop if we reached the goal

        for direction in 'ESNW':  # Check all possible directions (East, West, North, South)
            if maze_obj.maze_map[current][direction] == 1:  # If a wall is not blocking
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
    goal_position = (1, 1)  # Define goal position
    exploration_order, visited_cells, path_to_goal = bfs_search(m, goal=goal_position)  # Perform BFS search

    if path_to_goal:
        agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)  # Agent for BFS exploration
        agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)  # Agent for path trace
        agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)  # Agent for goal
        m.tracePath({agent_bfs: exploration_order}, delay=1)  # Trace the exploration path
        m.tracePath({agent_trace: path_to_goal}, delay=1)  # Trace the path to goal
        m.tracePath({agent_goal: visited_cells}, delay=1)  # Trace the visited cells
        textLabel(m, 'Goal Position', str(goal_position))  # Display goal position
        textLabel(m, 'BFS Path Length', len(path_to_goal) + 1)  # Display BFS path length
        textLabel(m, 'BFS Search Length', len(exploration_order))  # Display BFS search length
    else:
        print("No path found to the goal!")  # Print message if no path found
    m.run()  # Run the maze visualization
