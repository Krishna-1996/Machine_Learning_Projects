from flask import Flask, render_template, request
from pyamaze import maze, agent, COLOR, textLabel
import random
import os
from collections import deque

app = Flask(__name__)

# Function to add random obstacles to the maze
def add_obstacles(m, obstacle_percentage):
    total_cells = m.rows * m.cols
    num_obstacles = int(total_cells * (obstacle_percentage / 100))
    
    valid_cells = [(row, col) for row in range(m.rows) for col in range(m.cols)]
    blocked_cells = random.sample(valid_cells, num_obstacles)
    
    for (row, col) in blocked_cells:
        if (row, col) in m.maze_map:
            # Block walls (North, South, East, West) randomly
            if random.choice([True, False]):
                m.maze_map[(row, col)]["E"] = 0  # Block East wall
                m.maze_map[(row, col)]["W"] = 0  # Block West wall
            if random.choice([True, False]):
                m.maze_map[(row, col)]["N"] = 0  # Block North wall
                m.maze_map[(row, col)]["S"] = 0  # Block South wall
    return m

# Function to save maze visualization as an image
def save_maze_image(m):
    maze_image_path = 'static/maze_image.png'
    m.saveMaze(maze_image_path)
    return maze_image_path

# BFS Algorithm
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

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html', maze_image=None)

# Route to run the algorithm
@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    obstacle_percentage = int(request.form.get('obstacle_percentage'))
    algorithm_choice = request.form.get('algorithm_choice')
    
    m = maze(50, 100)  # Create a 50x100 maze
    m.CreateMaze(loadMaze='D:/Machine_Learning_Projects/4. Efficient_Maze_Pathfinding_Under_Obstacles/maze_solver_project/maze_with_obstacles.csv')  # Optionally load an existing maze CSV
    # D:/Machine_Learning_Projects/4. Efficient_Maze_Pathfinding_Under_Obstacles/maze_solver_project/maze_with_obstacles.csv

    # Generate maze with obstacles
    m = add_obstacles(m, obstacle_percentage)
    
    goal_position = (1, 1)  # Define the goal position
    print(f"Running {algorithm_choice} algorithm with goal {goal_position}")

    # Run BFS Algorithm
    if algorithm_choice == "BFS":
        exploration_order, visited_cells, path_to_goal = bfs_search(m, goal=goal_position)

    # Save the maze visualization image
    maze_image_path = save_maze_image(m)

    # Return the result and show the generated maze
    return render_template(
        'index.html', 
        maze_image=maze_image_path,
        path_length=len(path_to_goal) + 1,
        search_length=len(exploration_order),
        algorithm=algorithm_choice
    )

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
