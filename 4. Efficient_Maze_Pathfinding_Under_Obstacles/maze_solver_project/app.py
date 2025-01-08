from flask import Flask, render_template, request
from pyamaze import maze, agent, COLOR, textLabel
import random
import os

app = Flask(__name__)

# Create maze from a CSV file or randomize obstacles
def generate_maze(obstacle_percentage):
    m = maze(50, 50)  # 50x50 maze
    # Add random obstacles
    total_cells = m.rows * m.cols
    num_obstacles = int(total_cells * (obstacle_percentage / 100))
    valid_cells = [(row, col) for row in range(m.rows) for col in range(m.cols)]
    blocked_cells = random.sample(valid_cells, num_obstacles)
    
    for (row, col) in blocked_cells:
        if (row, col) in m.maze_map:
            if random.choice([True, False]):
                m.maze_map[(row, col)]["E"] = 0  # Block East wall
                m.maze_map[(row, col)]["W"] = 0  # Block West wall
            if random.choice([True, False]):
                m.maze_map[(row, col)]["N"] = 0  # Block North wall
                m.maze_map[(row, col)]["S"] = 0  # Block South wall
    
    return m

@app.route('/')
def index():
    return render_template('index.html', maze_image=None)

@app.route('/run_algorithm', methods=['POST'])


def run_algorithm():
    obstacle_percentage = int(request.form.get('obstacle_percentage'))
    algorithm_choice = request.form.get('algorithm_choice')
    m = maze(50, 100)
    m.CreateMaze(loadMaze='path/to/your/maze.csv')
    # Generate maze with obstacles
    obstacles = add_obstacles(m, obstacle_percentage)
    m = generate_maze(obstacle_percentage)
    goal_position = (1, 1)  # Goal position
    print(f"Running BFS algorithm with goal {goal_position}")
    exploration_order, visited_cells, path_to_goal = bfs_search(m, goal=goal_position)
    print(f"Exploration Order: {exploration_order}")
    print(f"Visited Cells: {visited_cells}")
    print(f"Path to Goal: {path_to_goal}")

    # Run the selected algorithm
    if algorithm_choice == "BFS":
        from algorithms.BFS_Algorithm import bfs_search
        exploration_order, visited_cells, path_to_goal = bfs_search(m, goal=goal_position)
    elif algorithm_choice == "DFS":
        from algorithms.DFS_Algorithm import dfs_search
        exploration_order, visited_cells, path_to_goal = dfs_search(m, goal=goal_position)
    elif algorithm_choice == "Greedy BFS":
        from algorithms.Greedy_BFS import greedy_bfs_search
        exploration_order, visited_cells, path_to_goal = greedy_bfs_search(m, goal=goal_position)
    elif algorithm_choice == "A*":
        from algorithms.A_Star import A_star_search
        exploration_order, visited_cells, path_to_goal = A_star_search(m, goal=goal_position)

    # Save the maze visualization
    maze_image_path = 'static/maze_image.png'
    m.saveMaze(maze_image_path)

    # Return the result and show the generated maze
    return render_template(
        'index.html', 
        maze_image=maze_image_path,
        path_length=len(path_to_goal) + 1,
        search_length=len(exploration_order),
        algorithm=algorithm_choice
    )

if __name__ == '__main__':
    app.run(debug=True)
