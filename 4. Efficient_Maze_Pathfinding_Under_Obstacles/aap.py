import random
from flask import Flask, render_template, request, jsonify
import time
import importlib
from pyamaze import maze

app = Flask(__name__)

# Maze generation function
def generate_maze_with_obstacles(obstacle_percentage):
    m = maze(50, 100)
    m.CreateMaze(loopPercent=90)
    
    total_cells = 50 * 100
    num_obstacles = int(total_cells * (obstacle_percentage / 100))
    
    placed_obstacles = 0
    while placed_obstacles < num_obstacles:
        row = random.randint(1, 49)
        col = random.randint(1, 99)
        if m.mazeMap[row][col] == 0:
            m.mazeMap[row][col] = 1
            placed_obstacles += 1
    
    return m

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    algorithm_choice = request.form['algorithm']
    obstacle_percentage = int(request.form['obstacle_percentage'])
    
    goal_position = (1, 1)
    try:
        m = generate_maze_with_obstacles(obstacle_percentage)

        algorithm_module = importlib.import_module(f'algorithms.{algorithm_choice}')
        
        start_time = time.time()
        
        # Run the selected algorithm
        if algorithm_choice == "BFS_Algorithm":
            exploration_order, visited_cells, path_to_goal = algorithm_module.bfs_search(m, goal=goal_position)
        elif algorithm_choice == "DFS_Algorithm":
            exploration_order, visited_cells, path_to_goal = algorithm_module.dfs_search(m, goal=goal_position)
        elif algorithm_choice == "A_Star":
            exploration_order, visited_cells, path_to_goal = algorithm_module.A_star_search(m, goal=goal_position)
        elif algorithm_choice == "Greedy_BFS":
            exploration_order, visited_cells, path_to_goal = algorithm_module.greedy_bfs_search(m, goal=goal_position)
        
        elapsed_time = time.time() - start_time
        
        result = {
            'path_length': len(path_to_goal) + 1 if path_to_goal else 0,
            'exploration_length': len(exploration_order),
            'elapsed_time': elapsed_time,
            'algorithm': algorithm_choice,
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
