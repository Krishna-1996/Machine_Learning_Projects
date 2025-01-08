from flask import Flask, jsonify, request
from pyamaze import maze, agent, COLOR
import importlib
import os

app = Flask(__name__)

# Mapping obstacle percentage to CSV file names
obstacle_files = {
    0: "Obstacles_Design_1_0p.csv",
    10: "Obstacles_Design_2_10p.csv",
    30: "Obstacles_Design_3_30p.csv",
    50: "Obstacles_Design_4_50p.csv"
}

goal_positions = {
    "Top Left": (1, 1),
    "Top Right": (1, 99),
    "Bottom Left": (49, 1),
    "Bottom Right": (49, 99),
    "Center": (25, 50)
}

@app.route('/solve_maze', methods=['POST'])
def solve_maze():
    # Get user input from the request
    data = request.json
    algorithm_choice = data.get("algorithm")
    obstacle_percentage = data.get("obstacle_percentage")
    goal_choice = data.get("goal_position")
    
    goal_position = goal_positions[goal_choice]
    csv_file_path = os.path.join("path_to_csvs", obstacle_files[obstacle_percentage])

    try:
        # Create the maze object
        m = maze(50, 100)
        m.CreateMaze(loadMaze=csv_file_path)

        # Select the algorithm to run
        algorithm_module = importlib.import_module(f"maze_solver.algorithms.{algorithm_choice}")
        exploration_order, visited_cells, path_to_goal = algorithm_module.run_algorithm(m, goal=goal_position)

        # Return the solution
        return jsonify({
            "maze": m.maze,
            "exploration_order": exploration_order,
            "path_to_goal": path_to_goal,
            "goal_position": goal_position
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
