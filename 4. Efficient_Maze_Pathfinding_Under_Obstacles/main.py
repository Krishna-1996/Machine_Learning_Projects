import tkinter as tk
from tkinter import messagebox
from pyamaze import maze, agent, COLOR, textLabel
import importlib
import os

# Mapping obstacle percentage to CSV file names
obstacle_files = {
    0: "Obstacles_Design_1_0p.csv",
    10: "Obstacles_Design_2_10p.csv",
    30: "Obstacles_Design_3_30p.csv",
    50: "Obstacles_Design_4_50p.csv"
}

# Main class for the GUI
class MazeSolverApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Maze Solver")
        self.geometry("600x150")

        # Algorithm choice (default is BFS)
        self.algorithm = tk.StringVar(value="BFS")

        # Obstacle percentage choice (default is 0%)
        self.obstacle_percentage = tk.IntVar(value=0)

        # Frame for the dropdowns and button (to place them on the same line)
        self.dropdown_frame = tk.Frame(self)
        self.dropdown_frame.pack(pady=20)

        # Dropdown for algorithm selection
        self.create_algorithm_selection()

        # Dropdown for obstacle percentage selection
        self.create_obstacle_selection()

        # Button to run the selected algorithm on the maze
        self.create_run_button()

    def create_algorithm_selection(self):
        """Create the dropdown for algorithm selection."""
        label = tk.Label(self.dropdown_frame, text="Select Algorithm:", font=("Arial", 10, "bold"))
        label.grid(row=0, column=0, padx=10)

        algorithm_choices = ["BFS", "DFS", "A*", "Greedy BFS"]
        algorithm_menu = tk.OptionMenu(self.dropdown_frame, self.algorithm, *algorithm_choices)
        algorithm_menu.grid(row=0, column=1, padx=10)

    def create_obstacle_selection(self):
        """Create the dropdown for obstacle percentage selection."""
        label = tk.Label(self.dropdown_frame, text="Select Obstacle Percentage:", font=("Arial", 10, "bold"))
        label.grid(row=0, column=2, padx=10)

        obstacle_choices = [0, 10, 30, 50]
        obstacle_menu = tk.OptionMenu(self.dropdown_frame, self.obstacle_percentage, *obstacle_choices)
        obstacle_menu.grid(row=0, column=3, padx=10)

    def create_run_button(self):
        """Create the 'Run' button to execute the algorithm."""
        run_button = tk.Button(self.dropdown_frame, text="Run", command=self.run_algorithm, 
                               font=("Arial", 12, "bold"), fg="white", bg="#4CAF50", relief="flat", width=10)
        
        # Add hover effect for the button
        run_button.bind("<Enter>", lambda e: self.on_hover(run_button))
        run_button.bind("<Leave>", lambda e: self.on_leave(run_button))
        
        run_button.grid(row=0, column=4, padx=10)

    def on_hover(self, button):
        """Change button color when the mouse hovers over it."""
        button.config(bg="#45a049")

    def on_leave(self, button):
        """Reset button color when the mouse leaves it."""
        button.config(bg="#4CAF50")

    def run_algorithm(self):
        """Run the selected algorithm on the maze."""
        # Get the algorithm and obstacle percentage
        algorithm_choice = self.algorithm.get().lower()
        obstacle_percentage = self.obstacle_percentage.get()

        # Path to the selected obstacle CSV file
        csv_file_path = os.path.join("D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/Advance Version of obstacle project/maze_csvs", obstacle_files[obstacle_percentage])

        try:
            # Load the maze
            m = maze(50, 100)
            m.CreateMaze(loadMaze=csv_file_path)

            # Define the goal position
            goal_position = (1, 1)

            # Dynamically import the algorithm based on user selection
            if algorithm_choice == "bfs":
                algorithm_module = importlib.import_module('algorithms.BFS_Algorithm')  # Now looks for BFS_Algorithm in algorithms folder
                exploration_order, visited_cells, path_to_goal = algorithm_module.bfs_search(m, goal=goal_position)
            elif algorithm_choice == "dfs":
                algorithm_module = importlib.import_module('algorithms.DFS_Algorithm')
                exploration_order, visited_cells, path_to_goal = algorithm_module.dfs_search(m, goal=goal_position)
            elif algorithm_choice == "a*":
                algorithm_module = importlib.import_module('algorithms.A_Star')
                exploration_order, visited_cells, path_to_goal = algorithm_module.A_star_search(m, goal=goal_position)
            elif algorithm_choice == "greedy bfs":
                algorithm_module = importlib.import_module('algorithms.Greedy_BFS')
                exploration_order, visited_cells, path_to_goal = algorithm_module.greedy_bfs_search(m, goal=goal_position)
            else:
                raise ValueError("Unknown algorithm choice!")

            # If a path to goal is found, show it in the PYTHON MAZE WORLD window
            if path_to_goal:
                agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
                agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)
                agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

                m.tracePath({agent_bfs: exploration_order}, delay=1)
                m.tracePath({agent_trace: path_to_goal}, delay=1)
                m.tracePath({agent_goal: visited_cells}, delay=1)

                textLabel(m, 'Goal Position', str(goal_position))
                textLabel(m, f'{algorithm_choice.capitalize()} Path Length', len(path_to_goal) + 1)
                textLabel(m, f'{algorithm_choice.capitalize()} Search Length', len(exploration_order))

                # Run the maze visualization in the PYTHON MAZE WORLD window
                m.run()

            else:
                raise ValueError("No path found to the goal!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Create and run the Tkinter app
if __name__ == "__main__":
    app = MazeSolverApp()
    app.mainloop()
