import csv
import random

def load_maze_from_csv(file_path):
    """Load the maze from the main CSV file."""
    maze_map = {}
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        
        for row in reader:
            coords = eval(row[0])  # Convert string to tuple (row, col)
            E, W, N, S = map(int, row[1:])  # Convert direction values to integers
            maze_map[coords] = {"E": E, "W": W, "N": N, "S": S}
    
    return maze_map


def add_obstacles(maze_map, obstacle_percentage=20):
    """Add random obstacles to the maze."""
    total_cells = len(maze_map)
    num_obstacles = int(total_cells * (obstacle_percentage / 100))
    
    # Randomly select obstacle positions
    blocked_cells = random.sample(list(maze_map.keys()), num_obstacles)
    
    for (row, col) in blocked_cells:
        # Randomly block directions (E, W, N, S) for obstacles
        if random.choice([True, False]):
            maze_map[(row, col)]["E"] = 0
            maze_map[(row, col)]["W"] = 0
        if random.choice([True, False]):
            maze_map[(row, col)]["N"] = 0
            maze_map[(row, col)]["S"] = 0
    
    return maze_map


def save_maze_to_csv(maze_map, file_path):
    """Save the maze (with obstacles) to a CSV file."""
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['cell', 'E', 'W', 'N', 'S'])
        
        # Write maze data with obstacles
        for (row, col), directions in maze_map.items():
            E = directions.get('E', 1)  # Default to 1 (open) if not specified
            W = directions.get('W', 1)  # Default to 1 (open) if not specified
            N = directions.get('N', 1)  # Default to 1 (open) if not specified
            S = directions.get('S', 1)  # Default to 1 (open) if not specified
            writer.writerow([(row, col), E, W, N, S])
    print(f"Maze with obstacles saved to {file_path}")


def main():
    # Paths to the original and modified maze CSV files
    original_csv_path = 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/Advance Version of obstacle project/Maze_1_90_loopPercent.csv'
    # D:\Masters Projects\Master-In-AI\Foundation of Artificial Intelligence\Project 3 ICA\Advance Version of obstacle project
    modified_csv_path = 'maze_with_obstacles.csv'
    
    # Load the original maze CSV
    maze_map = load_maze_from_csv(original_csv_path)
    
    # Add random obstacles to the maze
    maze_map_with_obstacles = add_obstacles(maze_map, obstacle_percentage=50)  # Change obstacle percentage as needed
    
    # Save the modified maze with obstacles to a new CSV
    save_maze_to_csv(maze_map_with_obstacles, modified_csv_path)


if __name__ == '__main__':
    main()




