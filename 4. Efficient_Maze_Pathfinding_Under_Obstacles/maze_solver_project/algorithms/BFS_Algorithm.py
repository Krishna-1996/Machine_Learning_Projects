from collections import deque

# Modified BFS Search function to handle KeyError
def bfs_search(m, goal):
    # Ensure the starting point (0, 0) is valid
    start = (0, 0)  # Starting point
    if start not in m.maze_map:
        raise ValueError("Start position (0, 0) is not valid in maze.")

    # Directions for moving (N, S, E, W)
    directions = ['N', 'S', 'E', 'W']
    queue = deque([start])  # BFS queue
    parent_map = {}  # To track the path
    visited = set([start])  # Set of visited cells
    exploration_order = []  # Cells visited during BFS
    path_to_goal = []  # Path from start to goal

    while queue:
        current = queue.popleft()
        exploration_order.append(current)

        if current == goal:
            # Reconstruct the path to the goal
            while current in parent_map:
                path_to_goal.append(current)
                current = parent_map[current]
            path_to_goal.append(start)
            path_to_goal.reverse()
            break

        # Explore neighbors in 4 directions (N, S, E, W)
        for direction in directions:
            if direction == 'N':
                neighbor = (current[0] - 1, current[1])
            elif direction == 'S':
                neighbor = (current[0] + 1, current[1])
            elif direction == 'E':
                neighbor = (current[0], current[1] + 1)
            elif direction == 'W':
                neighbor = (current[0], current[1] - 1)

            if neighbor not in visited:
                # Check if the neighbor is within bounds and if the wall in that direction is not blocked
                if neighbor in m.maze_map and m.maze_map[current].get(direction, 0) == 1:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    parent_map[neighbor] = current

    # Check if goal was reached
    if goal not in path_to_goal:
        path_to_goal = []  # No path found to goal

    return exploration_order, visited, path_to_goal
