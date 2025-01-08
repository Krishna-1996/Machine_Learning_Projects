# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Updated data
algorithms = ["A*", "BFS", "DFS", "Greedy_BFS"]
obstacle_percentages = [0, 10, 30, 50]
path_lengths = [
    [149, 149, 161, 231],  # A*
    [149, 149, 161, 231],  # BFS
    [167, 185, 327, 259],  # DFS
    [177, 209, 211, 271],  # Greedy_BFS
]
exploration_lengths = [
    [2492, 2618, 3321, 2198],  # A*
    [4998, 4984, 4889, 2389],  # BFS
    [301, 327, 700, 832],      # DFS
    [294, 402, 397, 655],      # Greedy_BFS
]

# %%
# Convert to NumPy arrays for easier manipulation
path_lengths = np.array(path_lengths)
exploration_lengths = np.array(exploration_lengths)

# %%
# Bar Chart: Path Length vs Obstacle Percentage
plt.figure(figsize=(10, 6))
width = 0.2  # Bar width
x = np.arange(len(obstacle_percentages))
for i, algo in enumerate(algorithms):
    plt.bar(x + i * width, path_lengths[i], width, label=algo)

plt.xticks(x + width * 1.5, obstacle_percentages)
plt.xlabel("Obstacle Percentage")
plt.ylabel("Path Length")
plt.title("Path Length vs Obstacle Percentage")
plt.legend()
plt.show()

# %%
# Bar Chart: Exploration Length vs Obstacle Percentage
plt.figure(figsize=(10, 6))
for i, algo in enumerate(algorithms):
    plt.bar(x + i * width, exploration_lengths[i], width, label=algo)

plt.xticks(x + width * 1.5, obstacle_percentages)
plt.xlabel("Obstacle Percentage")
plt.ylabel("Exploration Length")
plt.title("Exploration Length vs Obstacle Percentage")
plt.legend()
plt.show()

# %%
# Line Chart: Path Length & Exploration Length (Combined)
plt.figure(figsize=(12, 8))
for i, algo in enumerate(algorithms):
    plt.plot(obstacle_percentages, path_lengths[i], marker='o', label=f"{algo} - Path")
    plt.plot(obstacle_percentages, exploration_lengths[i], marker='x', linestyle='--', label=f"{algo} - Exploration")

plt.xlabel("Obstacle Percentage")
plt.ylabel("Length")
plt.title("Path Length & Exploration Length vs Obstacle Percentage")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Stacked Bar Chart: Total Length (Path + Exploration)
plt.figure(figsize=(10, 6))
for i, algo in enumerate(algorithms):
    plt.bar(x + i * width, path_lengths[i], width, label=f"{algo} - Path")
    plt.bar(x + i * width, exploration_lengths[i], width, bottom=path_lengths[i], label=f"{algo} - Exploration")

plt.xticks(x + width * 1.5, obstacle_percentages)
plt.xlabel("Obstacle Percentage")
plt.ylabel("Total Length")
plt.title("Total Length (Path + Exploration) vs Obstacle Percentage")
plt.legend()
plt.show()

# %%
# Heatmap: Path Lengths
plt.figure(figsize=(8, 6))
sns.heatmap(path_lengths, annot=True, fmt="d", xticklabels=obstacle_percentages, yticklabels=algorithms, cmap="Blues")
plt.title("Heatmap of Path Lengths")
plt.xlabel("Obstacle Percentage")
plt.ylabel("Algorithm")
plt.show()

# %%
# Heatmap: Exploration Lengths
plt.figure(figsize=(8, 6))
sns.heatmap(exploration_lengths, annot=True, fmt="d", xticklabels=obstacle_percentages, yticklabels=algorithms, cmap="Greens")
plt.title("Heatmap of Exploration Lengths")
plt.xlabel("Obstacle Percentage")
plt.ylabel("Algorithm")
plt.show()