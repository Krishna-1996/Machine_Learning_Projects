# **Efficient_Maze_Pathfinding_Under_Obstacles**

In this project, we explore the implementation and performance of **four pathfinding algorithms** — **Breadth-First Search (BFS)**, **Depth-First Search (DFS)**, **A\***, and **Greedy BFS** — on various mazes with different obstacle densities **0%** — **10%** — **30%** — and **50%**. The goal is to analyze each algorithm's efficiency and accuracy in terms of **path length** and **exploration length** while navigating through obstacle-laden mazes. In new update I have also added 5 goal positions i.e. **Top Left** — **Top Right** — **Bottom Left** — **Bottom Right** — and **Centre**.

---

## **Table of Contents**

- **[Introduction](#introduction)**
- **[Algorithms Overview](#algorithms-overview)**
- **[Maze Generation](#maze-generation)**
- **[Visualization](#visualization)**
- **[Experimental Results](#experimental-results)**
- **[Discussion and Future Work](#discussion-and-future-work)**
- **[Conclusion](#conclusion)**

---

## **Introduction**

The primary goal of this project is to compare the efficiency of four maze-solving algorithms — **BFS**, **DFS**, **A\***, and **Greedy BFS** — across different obstacle densities. Each algorithm was tested in mazes with obstacle densities of **0%**, **10%**, **30%**, and **50%**, different goal positions i.e. **Top Left** — **Top Right** — **Bottom Left** — **Bottom Right** — and **Centre**.

### **Performance Metrics Evaluated:**

- **Path Length**: The shortest distance from the start point to the goal.
- **Exploration Length**: The total number of cells explored by the algorithm before reaching the goal.

In addition to performance metrics, visualizations of the algorithm's exploration process are included to give a clear understanding of how each algorithm works. 


---

## **Algorithms Overview**

### 1. **Breadth-First Search (BFS)**  
- **Approach**: Explores all possible paths level by level and guarantees the shortest path in an unweighted grid.
- **Performance**: Optimal for finding the shortest path but inefficient in terms of exploration, especially in dense mazes.

### 2. **Depth-First Search (DFS)**  
- **Approach**: Explores a path as far as possible, backtracking when necessary. Does not guarantee the shortest path.
- **Performance**: Works well in sparse mazes but becomes inefficient in dense mazes, often backtracking excessively.

### 3. **A\***  
- **Approach**: Uses a heuristic (Manhattan Distance) to find the optimal path efficiently. Balances between the path taken and the remaining distance to the goal.
- **Performance**: Optimal for finding the shortest path and performs well in both sparse and dense mazes, though its exploration time increases in denser environments.

### 4. **Greedy BFS**  
- **Approach**: Uses only a heuristic to prioritize cells closer to the goal, potentially taking suboptimal paths.
- **Performance**: Faster than BFS and DFS, but may take longer detours, leading to less optimal paths.

---

## **Maze Generation**

The maze is generated from the **CSV file** `Maze_1_90_loopPercent.csv`, which has a **90% loop percentage**. Obstacles are added based on the specified densities (0%, 10%, 30%, and 50%).

### **Maze Creation Process:**
- **Loading the maze CSV file**.
- **Adding obstacles** based on the specified density.
- **Saving the modified maze** for use in the algorithms.

---

## **Visualization**

The maze and the agent's movements are visualized using the **Pyamaze** library. The visualizations include:

- **Open cells**: Shown in **white**.
- **Obstacles**: Shown in **black**.
- **The agent**: Depicted with different colors based on the algorithm used:
  - **Red** for BFS and Greedy BFS.
  - **Yellow** for A\*.
  - **Blue** for DFS.
- **The goal**: Marked with a **green square**.

Each algorithm's exploration and pathfinding process is displayed in **real-time** to understand how the algorithm behaves as it navigates the maze.

---

## **Experimental Results**
**All Experiment are done with goal position at top-left corner for better distance and highest result.**
### **Path Length Comparison** (Shortest Path)

| Algorithm        | 0% Density | 10% Density | 30% Density | 50% Density |
|------------------|------------|-------------|-------------|-------------|
| **A\***           | **149**    | **149**     | **161**     | **231**     |
| **BFS**           | **149**    | **149**     | **161**     | **231**     |
| **DFS**           | **167**    | **185**     | **327**     | **259**     |
| **Greedy BFS**    | **177**    | **209**     | **211**     | **271**     |

- **Key Takeaway**: **A\*** and **BFS** consistently find the shortest path, with **A\*** maintaining a slightly more efficient path length.  
- **DFS** struggles with denser mazes, while **Greedy BFS** sometimes takes suboptimal paths but is faster in finding a solution.

### **Exploration Length Comparison** (Total Explored Cells)

| Algorithm        | 0% Density | 10% Density | 30% Density | 50% Density |
|------------------|------------|-------------|-------------|-------------|
| **A\***           | **2492**   | **2618**    | **3321**    | **2198**    |
| **BFS**           | **4998**   | **4984**    | **4889**    | **2389**    |
| **DFS**           | **301**    | **327**     | **700**     | **832**     |
| **Greedy BFS**    | **294**    | **402**     | **397**     | **655**     |

- **Key Takeaway**: 
  - **A\*** exhibits efficient exploration, especially in denser mazes.  
  - **BFS** explores significantly more cells than **A\*** or **Greedy BFS**, particularly in obstacle-rich environments.  
  - **DFS** has the lowest exploration length in sparse mazes but becomes inefficient as the maze complexity increases.

---

## **Discussion and Future Work**

### **Future Work**
- **Optimizations**: Implement advanced data structures like **priority queues** for **A\*** and **BFS** to improve performance.
- **Additional Algorithms**: Consider including other algorithms such as **Dijkstra’s** and **Bidirectional BFS** for further comparison.
- **Dynamic Obstacles**: Introduce obstacles that change during the solving process to add an extra layer of complexity.
- **Interactive Visualization**: Enhance the GUI with **real-time exploration tracking** and the ability to interactively design mazes.

### **Discussion**
- The results highlight the trade-off between **algorithm efficiency** and **pathfinding accuracy**:
  - **A\*** is the best choice for optimal paths but can be slow in dense mazes.
  - **Greedy BFS** offers faster solutions but may not always provide the shortest path.
  - **BFS** guarantees the shortest path but at the cost of excessive exploration.
  - **DFS** works well in simple, sparse mazes but becomes inefficient as the maze density increases.

---

## **Conclusion**

The choice of the best pathfinding algorithm depends on the specific task and the maze's complexity:

- **A\*** is ideal when finding the shortest path is a priority.
- **Greedy BFS** is suitable for scenarios where **speed** is more important than path optimality.
- **BFS** is reliable but inefficient in **large, obstacle-dense mazes**.
- **DFS** is effective in **sparse mazes** but becomes inefficient as the maze density increases.

This study emphasizes that when selecting a maze-solving algorithm, the trade-off between **path optimality** and **exploration efficiency** must be carefully considered based on the specific requirements of the application.

---

**Happy Pathfinding!** 🚀
