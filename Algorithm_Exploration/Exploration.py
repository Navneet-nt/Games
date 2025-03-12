import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
from collections import deque
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import pandas as pd
import imageio
from io import BytesIO

# Set page config
st.set_page_config(page_title="Multi-Goal Navigation", layout="wide")

class MultiGoalNavigation:
    def __init__(self, grid, start, goals, exit_pos, weights=None):
        """
        Initialize the navigation problem
        
        Args:
            grid: 2D array where 1 represents obstacles and 0 represents free space
            start: tuple (x, y) representing the starting position
            goals: list of tuples [(x1, y1), (x2, y2), ...] representing goal positions
            exit_pos: tuple (x, y) representing the exit position
            weights: list of weights for each goal (if None, all goals have equal weight of 1)
        """
        self.grid = grid
        self.start = start
        self.goals = goals
        self.exit_pos = exit_pos
        self.weights = weights if weights else [1] * len(goals)
        self.rows, self.cols = len(grid), len(grid[0])
        
        # For visualization
        self.exploration_history = []
        self.final_path = None
        self.total_cost = None
        self.total_states_explored = 0
        
    def is_valid(self, pos):
        """Check if a position is valid (within grid bounds and not an obstacle)"""
        x, y = pos
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x][y] == 0
    
    def get_neighbors(self, pos):
        """Get valid neighboring positions (4-way movement: up, down, left, right)"""
        x, y = pos
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        return [(x + dx, y + dy) for dx, dy in directions if self.is_valid((x + dx, y + dy))]
    
    def record_state(self, current_pos, path, collected_goals, frontier=None, cost=None):
        """Record the current state for visualization"""
        # Add to exploration history with current time
        self.exploration_history.append({
            'position': current_pos,
            'path': path.copy() if path else [],
            'collected_goals': collected_goals.copy() if collected_goals else set(),
            'frontier': frontier.copy() if frontier else set(),
            'cost': cost,
            'time': time.time()
        })
        self.total_states_explored += 1
    
    def bfs(self):
        """
        Breadth-First Search algorithm to find a path collecting all goals before exit
        
        Returns:
            path: list of positions [(x1, y1), (x2, y2), ...] representing the path
            total_cost: total cost of the path
        """
        # Reset visualization data
        self.exploration_history = []
        self.final_path = None
        self.total_cost = None
        self.total_states_explored = 0
        
        # Initialize
        queue = deque([(self.start, [self.start], set())])  # (current_pos, path, collected_goals)
        visited = set([(self.start, frozenset())])  # (pos, collected_goals)
        
        while queue:
            current, path, collected_goals = queue.popleft()
            
            # Record state for visualization
            frontier = {item[0] for item in queue}
            self.record_state(current, path, collected_goals, frontier)
            
            # Check if we've reached all goals and exit
            if len(collected_goals) == len(self.goals) and current == self.exit_pos:
                self.final_path = path
                self.total_cost = len(path) - 1  # -1 because start doesn't count as a step
                return path, self.total_cost
            
            # If we've collected all goals, only consider moving towards exit
            if len(collected_goals) == len(self.goals) and current != self.exit_pos:
                neighbors = self.get_neighbors(current)
                for neighbor in neighbors:
                    new_state = (neighbor, frozenset(collected_goals))
                    if new_state not in visited:
                        visited.add(new_state)
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path, collected_goals))
                continue
            
            # Otherwise, explore all neighbors
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                # Check if we've reached a new goal
                new_collected = collected_goals.copy()
                if neighbor in self.goals and neighbor not in collected_goals:
                    new_collected.add(neighbor)
                
                new_state = (neighbor, frozenset(new_collected))
                if new_state not in visited:
                    visited.add(new_state)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, new_collected))
        
        return None, float('inf')
    
    def dfs(self):
        """
        Depth-First Search algorithm to find a path collecting all goals before exit
        
        Returns:
            path: list of positions [(x1, y1), (x2, y2), ...] representing the path
            total_cost: total cost of the path
        """
        # Reset visualization data
        self.exploration_history = []
        self.final_path = None
        self.total_cost = None
        self.total_states_explored = 0
        
        # Initialize
        stack = [(self.start, [self.start], set())]  # (current_pos, path, collected_goals)
        visited = set([(self.start, frozenset())])  # (pos, collected_goals)
        
        while stack:
            current, path, collected_goals = stack.pop()
            
            # Record state for visualization
            frontier = {item[0] for item in stack}
            self.record_state(current, path, collected_goals, frontier)
            
            # Check if we've reached all goals and exit
            if len(collected_goals) == len(self.goals) and current == self.exit_pos:
                self.final_path = path
                self.total_cost = len(path) - 1
                return path, self.total_cost
            
            # If we've collected all goals, only consider moving towards exit
            if len(collected_goals) == len(self.goals) and current != self.exit_pos:
                neighbors = self.get_neighbors(current)
                for neighbor in reversed(neighbors):
                    new_state = (neighbor, frozenset(collected_goals))
                    if new_state not in visited:
                        visited.add(new_state)
                        new_path = path + [neighbor]
                        stack.append((neighbor, new_path, collected_goals))
                continue
            
            # Otherwise, explore all neighbors
            neighbors = self.get_neighbors(current)
            for neighbor in reversed(neighbors):
                new_collected = collected_goals.copy()
                if neighbor in self.goals and neighbor not in collected_goals:
                    new_collected.add(neighbor)
                
                new_state = (neighbor, frozenset(new_collected))
                if new_state not in visited:
                    visited.add(new_state)
                    new_path = path + [neighbor]
                    stack.append((neighbor, new_path, new_collected))
        
        return None, float('inf')
    
    def ucs(self):
        """
        Uniform Cost Search algorithm to find a path collecting all goals before exit
        Takes into account the weights of goals
        
        Returns:
            path: list of positions [(x1, y1), (x2, y2), ...] representing the path
            total_cost: total cost of the path
        """
        # Reset visualization data
        self.exploration_history = []
        self.final_path = None
        self.total_cost = None
        self.total_states_explored = 0
        
        # Initialize
        # (cost, id, current_pos, path, collected_goals)
        counter = 0  # Unique ID for tie-breaking
        priority_queue = [(0, counter, self.start, [self.start], set())]
        visited = set([(self.start, frozenset())])
        
        while priority_queue:
            cost, _, current, path, collected_goals = heapq.heappop(priority_queue)
            
            # Record state for visualization
            frontier = {item[2] for item in priority_queue}
            self.record_state(current, path, collected_goals, frontier, cost)
            
            if len(collected_goals) == len(self.goals) and current == self.exit_pos:
                self.final_path = path
                self.total_cost = cost
                return path, cost
            
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                new_cost = cost + 1  # base cost for movement
                
                new_collected = collected_goals.copy()
                if neighbor in self.goals and neighbor not in collected_goals:
                    goal_idx = self.goals.index(neighbor)
                    new_cost -= self.weights[goal_idx]
                    new_collected.add(neighbor)
                
                new_state = (neighbor, frozenset(new_collected))
                if new_state not in visited:
                    visited.add(new_state)
                    new_path = path + [neighbor]
                    counter += 1
                    heapq.heappush(priority_queue, (new_cost, counter, neighbor, new_path, new_collected))
        
        return None, float('inf')
    
    def heuristic(self, pos, remaining_goals, exit_pos):
        """
        Heuristic function for A* algorithm
        Estimates the cost from current position to collect all remaining goals and reach exit
        
        Args:
            pos: current position (x, y)
            remaining_goals: set of goal positions that haven't been collected
            exit_pos: exit position (x, y)
            
        Returns:
            estimated cost to complete the mission
        """
        if not remaining_goals:
            return abs(pos[0] - exit_pos[0]) + abs(pos[1] - exit_pos[1])
        
        # Estimate by finding the closest goal
        closest_goal_dist = min(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) for goal in remaining_goals)
        
        # Estimate extra cost to traverse between remaining goals (simple MST-like approach)
        remaining_goals_list = list(remaining_goals)
        remaining_dist = 0
        if len(remaining_goals_list) > 1:
            for i in range(len(remaining_goals_list)):
                nearest = min(
                    abs(remaining_goals_list[i][0] - remaining_goals_list[j][0]) + 
                    abs(remaining_goals_list[i][1] - remaining_goals_list[j][1])
                    for j in range(len(remaining_goals_list)) if i != j
                )
                remaining_dist += nearest / 2  # division to avoid double counting
        else:
            remaining_dist = 0
        
        # Estimate from last goal to exit
        exit_dist = min(abs(goal[0] - exit_pos[0]) + abs(goal[1] - exit_pos[1]) for goal in remaining_goals)
        
        return closest_goal_dist + remaining_dist + exit_dist
    
    def a_star(self):
        """
        A* algorithm to find optimal path collecting all goals before exit
        Uses weights and heuristic
        
        Returns:
            path: list of positions [(x1, y1), (x2, y2), ...] representing the path
            total_cost: total cost of the path
        """
        # Reset visualization data
        self.exploration_history = []
        self.final_path = None
        self.total_cost = None
        self.total_states_explored = 0
        
        # Initialize
        # (f_cost, id, g_cost, current_pos, path, collected_goals)
        counter = 0
        open_set = [(0, counter, 0, self.start, [self.start], set())]
        closed_set = set()
        g_scores = {(self.start, frozenset()): 0}
        
        while open_set:
            _, _, g_cost, current, path, collected_goals = heapq.heappop(open_set)
            current_state = (current, frozenset(collected_goals))
            
            # Record state for visualization
            frontier = {item[3] for item in open_set}
            self.record_state(current, path, collected_goals, frontier, g_cost)
            
            if current_state in closed_set:
                continue
            
            closed_set.add(current_state)
            
            if len(collected_goals) == len(self.goals) and current == self.exit_pos:
                self.final_path = path
                self.total_cost = g_cost
                return path, g_cost
            
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                new_g_cost = g_cost + 1
                new_collected = collected_goals.copy()
                if neighbor in self.goals and neighbor not in collected_goals:
                    goal_idx = self.goals.index(neighbor)
                    new_g_cost -= self.weights[goal_idx]
                    new_collected.add(neighbor)
                
                neighbor_state = (neighbor, frozenset(new_collected))
                if neighbor_state in closed_set:
                    continue
                
                if neighbor_state not in g_scores or new_g_cost < g_scores[neighbor_state]:
                    g_scores[neighbor_state] = new_g_cost
                    remaining_goals = set(self.goals) - new_collected
                    h_cost = self.heuristic(neighbor, remaining_goals, self.exit_pos)
                    f_cost = new_g_cost + h_cost
                    new_path = path + [neighbor]
                    counter += 1
                    heapq.heappush(open_set, (f_cost, counter, new_g_cost, neighbor, new_path, new_collected))
        
        return None, float('inf')

def create_grid_with_obstacles(rows, cols, obstacles_percentage=0.2):
    """Create a grid with random obstacles"""
    grid = np.zeros((rows, cols), dtype=int)
    num_obstacles = int(rows * cols * obstacles_percentage)
    obstacles = np.random.choice(rows * cols, num_obstacles, replace=False)
    for obs in obstacles:
        r, c = divmod(obs, cols)
        grid[r, c] = 1
    # Ensure start and exit are clear
    grid[0, 0] = 0
    grid[rows-1, cols-1] = 0
    # Clear a path from start to exit
    for i in range(rows):
        grid[i, 0] = 0
    for j in range(cols):
        grid[rows-1, j] = 0
    return grid

def create_random_scenario(rows=10, cols=10, num_goals=3, obstacle_percentage=0.2):
    """Create a random scenario with start, goals, exit, and obstacles"""
    grid = create_grid_with_obstacles(rows, cols, obstacle_percentage)
    start = (0, 0)
    exit_pos = (rows-1, cols-1)
    goals = []
    weights = []
    available_positions = []
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0 and (i, j) != start and (i, j) != exit_pos:
                available_positions.append((i, j))
    if len(available_positions) < num_goals:
        num_goals = len(available_positions)
    goal_positions = np.random.choice(len(available_positions), num_goals, replace=False)
    for pos in goal_positions:
        goals.append(available_positions[pos])
        weights.append(np.random.randint(1, 6))
    return grid, start, goals, exit_pos, weights

def create_predefined_scenario():
    """Create a predefined scenario for consistent demonstrations"""
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    start = (0, 0)
    goals = [(2, 2), (6, 3), (8, 8)]
    exit_pos = (9, 9)
    weights = [3, 2, 5]
    return grid, start, goals, exit_pos, weights

def visualize_exploration(nav, step_idx, fig, ax):
    """Visualize exploration at a specific step"""
    if step_idx >= len(nav.exploration_history):
        step_idx = len(nav.exploration_history) - 1
    step_data = nav.exploration_history[step_idx]
    ax.clear()
    grid_vis = np.zeros((nav.rows, nav.cols, 3))
    # Set background colors: obstacles black, free space white.
    for i in range(nav.rows):
        for j in range(nav.cols):
            grid_vis[i, j] = [0, 0, 0] if nav.grid[i][j] == 1 else [1, 1, 1]
    # Mark explored positions (light blue)
    for i in range(step_idx + 1):
        pos = nav.exploration_history[i]['position']
        if (pos != nav.start and pos not in nav.goals and pos != nav.exit_pos and 
            pos not in step_data['path']):
            grid_vis[pos[0], pos[1]] = [0.8, 0.8, 1]
    # Mark frontier (light red)
    for pos in step_data['frontier']:
        if (pos != nav.start and pos not in nav.goals and pos != nav.exit_pos and 
            pos not in step_data['path']):
            grid_vis[pos[0], pos[1]] = [1, 0.8, 0.8]
    # Mark current path (blue)
    for pos in step_data['path']:
        if pos != nav.start and pos not in nav.goals and pos != nav.exit_pos:
            grid_vis[pos[0], pos[1]] = [0, 0, 1]
    # Mark current position (yellow)
    current_pos = step_data['position']
    if current_pos != nav.start and current_pos not in nav.goals and current_pos != nav.exit_pos:
        grid_vis[current_pos[0], current_pos[1]] = [1, 1, 0]
    # Mark start, goals, exit
    grid_vis[nav.start[0], nav.start[1]] = [0, 1, 0]
    for goal in nav.goals:
        if goal in step_data['collected_goals']:
            grid_vis[goal[0], goal[1]] = [0, 0.8, 0]
        else:
            grid_vis[goal[0], goal[1]] = [1, 0, 0]
    grid_vis[nav.exit_pos[0], nav.exit_pos[1]] = [1, 0, 1]
    ax.imshow(grid_vis)
    ax.plot(nav.start[1], nav.start[0], 'go', markersize=10, label='Start')
    ax.plot([goal[1] for goal in nav.goals], [goal[0] for goal in nav.goals], 'ro', markersize=10, label='Goals')
    ax.plot(nav.exit_pos[1], nav.exit_pos[0], 'mo', markersize=10, label='Exit')
    ax.plot(current_pos[1], current_pos[0], 'yo', markersize=8, label='Current')
    # Annotate goals with numbers and weights
    for i, goal in enumerate(nav.goals):
        if goal in step_data['collected_goals']:
            ax.text(goal[1], goal[0], f"{i+1}✓\n({nav.weights[i]})", color='white', 
                    ha='center', va='center', fontweight='bold')
        else:
            ax.text(goal[1], goal[0], f"{i+1}\n({nav.weights[i]})", color='white', 
                    ha='center', va='center', fontweight='bold')
    title = f"Step {step_idx+1}/{len(nav.exploration_history)}"
    if step_data['cost'] is not None:
        title += f", Cost: {step_data['cost']}"
    ax.set_title(title)
    ax.grid(True)
    legend_elements = [
        Rectangle((0, 0), 1, 1, color='green', label='Start'),
        Rectangle((0, 0), 1, 1, color='red', label='Goal (Uncollected)'),
        Rectangle((0, 0), 1, 1, color='darkgreen', label='Goal (Collected)'),
        Rectangle((0, 0), 1, 1, color='purple', label='Exit'),
        Rectangle((0, 0), 1, 1, color='yellow', label='Current Position'),
        Rectangle((0, 0), 1, 1, color='blue', label='Current Path'),
        Rectangle((0, 0), 1, 1, color='lightblue', label='Explored'),
        Rectangle((0, 0), 1, 1, color='lightcoral', label='Frontier')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              ncol=4, fancybox=True, shadow=True)
    return fig

def visualize_final_path(nav, fig, ax):
    """Visualize the final solution path"""
    ax.clear()
    grid_vis = np.zeros((nav.rows, nav.cols, 3))
    for i in range(nav.rows):
        for j in range(nav.cols):
            grid_vis[i, j] = [0, 0, 0] if nav.grid[i][j] == 1 else [1, 1, 1]
    for pos in nav.final_path:
        if pos != nav.start and pos not in nav.goals and pos != nav.exit_pos:
            grid_vis[pos[0], pos[1]] = [0, 0, 1]
    grid_vis[nav.start[0], nav.start[1]] = [0, 1, 0]
    for goal in nav.goals:
        grid_vis[goal[0], goal[1]] = [1, 0, 0]
    grid_vis[nav.exit_pos[0], nav.exit_pos[1]] = [1, 0, 1]
    ax.imshow(grid_vis)
    ax.set_title(f"Final Path (Cost: {nav.total_cost}, States Explored: {nav.total_states_explored})")
    for i in range(len(nav.final_path) - 1):
        dx = nav.final_path[i+1][1] - nav.final_path[i][1]
        dy = nav.final_path[i+1][0] - nav.final_path[i][0]
        ax.arrow(nav.final_path[i][1], nav.final_path[i][0], dx, dy, head_width=0.3, head_length=0.3, fc='blue', ec='blue')
    for i, goal in enumerate(nav.goals):
        ax.text(goal[1], goal[0], f"{i+1}\n({nav.weights[i]})", color='white', 
                ha='center', va='center', fontweight='bold')
    legend_elements = [
        Rectangle((0, 0), 1, 1, color='green', label='Start'),
        Rectangle((0, 0), 1, 1, color='red', label='Goal'),
        Rectangle((0, 0), 1, 1, color='purple', label='Exit'),
        Rectangle((0, 0), 1, 1, color='blue', label='Path')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              ncol=4, fancybox=True, shadow=True)
    ax.grid(True)
    return fig

def run_algorithm(nav, algorithm):
    """Run the selected algorithm and return the solution"""
    if algorithm == "BFS":
        return nav.bfs()
    elif algorithm == "DFS":
        return nav.dfs()
    elif algorithm == "UCS":
        return nav.ucs()
    elif algorithm == "A*":
        return nav.a_star()
    return None, None

def main():
    st.title("Multi-Goal Navigation Visualization")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    algorithm = st.sidebar.selectbox("Select Algorithm", ["BFS", "DFS", "UCS", "A*"])
    scenario_type = st.sidebar.radio("Scenario Type", ["Predefined", "Random"])
    
    if scenario_type == "Random":
        rows = st.sidebar.slider("Grid Rows", 5, 20, 10)
        cols = st.sidebar.slider("Grid Columns", 5, 20, 10)
        num_goals = st.sidebar.slider("Number of Goals", 1, 5, 3)
        obstacle_percentage = st.sidebar.slider("Obstacle Percentage", 0.0, 0.4, 0.2, 0.05)
        grid, start, goals, exit_pos, weights = create_random_scenario(rows, cols, num_goals, obstacle_percentage)
    else:
        grid, start, goals, exit_pos, weights = create_predefined_scenario()
    
    nav = MultiGoalNavigation(grid, start, goals, exit_pos, weights)
    
    tab1, tab2, tab3 = st.tabs(["Algorithm Exploration", "Solution Path", "Statistics"])
    
    with tab1:
        st.header(f"{algorithm} Algorithm Exploration")
        if st.button("Run Algorithm"):
            with st.spinner(f"Running {algorithm}..."):
                path, cost = run_algorithm(nav, algorithm)
            if path:
                st.success(f"Path found with {algorithm}! Cost: {cost}, Path length: {len(path)}")
            else:
                st.error(f"No path found with {algorithm}!")
            
            # Slider for exploration steps
            if nav.exploration_history:
                step = st.slider("Exploration Step", 1, len(nav.exploration_history), 1)
                fig, ax = plt.subplots(figsize=(6,6))
                visualize_exploration(nav, step-1, fig, ax)
                st.pyplot(fig)
                
                # Option to animate the exploration
                if st.checkbox("Animate Exploration"):
                    frames = []
                    for i in range(len(nav.exploration_history)):
                        fig_anim, ax_anim = plt.subplots(figsize=(6,6))
                        visualize_exploration(nav, i, fig_anim, ax_anim)
                        buf = BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        image = imageio.imread(buf)
                        frames.append(image)
                        plt.close(fig_anim)
                    gif_buf = BytesIO()
                    imageio.mimsave(gif_buf, frames, format="gif", duration=0.5)
                    st.image(gif_buf.getvalue(), caption="Exploration Animation")
    
    with tab2:
        st.header("Solution Path")
        if nav.final_path:
            fig2, ax2 = plt.subplots(figsize=(6,6))
            visualize_final_path(nav, fig2, ax2)
            st.pyplot(fig2)
        else:
            st.write("No solution path available. Please run an algorithm first.")
    
    with tab3:
        st.header("Statistics")
        if nav.final_path:
            st.write(f"Total Cost: {nav.total_cost}")
            st.write(f"Total States Explored: {nav.total_states_explored}")
            st.write(f"Final Path Length: {len(nav.final_path)}")
            df = pd.DataFrame([{
                "Step": i+1,
                "Position": data["position"],
                "Cost": data["cost"],
                "Collected Goals": list(data["collected_goals"])
            } for i, data in enumerate(nav.exploration_history)])
            st.dataframe(df)
        else:
            st.write("No statistics available. Please run an algorithm first.")

if __name__ == '__main__':
    main()
