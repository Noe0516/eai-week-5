"""
Problem 3.4: Building a Knowledge-Based Agent for Hazardous Warehouse

This implementation uses Z3 theorem prover to reason about safe squares
in the hazardous warehouse environment.

"""

# Note: Requires z3-solver to be installed
# pip install z3-solver

try:
    from z3 import Bool, Bools, Or, And, Not, Solver, sat, unsat
    Z3_AVAILABLE = True
except ImportError:
    print("WARNING: z3-solver not installed. Install with: pip install z3-solver")
    Z3_AVAILABLE = False
    # Mock for demonstration
    class Bool:
        def __init__(self, name): self.name = name
    class Solver:
        def add(self, clause): pass
        def check(self): return "sat"
        def model(self): return {}
        def push(self): pass
        def pop(self): pass

from hazardous_warehouse_env import HazardousWarehouseEnv, Action, Direction, Percept
from typing import Tuple, Set, List, Optional
from collections import deque


# ============================================================================
# TASK 1: SETUP AND EXPLORATION
# ============================================================================

def task1_z3_exploration():
    """
    Task 1: Verify Z3 setup and explore basic functionality
    """
    print("=" * 70)
    print("TASK 1: SETUP AND EXPLORATION")
    print("=" * 70)
    
    if not Z3_AVAILABLE:
        print("Z3 not available - skipping actual tests")
        return
    
    # Create boolean variables
    print("\n1. Creating boolean variables P and Q:")
    P, Q = Bools('P Q')
    print(f"   P = {P}, Q = {Q}")
    
    # Create solver and add biconditional
    print("\n2. Creating solver and adding P == Q:")
    s = Solver()
    s.add(P == Q)
    print("   Added: P ↔ Q")
    
    # Add fact
    print("\n3. Adding fact P:")
    s.add(P)
    print("   Added: P")
    
    # Check satisfiability
    print("\n4. Checking satisfiability:")
    result = s.check()
    print(f"   s.check() = {result}")
    
    # Inspect model
    if result == sat:
        print("\n5. Inspecting model:")
        model = s.model()
        print(f"   {model}")
        print(f"   P = {model[P]}, Q = {model[Q]}")
    
    # Test z3_entails
    print("\n6. Testing z3_entails function:")
    entails_Q = z3_entails(s, Q)
    print(f"   Does KB entail Q? {entails_Q}")
    print("   Expected: True (since P ↔ Q and P, therefore Q)")
    
    print("\n✓ Task 1 complete!")


def z3_entails(solver: Solver, query) -> bool:
    """
    Check if the knowledge base entails the query using push/pop
    
    The KB entails query iff KB ∧ ¬query is unsatisfiable
    
    Args:
        solver: Z3 solver containing the knowledge base
        query: Boolean formula to check
    
    Returns:
        True if KB entails query, False otherwise
    """
    if not Z3_AVAILABLE:
        return False
    
    solver.push()  # Save current state
    solver.add(Not(query))  # Add negation of query
    result = solver.check()  # Check if KB ∧ ¬query is satisfiable
    solver.pop()  # Restore state
    
    # If unsatisfiable, then KB entails query
    return result == unsat


# ============================================================================
# TASK 2: SYMBOLS AND PHYSICS
# ============================================================================

def damaged(x: int, y: int):
    """Boolean variable: Damaged floor at (x, y)"""
    return Bool(f'D_{x}_{y}')


def forklift_at(x: int, y: int):
    """Boolean variable: Forklift at (x, y)"""
    return Bool(f'F_{x}_{y}')


def creaking_at(x: int, y: int):
    """Boolean variable: Creaking perceived at (x, y)"""
    return Bool(f'C_{x}_{y}')


def rumbling_at(x: int, y: int):
    """Boolean variable: Rumbling perceived at (x, y)"""
    return Bool(f'R_{x}_{y}')


def safe(x: int, y: int):
    """Boolean variable: Square (x, y) is safe"""
    return Bool(f'S_{x}_{y}')


def get_adjacent_positions(x: int, y: int, width: int, height: int) -> List[Tuple[int, int]]:
    """Get valid adjacent positions (4-connectivity)"""
    adjacent = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 1 <= nx <= width and 1 <= ny <= height:
            adjacent.append((nx, ny))
    return adjacent


def build_warehouse_kb(width: int = 4, height: int = 4) -> Solver:
    """
    Build the knowledge base for the hazardous warehouse
    
    Encodes the physics:
    1. Creaking at location ↔ damaged floor in adjacent square
    2. Rumbling at location ↔ forklift in adjacent square  
    3. Safe square ↔ no damaged floor AND no forklift
    
    Args:
        width: Width of grid (columns 1 to width)
        height: Height of grid (rows 1 to height)
    
    Returns:
        Z3 Solver with warehouse physics encoded
    """
    print("\n" + "=" * 70)
    print("TASK 2: BUILDING WAREHOUSE KNOWLEDGE BASE")
    print("=" * 70)
    
    if not Z3_AVAILABLE:
        print("Z3 not available - returning mock solver")
        return Solver()
    
    solver = Solver()
    
    print(f"\nBuilding KB for {width}x{height} grid...")
    print(f"Coordinates: x ∈ [1, {width}], y ∈ [1, {height}]")
    
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            adjacent = get_adjacent_positions(x, y, width, height)
            
            # Rule 1: Creaking ↔ damaged floor in adjacent square
            if adjacent:
                damaged_adjacent = Or([damaged(ax, ay) for ax, ay in adjacent])
                solver.add(creaking_at(x, y) == damaged_adjacent)
            
            # Rule 2: Rumbling ↔ forklift in adjacent square
            if adjacent:
                forklift_adjacent = Or([forklift_at(ax, ay) for ax, ay in adjacent])
                solver.add(rumbling_at(x, y) == forklift_adjacent)
            
            # Rule 3: Safe ↔ no damaged floor AND no forklift
            solver.add(safe(x, y) == And(Not(damaged(x, y)), 
                                          Not(forklift_at(x, y))))
    
    # Verify KB is satisfiable
    print("\nVerifying KB is satisfiable...")
    result = solver.check()
    print(f"solver.check() = {result}")
    
    if result == sat:
        print("✓ Knowledge base is consistent (satisfiable)")
    else:
        print("✗ WARNING: Knowledge base is inconsistent!")
    
    print("\n✓ Task 2 complete!")
    
    return solver


# ============================================================================
# TASK 3: MANUAL REASONING
# ============================================================================

def task3_manual_reasoning():
    """
    Task 3: Manually test reasoning before building full agent
    """
    print("\n" + "=" * 70)
    print("TASK 3: MANUAL REASONING")
    print("=" * 70)
    
    if not Z3_AVAILABLE:
        print("Z3 not available - skipping manual reasoning")
        return
    
    # Build KB
    solver = build_warehouse_kb(width=4, height=4)
    
    # Step 1: Observation at (1, 1) - start position
    print("\n" + "-" * 70)
    print("OBSERVATION AT (1, 1): Initial position")
    print("-" * 70)
    
    # At (1,1), we know it's safe (we're standing on it)
    solver.add(safe(1, 1))
    print("TELL: S_1_1 (starting square is safe)")
    
    # Assume no creaking and no rumbling at start (typical scenario)
    solver.add(Not(creaking_at(1, 1)))
    solver.add(Not(rumbling_at(1, 1)))
    print("TELL: ¬C_1_1 (no creaking)")
    print("TELL: ¬R_1_1 (no rumbling)")
    
    # ASK: What can we infer?
    adjacent_to_1_1 = [(2, 1), (1, 2)]
    print(f"\nAdjacent to (1,1): {adjacent_to_1_1}")
    
    for x, y in adjacent_to_1_1:
        is_safe = z3_entails(solver, safe(x, y))
        print(f"ASK: Is ({x}, {y}) safe? {is_safe}")
    
    # Step 2: Move to (2, 1) and observe
    print("\n" + "-" * 70)
    print("OBSERVATION AT (2, 1): After moving East")
    print("-" * 70)
    
    solver.add(safe(2, 1))
    print("TELL: S_2_1 (survived visiting)")
    
    # Simulate: creaking but no rumbling
    solver.add(creaking_at(2, 1))
    solver.add(Not(rumbling_at(2, 1)))
    print("TELL: C_2_1 (creaking)")
    print("TELL: ¬R_2_1 (no rumbling)")
    
    adjacent_to_2_1 = [(1, 1), (3, 1), (2, 2)]
    print(f"\nAdjacent to (2,1): {adjacent_to_2_1}")
    print("Creaking means damaged floor at one of: (1,1), (3,1), (2,2)")
    print("We know (1,1) is safe, so damage at (3,1) or (2,2)")
    
    # Check what we can infer
    for x, y in [(3, 1), (2, 2), (1, 2)]:
        is_safe = z3_entails(solver, safe(x, y))
        is_damaged = z3_entails(solver, damaged(x, y))
        print(f"ASK: ({x},{y}) - Safe: {is_safe}, Damaged: {is_damaged}")
    
    # Step 3: Move to (2, 2) and observe
    print("\n" + "-" * 70)
    print("OBSERVATION AT (2, 2): After moving North")
    print("-" * 70)
    
    solver.add(safe(2, 2))
    print("TELL: S_2_2 (survived visiting)")
    
    # This tells us (2,2) is NOT damaged
    print("Since (2,2) is safe, and we know damage is at (3,1) or (2,2)...")
    
    is_3_1_damaged = z3_entails(solver, damaged(3, 1))
    print(f"ASK: Is (3,1) damaged? {is_3_1_damaged}")
    
    print("\n✓ Task 3 complete!")


# ============================================================================
# TASK 4: AGENT LOOP
# ============================================================================

class WarehouseKBAgent:
    """
    Knowledge-based agent for Hazardous Warehouse
    
    Uses Z3 to maintain knowledge about the environment and make
    safe decisions about where to move.
    """
    
    def __init__(self, width: int = 4, height: int = 4):
        """
        Initialize the KB agent
        
        Args:
            width: Width of warehouse grid
            height: Height of warehouse grid
        """
        self.width = width
        self.height = height
        
        # Build knowledge base with physics
        self.solver = build_warehouse_kb(width, height) if Z3_AVAILABLE else Solver()
        
        # Track what we know
        self.known_safe: Set[Tuple[int, int]] = set()
        self.known_dangerous: Set[Tuple[int, int]] = set()
        self.visited: Set[Tuple[int, int]] = set()
        
        # Track agent state
        self.current_pos: Optional[Tuple[int, int]] = None
        self.current_direction: Optional[Direction] = None
        self.has_package: bool = False
        
        # For planning
        self.package_found: bool = False
        self.package_location: Optional[Tuple[int, int]] = None
        
        # For debugging
        self.step_count = 0
    
    def reset(self):
        """Reset agent for new episode"""
        self.solver = build_warehouse_kb(self.width, self.height) if Z3_AVAILABLE else Solver()
        self.known_safe.clear()
        self.known_dangerous.clear()
        self.visited.clear()
        self.current_pos = None
        self.current_direction = None
        self.has_package = False
        self.package_found = False
        self.package_location = None
        self.step_count = 0
    
    def perceive(self, pos: Tuple[int, int], direction: Direction, percept: Percept, has_package: bool):
        """
        PERCEIVE: Process sensory input from environment
        
        Args:
            pos: Current position (x, y)
            direction: Current facing direction
            percept: Sensor readings
            has_package: Whether robot has package
        """
        self.current_pos = pos
        self.current_direction = direction
        self.has_package = has_package
        self.visited.add(pos)
        
        print(f"\n[Step {self.step_count}] PERCEIVE at {pos}, facing {direction.name}")
        print(f"  Percept: {percept}")
        
        # TELL the solver what we perceived
        self.tell(pos, percept)
        
        # Update package location if beacon detected
        if percept.beacon:
            self.package_found = True
            self.package_location = pos
            print(f"  ✓ Found package at {pos}!")
        
        # Update our knowledge about safe squares
        self.update_knowledge()
    
    def tell(self, pos: Tuple[int, int], percept: Percept):
        """
        TELL: Add percepts to knowledge base
        
        Args:
            pos: Position where percepts were observed
            percept: Percept readings
        """
        if not Z3_AVAILABLE:
            return
        
        x, y = pos
        
        # Tell solver about creaking
        if percept.creaking:
            self.solver.add(creaking_at(x, y))
            print(f"  TELL: C_{x}_{y} (creaking)")
        else:
            self.solver.add(Not(creaking_at(x, y)))
            print(f"  TELL: ¬C_{x}_{y} (no creaking)")
        
        # Tell solver about rumbling
        if percept.rumbling:
            self.solver.add(rumbling_at(x, y))
            print(f"  TELL: R_{x}_{y} (rumbling)")
        else:
            self.solver.add(Not(rumbling_at(x, y)))
            print(f"  TELL: ¬R_{x}_{y} (no rumbling)")
        
        # We visited this square and survived, so it's safe
        self.solver.add(safe(x, y))
        print(f"  TELL: S_{x}_{y} (survived visiting)")
    
    def ask(self, x: int, y: int) -> bool:
        """
        ASK: Query if a square is safe
        
        Args:
            x, y: Coordinates to check
        
        Returns:
            True if provably safe, False otherwise
        """
        if not Z3_AVAILABLE:
            return False
        
        return z3_entails(self.solver, safe(x, y))
    
    def update_knowledge(self):
        """
        Update sets of known safe/dangerous squares by querying KB
        """
        if not Z3_AVAILABLE:
            return
        
        for x in range(1, self.width + 1):
            for y in range(1, self.height + 1):
                pos = (x, y)
                
                # Skip if already categorized
                if pos in self.known_safe or pos in self.known_dangerous:
                    continue
                
                # ASK if safe
                if self.ask(x, y):
                    self.known_safe.add(pos)
                    print(f"  ASK: ({x}, {y}) is SAFE")
                # ASK if dangerous
                elif (z3_entails(self.solver, damaged(x, y)) or 
                      z3_entails(self.solver, forklift_at(x, y))):
                    self.known_dangerous.add(pos)
                    print(f"  ASK: ({x}, {y}) is DANGEROUS")
    
    def act(self, env: HazardousWarehouseEnv) -> Action:
        """
        ACT: Choose next action based on knowledge
        
        Strategy:
        1. If at package location and don't have package: GRAB
        2. If have package and at exit (1,1): EXIT
        3. Otherwise: Navigate toward goal using safe squares
        
        Args:
            env: The warehouse environment
        
        Returns:
            Action to take
        """
        self.step_count += 1
        
        # Check if at package
        if (self.package_found and 
            self.current_pos == self.package_location and 
            not self.has_package):
            print(f"  ACT: GRAB package at {self.current_pos}")
            return Action.GRAB
        
        # Check if at exit with package
        if self.has_package and self.current_pos == (1, 1):
            print(f"  ACT: EXIT at {self.current_pos}")
            return Action.EXIT
        
        # Otherwise, navigate toward goal
        if self.has_package or not self.package_found:
            goal = (1, 1)  # Go to exit or explore
        else:
            goal = self.package_location
        
        # Plan next move
        action = self.plan_move(goal)
        print(f"  ACT: {action.name}")
        return action
    
    def plan_move(self, goal: Optional[Tuple[int, int]]) -> Action:
        """
        Plan next move toward goal using BFS through safe squares
        
        Args:
            goal: Target position, or None to explore
        
        Returns:
            Action to take
        """
        # If we have a specific goal and path, follow it
        if goal:
            path = self.find_path(self.current_pos, goal)
            if path and len(path) > 1:
                next_pos = path[1]
                return self.get_action_to_move(self.current_pos, next_pos, self.current_direction)
        
        # Otherwise, explore to nearest unknown adjacent square
        return self.explore_action()
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find path using BFS through known safe squares
        
        Args:
            start: Starting position
            goal: Goal position
        
        Returns:
            List of positions from start to goal, or None
        """
        if start == goal:
            return [start]
        
        queue = deque([(start, [start])])
        visited_bfs = {start}
        
        while queue:
            pos, path = queue.popleft()
            
            # Check adjacent positions
            for next_pos in self.get_adjacent(pos):
                if next_pos == goal:
                    return path + [next_pos]
                
                if next_pos in self.known_safe and next_pos not in visited_bfs:
                    visited_bfs.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
        
        return None
    
    def explore_action(self) -> Action:
        """
        Choose exploration action
        
        Returns best action to reach a safe, unvisited square
        """
        # Find adjacent safe squares
        for next_pos in self.get_adjacent(self.current_pos):
            if next_pos in self.known_safe and next_pos not in self.visited:
                return self.get_action_to_move(self.current_pos, next_pos, self.current_direction)
        
        # No safe unvisited adjacent - try any safe adjacent
        for next_pos in self.get_adjacent(self.current_pos):
            if next_pos in self.known_safe:
                return self.get_action_to_move(self.current_pos, next_pos, self.current_direction)
        
        # No safe moves - turn to explore
        return Action.TURN_RIGHT
    
    def get_action_to_move(self, current: Tuple[int, int], target: Tuple[int, int], 
                           direction: Direction) -> Action:
        """
        Get action needed to move from current to target
        
        Returns FORWARD, TURN_LEFT, or TURN_RIGHT
        """
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        
        # Determine required direction
        if dx == 1 and dy == 0:
            required_dir = Direction.EAST
        elif dx == -1 and dy == 0:
            required_dir = Direction.WEST
        elif dx == 0 and dy == 1:
            required_dir = Direction.NORTH
        elif dx == 0 and dy == -1:
            required_dir = Direction.SOUTH
        else:
            return Action.TURN_RIGHT  # Can't move there directly
        
        # Turn to face required direction
        if direction == required_dir:
            return Action.FORWARD
        elif direction.turn_left() == required_dir:
            return Action.TURN_LEFT
        elif direction.turn_right() == required_dir:
            return Action.TURN_RIGHT
        else:  # Need 180 degree turn
            return Action.TURN_RIGHT
    
    def get_adjacent(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid adjacent positions"""
        return get_adjacent_positions(pos[0], pos[1], self.width, self.height)


# ============================================================================
# TASK 5: TESTING
# ============================================================================

def task5_testing():
    """
    Task 5: Test the agent on example layout
    """
    print("\n" + "=" * 70)
    print("TASK 5: TESTING AGENT")
    print("=" * 70)
    
    if not Z3_AVAILABLE:
        print("Z3 not available - skipping agent testing")
        return
    
    # Create environment
    env = HazardousWarehouseEnv(seed=42)
    percept = env.reset(seed=42)
    
    print("\nInitial environment (revealed):")
    print(env.render(reveal=True))
    print(f"\nInitial percept: {percept}")
    print(f"Robot at: {env.robot_position}, facing {env.robot_direction.name}")
    
    # Create agent
    agent = WarehouseKBAgent(width=4, height=4)
    
    # Initial percepts
    agent.perceive(env.robot_position, env.robot_direction, percept, env.has_package)
    
    # Run episode
    max_steps = 100
    
    for step in range(max_steps):
        if env._terminated:
            break
        
        # Agent chooses action
        action = agent.act(env)
        
        # Execute action
        percept, reward, done, info = env.step(action)
        
        # Agent perceives new state
        if not done:
            agent.perceive(env.robot_position, env.robot_direction, percept, env.has_package)
        
        # Show progress occasionally
        if step % 10 == 0 or done:
            print(f"\n--- Step {step} ---")
            print(env.render(reveal=False))
    
    # Results
    print("\n" + "=" * 70)
    print("EPISODE RESULTS")
    print("=" * 70)
    print(f"Steps: {env.steps}")
    print(f"Total reward: {env.total_reward:.1f}")
    print(f"Success: {env._success}")
    print(f"Alive: {env.is_alive}")
    print(f"Has package: {env.has_package}")
    
    print("\nFinal environment:")
    print(env.render(reveal=True))
    
    print("\n✓ Task 5 complete!")
    
    return env.steps, env.total_reward, env._success


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tasks"""
    
    print("\n" + "=" * 70)
    print("PROBLEM 3.4: BUILDING A KNOWLEDGE-BASED AGENT")
    print("=" * 70)
    
    if not Z3_AVAILABLE:
        print("\n⚠️  WARNING: Z3 solver not available!")
        print("Install with: pip install z3-solver")
        print("\nProceeding with mock implementation for demonstration...\n")
    
    # Task 1: Setup and exploration
    task1_z3_exploration()
    
    # Task 3: Manual reasoning
    task3_manual_reasoning()
    
    # Task 5: Testing
    task5_testing()
    
    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETE")
    print("=" * 70)
    print("\nSee README.md for:")
    print("- Task 3 observations")
    print("- Task 5 results")
    print("- Task 6 reflection")


if __name__ == "__main__":
    main()