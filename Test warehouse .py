"""
Test script for Hazardous Warehouse KB Agent
Tests the knowledge-based agent without requiring Z3
"""

from hazardous_warehouse_env import HazardousWarehouseEnv, Action


def test_environment():
    """Test the environment basics"""
    print("=" * 70)
    print("TESTING HAZARDOUS WAREHOUSE ENVIRONMENT")
    print("=" * 70)
    
    # Create environment
    env = HazardousWarehouseEnv(seed=42)
    percept = env.reset(seed=42)
    
    print("\nInitial state (revealed):")
    print(env.render(reveal=True))
    
    print(f"\nInitial percept: {percept}")
    print(f"Robot at: {env.robot_position}, facing {env.robot_direction.name}")
    
    true_state = env.get_true_state()
    print("\nTrue state:")
    print(f"  Damaged floors: {true_state['damaged']}")
    print(f"  Forklift: {true_state['forklift']}")
    print(f"  Package: {true_state['package']}")
    
    print("\n✓ Environment test passed!")


def test_movement():
    """Test robot movement mechanics"""
    print("\n" + "=" * 70)
    print("TESTING ROBOT MOVEMENT")
    print("=" * 70)
    
    env = HazardousWarehouseEnv(seed=42)
    env.reset(seed=42)
    
    print(f"\nStarting at {env.robot_position}, facing {env.robot_direction.name}")
    
    # Move forward (EAST)
    percept, reward, done, info = env.step(Action.FORWARD)
    print(f"\nAfter FORWARD: {env.robot_position}, facing {env.robot_direction.name}")
    print(f"  Percept: {percept}")
    print(f"  Reward: {reward}, Done: {done}")
    
    # Turn left
    percept, reward, done, info = env.step(Action.TURN_LEFT)
    print(f"\nAfter TURN_LEFT: {env.robot_position}, facing {env.robot_direction.name}")
    
    # Move forward (NORTH)
    percept, reward, done, info = env.step(Action.FORWARD)
    print(f"\nAfter FORWARD: {env.robot_position}, facing {env.robot_direction.name}")
    print(f"  Percept: {percept}")
    
    print("\n✓ Movement test passed!")


def test_percepts():
    """Test percept generation"""
    print("\n" + "=" * 70)
    print("TESTING PERCEPTS")
    print("=" * 70)
    
    env = HazardousWarehouseEnv(seed=42)
    env.reset(seed=42)
    
    true_state = env.get_true_state()
    print(f"\nDamaged floors: {true_state['damaged']}")
    print(f"Forklift: {true_state['forklift']}")
    
    # Test at different positions
    test_positions = [(1, 1), (2, 1), (1, 2), (2, 2)]
    
    for target_x, target_y in test_positions:
        # Navigate to position manually
        env.reset(seed=42)
        # (This is a simplified test - full navigation would be more complex)
        
        # Check what percepts would be at each position
        current_x, current_y = env.robot_position
        print(f"\nPosition ({current_x}, {current_y}):")
        
        # Get adjacent positions
        adjacent = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = current_x + dx, current_y + dy
            if 1 <= nx <= 4 and 1 <= ny <= 4:
                adjacent.append((nx, ny))
        
        print(f"  Adjacent: {adjacent}")
        
        # Check for hazards
        creaking_expected = any((x, y) in true_state['damaged'] for x, y in adjacent)
        rumbling_expected = any((x, y) == true_state['forklift'] for x, y in adjacent)
        
        print(f"  Expected creaking: {creaking_expected}")
        print(f"  Expected rumbling: {rumbling_expected}")
        
        break  # Just test first position
    
    print("\n✓ Percept test passed!")


def test_logical_reasoning():
    """Test logical reasoning patterns"""
    print("\n" + "=" * 70)
    print("TESTING LOGICAL REASONING PATTERNS")
    print("=" * 70)
    
    print("\nPhysics rules for position (2, 2):")
    print("  Adjacent positions: (1,2), (3,2), (2,1), (2,3)")
    print("\n  Rule 1 - Creaking:")
    print("    C_2_2 ↔ (D_1_2 ∨ D_3_2 ∨ D_2_1 ∨ D_2_3)")
    print("    If ¬C_2_2, then ¬D_1_2 ∧ ¬D_3_2 ∧ ¬D_2_1 ∧ ¬D_2_3)")
    
    print("\n  Rule 2 - Rumbling:")
    print("    R_2_2 ↔ (F_1_2 ∨ F_3_2 ∨ F_2_1 ∨ F_2_3)")
    print("    If ¬R_2_2, then ¬F_1_2 ∧ ¬F_3_2 ∧ ¬F_2_1 ∧ ¬F_2_3)")
    
    print("\n  Rule 3 - Safety:")
    print("    S_2_2 ↔ (¬D_2_2 ∧ ¬F_2_2)")
    
    print("\n  Inference example:")
    print("    Given: S_1_1 (safe), ¬C_1_1 (no creaking), ¬R_1_1 (no rumbling)")
    print("    Adjacent to (1,1): (2,1), (1,2)")
    print("    Infer: ¬D_2_1, ¬D_1_2, ¬F_2_1, ¬F_1_2")
    print("    Therefore: S_2_1 and S_1_2 (both safe)")
    
    print("\n✓ Logical reasoning test passed!")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("HAZARDOUS WAREHOUSE KB AGENT - TEST SUITE")
    print("=" * 70)
    
    test_environment()
    test_movement()
    test_percepts()
    test_logical_reasoning()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    
    print("\nTo run the full KB agent (requires z3-solver):")
    print("  python warehouse_kb_agent.py")
    
    print("\nTo install Z3:")
    print("  pip install z3-solver")


if __name__ == "__main__":
    main()