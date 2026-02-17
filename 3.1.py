import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PropositionalKB:
    """Knowledge Base for Propositional Logic"""
    
    def __init__(self, grid_size=(3, 3)):
        self.grid_size = grid_size
        self.clauses = []  # CNF clauses
        self.facts = set()  # Known facts
        
    def add_clause(self, clause):
        """Add a clause in CNF format (list of literals)"""
        self.clauses.append(clause)
    
    def add_fact(self, fact):
        """Add a definite fact"""
        self.facts.add(fact)
        self.clauses.append([fact])
    
    def get_adjacent(self, i, j):
        """Get adjacent squares (4-connectivity)"""
        adjacent = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 1 <= ni <= self.grid_size[0] and 1 <= nj <= self.grid_size[1]:
                adjacent.append((ni, nj))
        
        return adjacent


class HazardousWarehouse:
    """Hazardous Warehouse Environment with Propositional Logic"""
    
    def __init__(self, grid_size=(3, 3)):
        self.grid_size = grid_size
        self.kb = PropositionalKB(grid_size)
        self.safe_squares = set()
        self.dangerous_squares = set()
        self.uncertain_squares = set()
        
        # Initialize all squares as uncertain
        for i in range(1, grid_size[0] + 1):
            for j in range(1, grid_size[1] + 1):
                self.uncertain_squares.add((i, j))
        
        self.encode_physics()
        self.encode_initial_knowledge()
    
    def encode_physics(self):
        """Encode the physics of the environment"""
        print("=" * 60)
        print("PART 1: ENCODING THE PHYSICS")
        print("=" * 60)
        
        # For each square, encode creaking and rumbling rules
        for i in range(1, self.grid_size[0] + 1):
            for j in range(1, self.grid_size[1] + 1):
                adjacent = self.kb.get_adjacent(i, j)
                
                print(f"\nSquare ({i},{j}):")
                print(f"  Adjacent squares: {adjacent}")
                
                # Creaking rule: C_i,j ↔ (D_adj1 ∨ D_adj2 ∨ ...)
                damage_locs = [f"D_{ai}_{aj}" for ai, aj in adjacent]
                print(f"  C_{i}_{j} ↔ ({' ∨ '.join(damage_locs)})")
                
                # Rumbling rule: R_i,j ↔ (F_adj1 ∨ F_adj2 ∨ ...)
                forklift_locs = [f"F_{ai}_{aj}" for ai, aj in adjacent]
                print(f"  R_{i}_{j} ↔ ({' ∨ '.join(forklift_locs)})")
                
                # Safe rule: S_i,j ↔ (¬D_i,j ∧ ¬F_i,j)
                print(f"  S_{i}_{j} ↔ (¬D_{i}_{j} ∧ ¬F_{i}_{j})")
    
    def encode_initial_knowledge(self):
        """Encode initial knowledge"""
        print("\n" + "=" * 60)
        print("PART 2: INITIAL KNOWLEDGE")
        print("=" * 60)
        
        # Starting square (1,1) is safe
        print("\n1. Starting square (1,1) is safe:")
        print("   S_1_1")
        print("   Which implies: ¬D_1_1 ∧ ¬F_1_1")
        self.kb.add_fact("S_1_1")
        self.kb.add_fact("¬D_1_1")
        self.kb.add_fact("¬F_1_1")
        self.safe_squares.add((1, 1))
        self.uncertain_squares.discard((1, 1))
        
        # At least one damaged floor
        damage_squares = [f"D_{i}_{j}" for i in range(1, self.grid_size[0] + 1) 
                         for j in range(1, self.grid_size[1] + 1)]
        print("\n2. At least one square has damaged floor:")
        print(f"   {' ∨ '.join(damage_squares)}")
        
        # At least one forklift
        forklift_squares = [f"F_{i}_{j}" for i in range(1, self.grid_size[0] + 1) 
                           for j in range(1, self.grid_size[1] + 1)]
        print("\n3. At least one square has forklift:")
        print(f"   {' ∨ '.join(forklift_squares)}")
    
    def observe_at_location(self, i, j, creaking, rumbling):
        """Process observations at a location"""
        print(f"\n{'=' * 60}")
        print(f"OBSERVATION AT ({i},{j})")
        print(f"{'=' * 60}")
        print(f"Creaking: {creaking}")
        print(f"Rumbling: {rumbling}")
        
        adjacent = self.kb.get_adjacent(i, j)
        print(f"Adjacent squares: {adjacent}")
        
        inferences = []
        
        # Process creaking observation
        if not creaking:
            print(f"\n¬C_{i}_{j} observed")
            print(f"From: C_{i}_{j} ↔ (D_{adjacent[0][0]}_{adjacent[0][1]} ∨ ...)")
            print(f"Infer: ¬C_{i}_{j} → ¬(D_{adjacent[0][0]}_{adjacent[0][1]} ∨ ...)")
            print("By De Morgan's Law: ", end="")
            
            damage_facts = []
            for ai, aj in adjacent:
                fact = f"¬D_{ai}_{aj}"
                self.kb.add_fact(fact)
                damage_facts.append(fact)
                inferences.append((ai, aj, "no_damage"))
            
            print(" ∧ ".join(damage_facts))
        else:
            print(f"\nC_{i}_{j} observed")
            damage_options = [f"D_{ai}_{aj}" for ai, aj in adjacent]
            # Filter out known false ones
            known_no_damage = [f"¬D_{ai}_{aj}" in self.kb.facts for ai, aj in adjacent]
            remaining = [damage_options[idx] for idx, known in enumerate(known_no_damage) if not known]
            
            if remaining:
                print(f"From: C_{i}_{j} → ({' ∨ '.join(damage_options)})")
                known_false = [damage_options[idx] for idx, known in enumerate(known_no_damage) if known]
                if known_false:
                    print(f"Known false: {', '.join(known_false)}")
                print(f"Therefore: {' ∨ '.join(remaining)}")
                inferences.append((i, j, "damage_disjunction", remaining))
        
        # Process rumbling observation
        if not rumbling:
            print(f"\n¬R_{i}_{j} observed")
            print(f"From: R_{i}_{j} ↔ (F_{adjacent[0][0]}_{adjacent[0][1]} ∨ ...)")
            print(f"Infer: ¬R_{i}_{j} → ¬(F_{adjacent[0][0]}_{adjacent[0][1]} ∨ ...)")
            print("By De Morgan's Law: ", end="")
            
            forklift_facts = []
            for ai, aj in adjacent:
                fact = f"¬F_{ai}_{aj}"
                self.kb.add_fact(fact)
                forklift_facts.append(fact)
                inferences.append((ai, aj, "no_forklift"))
            
            print(" ∧ ".join(forklift_facts))
        else:
            print(f"\nR_{i}_{j} observed")
            forklift_options = [f"F_{ai}_{aj}" for ai, aj in adjacent]
            print(f"From: R_{i}_{j} → ({' ∨ '.join(forklift_options)})")
            inferences.append((i, j, "forklift_disjunction", forklift_options))
        
        # Update safe squares
        self.update_safety_status(adjacent, inferences)
        
        return inferences
    
    def update_safety_status(self, adjacent, inferences):
        """Update which squares are safe based on inferences"""
        print("\n" + "-" * 60)
        print("SAFETY STATUS UPDATE")
        print("-" * 60)
        
        for ai, aj in adjacent:
            no_damage = f"¬D_{ai}_{aj}" in self.kb.facts
            no_forklift = f"¬F_{ai}_{aj}" in self.kb.facts
            
            if no_damage and no_forklift:
                if (ai, aj) not in self.safe_squares:
                    print(f"Square ({ai},{aj}): No damage AND no forklift → SAFE")
                    self.safe_squares.add((ai, aj))
                    self.uncertain_squares.discard((ai, aj))
                    self.dangerous_squares.discard((ai, aj))
                    self.kb.add_fact(f"S_{ai}_{aj}")
    
    def print_status(self):
        """Print current status of all squares"""
        print("\n" + "=" * 60)
        print("CURRENT KNOWLEDGE STATE")
        print("=" * 60)
        
        print("\nSafe squares:", sorted(self.safe_squares))
        print("Dangerous squares:", sorted(self.dangerous_squares))
        print("Uncertain squares:", sorted(self.uncertain_squares))
        
        print("\nKnown facts in KB:")
        for fact in sorted(self.kb.facts):
            print(f"  {fact}")
    
    def visualize_grid(self, title="Warehouse Grid", robot_pos=None):
        """Visualize the warehouse grid"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Draw grid
        for i in range(1, self.grid_size[0] + 1):
            for j in range(1, self.grid_size[1] + 1):
                x, y = j - 1, self.grid_size[0] - i
                
                # Determine color
                if (i, j) in self.safe_squares:
                    color = 'lightgreen'
                    label = 'S'
                elif (i, j) in self.dangerous_squares:
                    color = 'lightcoral'
                    label = 'D'
                elif (i, j) in self.uncertain_squares:
                    color = 'lightyellow'
                    label = '?'
                else:
                    color = 'white'
                    label = ''
                
                # Draw rectangle
                rect = patches.Rectangle((x, y), 1, 1, linewidth=2, 
                                        edgecolor='black', facecolor=color)
                ax.add_patch(rect)
                
                # Add label
                ax.text(x + 0.5, y + 0.7, f'({i},{j})', 
                       ha='center', va='center', fontsize=10, weight='bold')
                ax.text(x + 0.5, y + 0.3, label, 
                       ha='center', va='center', fontsize=16, weight='bold')
                
                # Add robot marker
                if robot_pos and robot_pos == (i, j):
                    ax.text(x + 0.5, y + 0.5, '🤖', 
                           ha='center', va='center', fontsize=20)
        
        ax.set_xlim(0, self.grid_size[1])
        ax.set_ylim(0, self.grid_size[0])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, weight='bold')
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='lightgreen', edgecolor='black', label='Safe (S)'),
            patches.Patch(facecolor='lightcoral', edgecolor='black', label='Dangerous (D)'),
            patches.Patch(facecolor='lightyellow', edgecolor='black', label='Uncertain (?)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        return fig


def main():
    """Main function to run the homework solution"""
    
    # Create warehouse environment
    warehouse = HazardousWarehouse(grid_size=(3, 3))
    
    # Part 3: Observation at (1,1)
    print("\n" + "=" * 60)
    print("PART 3: SCENARIO REASONING AT (1,1)")
    print("=" * 60)
    
    warehouse.observe_at_location(1, 1, creaking=False, rumbling=False)
    warehouse.print_status()
    
    # Visualize after first observation
    fig1 = warehouse.visualize_grid(
        title="After Observation at (1,1): No Creaking, No Rumbling",
        robot_pos=(1, 1)
    )
    plt.savefig('warehouse_after_1_1.png', dpi=150, bbox_inches='tight')
    print("\n[Saved visualization: warehouse_after_1_1.png]")
    
    # Part 4: Observation at (2,1)
    print("\n" + "=" * 60)
    print("PART 4: EXPLORATION AT (2,1)")
    print("=" * 60)
    
    warehouse.observe_at_location(2, 1, creaking=True, rumbling=False)
    warehouse.print_status()
    
    # Visualize after second observation
    fig2 = warehouse.visualize_grid(
        title="After Observation at (2,1): Creaking, No Rumbling",
        robot_pos=(2, 1)
    )
    plt.savefig('warehouse_after_2_1.png', dpi=150, bbox_inches='tight')
    print("\n[Saved visualization: warehouse_after_2_1.png]")
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print("\nAfter observations at (1,1) and (2,1):")
    print(f"  Safe squares: {sorted(warehouse.safe_squares)}")
    print(f"  Uncertain squares: {sorted(warehouse.uncertain_squares)}")
    print("\nKey constraint from (2,1) observation:")
    print("  D_3_1 ∨ D_2_2 (at least one has damaged floor)")
    print("  ¬F_3_1 ∧ ¬F_2_2 (neither has forklift)")
    
    plt.show()


if __name__ == "__main__":
    main()

    