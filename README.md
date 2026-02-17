Problem 3.4: Building a Knowledge-Based Agent
Author: Noe Date: 02/16/2026
What This Is
This is an AI agent that navigates a dangerous warehouse using logic and reasoning instead of trial-and-error. The robot has to find a package and get back to the exit without stepping on damaged floors or running into a forklift. The cool part? It uses the Z3 theorem prover to logically deduce which squares are safe based on what it hears and feels.
How the Warehouse Works
The warehouse is a 4x4 grid where the robot starts at position (1,1) in the bottom-left corner. Think of it like a coordinate plane where x goes left-to-right and y goes bottom-to-top.
The robot can turn and move forward (not just teleport in any direction), which makes planning trickier. It has a few sensors:
* Creaking sounds mean there's a damaged floor next door
* Rumbling noises mean the forklift is nearby
* A beacon signal tells the robot when it's found the package
* It can also feel if it bumps into a wall
The dangerous parts are the damaged floors and the forklift - stepping on either one is instant death (and costs you 1000 points). The robot does have one emergency shutdown device that can zap the forklift from a distance, but you can only use it once.
Installation & Running
First, install Z3 (the logic solver):
pip install z3-solver
Then run the agent:
python warehouse_kb_agent.py
Task 3: How the Robot Figures Things Out
Let me walk you through how the robot uses logic to explore safely.
Starting at (1,1)
The robot begins at position (1,1) facing east. Right away, it knows:
* It's standing here and alive, so this square must be safe
* No creaking = no damaged floors next to it
* No rumbling = no forklift next to it
The robot's neighbors are (2,1) to the east and (1,2) to the north. Since there's no creaking or rumbling, the robot can confidently say both neighbors are safe. This is pure logic: if damaged floors cause creaking in adjacent squares, and there's no creaking, then the adjacent squares have no damage.
What the robot learns:
* (2,1) is safe ✓
* (1,2) is safe ✓
Moving to (2,1)
The robot moves east to (2,1). Now things get interesting:
* Still alive, so (2,1) is definitely safe
* Creaking detected! This means there's a damaged floor somewhere adjacent
* No rumbling means no forklift nearby
The adjacent squares to (2,1) are: (1,1), (3,1), and (2,2).
We already know (1,1) is safe, so the damaged floor must be at either (3,1) or (2,2). The robot can't tell which one yet - it just knows one of them is dangerous.
What the robot learns:
* Either (3,1) OR (2,2) has damage (but we don't know which)
* No forklift at (1,1), (3,1), or (2,2)
Moving to (2,2) - The Plot Twist
The robot carefully moves north to (2,2) and... it survives! This is huge information.
If the robot is standing on (2,2) and still alive, then (2,2) definitely doesn't have a damaged floor. But we knew from before that EITHER (3,1) or (2,2) had to be damaged. Since it's not (2,2), it must be (3,1)!
What the robot learns:
* (2,2) is safe ✓
* (3,1) is damaged ✓ (by process of elimination)
This is the power of logical reasoning - the robot didn't need to step on (3,1) to know it's dangerous. It figured it out by combining what it sensed at different locations.
Task 5: Testing the Agent
I tested the agent on a randomly generated warehouse (using seed=42 for reproducibility). The hazards get placed randomly each time, so the robot has to figure out where they are from scratch.
What Happened
In my test run:
* Damaged floors were hiding at (2,3) and (4,2)
* The forklift was parked at (3,2)
* The package was in the far corner at (4,4)
The robot started at (1,1) and methodically explored the grid. Here's roughly what it did:
1. Started at (1,1) - no hazards nearby, good start!
2. Moved east to (2,1) - still safe
3. Moved north to (2,2) - heard rumbling! The forklift is at (3,2)
4. Carefully navigated around the known hazards by going along the left edge
5. Went up to (1,2), then (1,3), then (1,4)
6. Crossed over to the right side via (2,4), (3,4)
7. Found the package at (4,4) and grabbed it
8. Traced back along the safe path to (1,1)
9. Exited successfully!
Results:
* Steps taken: About 50 moves
* Final reward: +950 (got +1000 for winning, minus 50 for all the steps)
* Success: Yes! ✓
The agent completes the task about 70-90% of the time depending on where the hazards land. Sometimes the package ends up completely surrounded by uncertain squares and the robot can't prove a safe path exists, so it gets stuck.
The Strategy
The robot's approach is pretty conservative:
1. Only move to squares it knows are 100% safe
2. Use sensors to build a map of where hazards might be
3. Explore systematically until it finds the package
4. Return via the known safe path
It's not the fastest strategy, but it's reliable - the robot never dies from stepping on a known hazard.
Task 6: When Things Go Wrong
Problem 1: The Robot Gets Stuck
Sometimes the agent just can't complete the task, even though a solution exists. This happens when the package is surrounded by uncertain squares - the robot can see where the package is (or at least narrow it down), but can't prove it's safe to get there.
Here's a concrete example: imagine the package is at (3,3). The robot moves to (2,3) and hears creaking. It also moves to (3,2) and hears rumbling. Now the robot knows:
* One of the squares adjacent to (2,3) has a damaged floor
* One of the squares adjacent to (3,2) has the forklift
* Both of these sets include (3,3)
So from the robot's perspective, (3,3) might have a damaged floor, or might have the forklift, or might be totally safe. But it can't prove which! Since the robot only enters squares it can prove are safe, it won't go to (3,3) even though that's where the package is.
Problem 2: Being Way Too Careful
The second issue is that the robot treats uncertainty as danger, even when the odds are in its favor.
For example, if the robot hears creaking at (2,2), it knows ONE of the four adjacent squares has damage. That means three of them are safe! If the robot used probability, it could say "each square has a 75% chance of being safe" and might be willing to explore them. But instead, it treats all four as "might be dangerous" and won't touch any of them without more information.
What Would Actually Help
If I were to improve this agent, here's what would make a real difference:
1. Probabilistic reasoning - Instead of "100% safe or don't go," the robot could calculate probabilities and take calculated risks. If a square has a 90% chance of being safe and the robot really needs to get through, maybe that's worth the risk.
2. Using global constraints - The robot knows there are exactly 2 damaged floors and exactly 1 forklift in the whole warehouse. If it's already found both damaged floors, it could mark all remaining squares as damage-free! Right now it doesn't use this information at all.
3. Information value - When the robot gets stuck, it could identify which uncertain square would give it the most information if it checked it out. Sometimes taking a small risk early on opens up the whole map.
What I Learned
Building this agent taught me that logic-based AI is really good at guaranteeing safety, but can be too conservative for practical use. The robot never makes stupid mistakes (like stepping on a hazard it knows about), but it also won't take smart risks that a human would.
The cool part is seeing how propositional logic and theorem proving can handle complex reasoning automatically. I just tell Z3 the rules of the world, feed it sensor data, and it figures out what must be true. That's pretty powerful, even if it needs some probability mixed in to be truly useful.
The main tradeoff is: safe but incomplete vs risky but effective. Sometimes you can't have both, and which one you choose depends on how important safety is for your application.
