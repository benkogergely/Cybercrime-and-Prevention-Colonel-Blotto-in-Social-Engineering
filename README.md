# Cybercrime and Prevention: Colonel-Blotto in Social-Engineering
## basegame.py

This script analyzes optimal soldier allocations in a simplified scenario across multiple battlefields.

### Key Steps

1. **Settings:**  
   - Number of defending and attacking soldiers (`defended_soldiers`, `attacking_soldiers`).  
   - Limits of soldiers per battlefield (`battlefield_limits`).

2. **Generate allocations:**  
   - Computes all possible ways to distribute soldiers across battlefields without exceeding the limits using a recursive function (`bounded_integer_compositions`).

3. **Compute win probabilities:**  
   - For each defender allocation, calculates the probability of winning a majority of battlefields against all possible attacker allocations.

4. **Identify top strategies:**  
   - Selects the 10 defender allocations with the highest win probabilities.

5. **Output & visualization:**  
   - Prints the top 10 strategies with their win probabilities.  
   - Displays a heatmap showing the soldier distribution in these top strategies (darker blue = more soldiers).

### Purpose

The script helps determine the strongest defensive strategies for distributing soldiers across multiple battlefields to maximize the chance of winning.

## extendedgame.py

This script simulates a Colonel Blotto game and approximates a Nash equilibrium using a regret-matching algorithm.

### Key Steps

1. **Parameters:**  
   - Number of attacking and defending soldiers (`T`, `V`).  
   - Battlefield capacities (`capacities`) and values (`values`).  
   - Number of sampled actions (`NUM_ACTIONS`) and iterations (`ITERATIONS`).

2. **Random strategy sampling:**  
   - Generates `NUM_ACTIONS` random valid soldier allocations for both attacker and defender.

3. **Payoff calculation:**  
   - Computes the payoff matrix `A` where each entry represents the attacker’s gain over the defender in a given pair of strategies.

4. **Regret-matching algorithm:**  
   - Iteratively updates mixed strategies based on accumulated regrets (CFR-like).  
   - Keeps track of expected payoffs and adjusts strategies to approximate a Nash equilibrium.

5. **Compute expected allocations:**  
   - Calculates the expected average soldier distribution for attacker and defender under the learned mixed strategies.

6. **Visualization:**  
   - **Learning curve:** Plots the attacker’s expected payoff over iterations (with moving average).  
   - **Final strategy allocations:** Bar chart showing average troop allocations across battlefields for both sides.

### Purpose

The script approximates optimal strategies in a Colonel Blotto game, showing how attackers and defenders should distribute troops to maximize expected outcomes.


