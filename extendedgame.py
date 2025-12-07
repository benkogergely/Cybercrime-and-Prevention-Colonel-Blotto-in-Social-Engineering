import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Parameters
# -------------------------------------------------
V = 17   # defenders
T = 16   # attackers
capacities = np.array([4, 4, 4, 4, 4, 1, 2, 1])
values = np.array([2, 3, 4, 2, 3, 4, 4, 4])
num_battles = len(capacities)

NUM_ACTIONS = 5000
ITERATIONS = 5000

# -------------------------------------------------
# Helper function: random valid Blotto allocation
# -------------------------------------------------
def random_allocation(total, caps):
    remaining = total
    alloc = np.zeros_like(caps)
    for i in range(len(caps)-1):
        m = min(caps[i], remaining)
        alloc[i] = np.random.randint(0, m + 1)
        remaining -= alloc[i]
    alloc[-1] = min(remaining, caps[-1])
    return alloc

def payoff(att, dfn, values):
    return np.sum(values * (att > dfn))


# -------------------------------------------------
# 1. Sampled strategy sets
# -------------------------------------------------
att_strategies = np.array([random_allocation(T, capacities) for _ in range(NUM_ACTIONS)])
def_strategies = np.array([random_allocation(V, capacities) for _ in range(NUM_ACTIONS)])

# payoff matrix A[a][d]
A = np.zeros((NUM_ACTIONS, NUM_ACTIONS))
for i in range(NUM_ACTIONS):
    for j in range(NUM_ACTIONS):
        A[i, j] = payoff(att_strategies[i], def_strategies[j], values)

# -------------------------------------------------
# 2. Initialization
# -------------------------------------------------
att_regret = np.zeros(NUM_ACTIONS)
def_regret = np.zeros(NUM_ACTIONS)

att_strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS
def_strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS

att_payoff_history = []

# -------------------------------------------------
# 3. FULL Regret-Matching Algorithm (CFR-like)
# -------------------------------------------------
for it in range(ITERATIONS):

    # Expected payoffs under current mixed strategies
    # attacker’s expected payoff for each pure action
    att_exp_payoffs = A @ def_strategy
    def_exp_payoffs = (-A.T) @ att_strategy

    # Actual expected payoff this round
    actual_att = att_strategy @ A @ def_strategy
    att_payoff_history.append(actual_att)

    # Regret update (full information)
    att_regret += att_exp_payoffs - actual_att
    def_regret += def_exp_payoffs - (-actual_att)

    # Update mixed strategies
    att_pos = np.maximum(att_regret, 0)
    def_pos = np.maximum(def_regret, 0)

    if att_pos.sum() > 1e-12:
        att_strategy = att_pos / att_pos.sum()
    else:
        att_strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS

    if def_pos.sum() > 1e-12:
        def_strategy = def_pos / def_pos.sum()
    else:
        def_strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS


# -------------------------------------------------
# Final mixed strategies → expected allocations
# -------------------------------------------------
att_avg_alloc = np.sum(att_strategies.T * att_strategy, axis=1)
def_avg_alloc = np.sum(def_strategies.T * def_strategy, axis=1)

# -------------------------------------------------
# Visualization 1: Learning Curve
# -------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(att_payoff_history, color="black", alpha=0.3, label="Instant expected payoff")
plt.plot(np.convolve(att_payoff_history, np.ones(200)/200, mode='valid'),
         color="red", linewidth=2, label="Moving average (200)")
plt.xlabel("Iteration")
plt.ylabel("Attacker expected payoff")
plt.title("High-Precision Regret-Matching (5000 iterations)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# -------------------------------------------------
# Visualization 2: Final Mixed Strategy Allocations
# -------------------------------------------------
b = np.arange(1, num_battles + 1)

plt.figure(figsize=(10, 6))
plt.bar(b - 0.15, att_avg_alloc, width=0.3, color="red",
        label="Attacker avg allocation")
plt.bar(b + 0.15, def_avg_alloc, width=0.3, color="navy",
        label="Defender avg allocation")
plt.xticks(b)
plt.xlabel("Battlefield")
plt.ylabel("Expected troops")
plt.title("Approximate Nash Equilibrium: Average Allocations in Colonel Blotto")
plt.legend()
plt.show()
