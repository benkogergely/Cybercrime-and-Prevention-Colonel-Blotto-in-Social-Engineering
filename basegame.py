import itertools
import numpy as np
import matplotlib.pyplot as plt

# Settings
defended_soldiers = 15
attacking_soldiers = 16
battlefield_limits = [4, 3, 6, 4, 3]

# Function: generate all bounded integer allocations
def bounded_integer_compositions(total, bounds):
    if not bounds:
        return []
    if len(bounds) == 1:
        if 0 <= total <= bounds[0]:
            return [[total]]
        else:
            return []
    compositions = []
    for i in range(min(total, bounds[0]) + 1):
        for tail in bounded_integer_compositions(total - i, bounds[1:]):
            compositions.append([i] + tail)
    return compositions

# Generate all possible allocations for defender and attacker
defender_allocations = bounded_integer_compositions(defended_soldiers, battlefield_limits)
attacker_allocations = bounded_integer_compositions(attacking_soldiers, battlefield_limits)

# Compute win probabilities for defender
defender_win_probs = np.zeros(len(defender_allocations))
for idx, d_alloc in enumerate(defender_allocations):
    wins = 0
    for a_alloc in attacker_allocations:
        won_battles = sum(1 for d, a in zip(d_alloc, a_alloc) if d > a)
        if won_battles > len(battlefield_limits)//2:
            wins += 1
    defender_win_probs[idx] = wins / len(attacker_allocations)

# Select top 10 strongest strategies
top_indices = np.argsort(defender_win_probs)[-10:][::-1]
top_heatmap_data = np.array([defender_allocations[i] for i in top_indices])

# Print top 10 strategies
print("Top 10 strongest defender strategies (Soldier allocation, Win probability):")
for i in top_indices:
    print(f"{defender_allocations[i]} -> {defender_win_probs[i]:.2f}")

# Heatmap visualization of top strategies
plt.figure(figsize=(10,6))
plt.imshow(top_heatmap_data, cmap='Blues', aspect='auto', vmin=0, vmax=np.max(top_heatmap_data))  # Darker blue for more soldiers
plt.colorbar(label='Number of Soldiers')

x_labels = [
    "Tech Support\nScam",
    "Grandparent/Relative\nImpersonation",
    "Government\nImpersonation",
    "Investment\nScam",
    "Confidence/Romance\nFraud"
]

plt.xticks(range(len(battlefield_limits)), x_labels)

plt.yticks(range(len(top_indices)), [f'Index {i} ({defender_win_probs[i]:.2f})' for i in top_indices])
plt.xlabel('Battlefields')
plt.ylabel('Top Defender Strategy Index and Win Probability')
plt.title('Top 10 Defender Strategies')
plt.show()
