"""
visualize.py
Handles plotting and saving fitness curve to results/ directory.
"""

import matplotlib.pyplot as plt
import os

def plot_fitness_curve(fitness_history, out_path):
    """Plot fitness trend and save PNG."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm - Fitness over Generations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
