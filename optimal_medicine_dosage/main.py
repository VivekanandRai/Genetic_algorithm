"""
main.py
Entry point: runs GA, saves results, and prints final summary.
"""

import os
import json
import numpy as np
import random
from ga import run_ga
from fitness_functions import effectiveness, side_effects
from visualize import plot_fitness_curve

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

def save_best_result(best_individual, out_json_path):
    """Save best dosage info as JSON."""
    eff = float(effectiveness(best_individual))
    se = float(side_effects(best_individual))
    fitness = eff - se

    data = {
        "best_dosage_mg": {
            "dose_A": round(float(best_individual[0]), 2),
            "dose_B": round(float(best_individual[1]), 2),
            "dose_C": round(float(best_individual[2]), 2)
        },
        "effectiveness": round(eff, 2),
        "side_effects": round(se, 2),
        "fitness": round(fitness, 2)
    }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(data, f, indent=4)
    return data

def main():
    params = {
        "pop_size": 50,
        "generations": 150,
        "elite_frac": 0.1,
        "tournament_k": 3,
        "crossover_rate": 0.9,
        "mutation_std": 2.0,
        "rng_seed": GLOBAL_SEED
    }

    print("Running Genetic Algorithm with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    best_ind, best_fit, fitness_hist, _ = run_ga(**params)

    result_dir = "results"
    json_path = os.path.join(result_dir, "best_dosage.json")
    png_path = os.path.join(result_dir, "fitness_curve.png")

    result = save_best_result(best_ind, json_path)
    plot_fitness_curve(fitness_hist, png_path)

    print(f"\nGeneration {params['generations']} | Best Fitness: {result['fitness']:.2f}")
    print("Best Dosages:")
    print(f"  Dose A: {result['best_dosage_mg']['dose_A']} mg")
    print(f"  Dose B: {result['best_dosage_mg']['dose_B']} mg")
    print(f"  Dose C: {result['best_dosage_mg']['dose_C']} mg")
    print(f"Effectiveness: {result['effectiveness']} | Side Effects: {result['side_effects']}")
    print(f"Results saved to {json_path}")
    print(f"Fitness curve plotted as {png_path}")

if __name__ == "__main__":
    main()
