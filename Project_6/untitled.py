"""
Simple training script for the Tetris AI

Usage:
    python train.py              # Train 1 generation (quick test)
    python train.py 10           # Train 10 generations
    python train.py 20 30 10     # Train 20 gens, 30 agents, 10 games each
"""

import sys
from custom_model import train_multiple_generations, train_one_generation

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Training one generation (quick test)...")
        train_one_generation(population_size=20, num_games=5)
    
    # elif len(sys.argv) == 2:
    #     generations = int(sys.argv[1])
    #     print(f"Training {generations} generations...")
    #     train_multiple_generations()
    
    # elif len(sys.argv) == 4:
    #     generations = int(sys.argv[1])
    #     population = int(sys.argv[2])
    #     games = int(sys.argv[3])
    #     print(f"Training {generations} generations with {population} agents, {games} games each...")
    #     train_multiple_generations(
    #         generations=generations,
    #         population_size=population,
    #         num_games=games
    #     )
    
