# Previous was volatile
import numpy as np
import random
from copy import deepcopy
import sys
import os
from contextlib import contextmanager
import multiprocessing

# ASSUME custom_model.py and game.py are accessible
from game import Game
from custom_model import CUSTOM_AI_MODEL

# --- Critical Hyperparameters for 10-Feature Model ---
POP_SIZE = 500  # Changed: Must be large for 10 features
NUM_FEATURES = 10 
NUM_TRIALS = 1
NOISE_SD = 0.2  # Recommended strength for exploration
MUTATION_RATE = 0.2 # Recommended frequency for exploration
NUM_GENERATIONS = 30

# --- GA Functions ---

def initialize_population(pop_size, num_features):
    """Generates the initial population data (list of random weight arrays)."""
    # Population is a list of N=POP_SIZE genotype arrays
    population = [np.random.uniform(-1, 1, num_features) for _ in range(pop_size)]
    return population

@contextmanager
def suppress_output():
    """Temporarily redirects stdout to /dev/null to suppress prints."""
    original_stdout = sys.stdout
    with open(os.devnull, 'w') as f:
        sys.stdout = f
        try:
            yield
        finally:
            sys.stdout = original_stdout

def run_game_simulation(genotype):
    """
    Runs game simulation multiple times (NUM_TRIALS) and returns the average score.
    Uses CUSTOM_AI_MODEL and suppresses output.
    """
    scores_list = []
    
    for _ in range(NUM_TRIALS):
        # Instantiate the agent and game inside the loop for a fresh start
        agent = CUSTOM_AI_MODEL(genotype=genotype)
        game = Game(mode="student", agent=agent) 
        
        pieces_dropped, rows_cleared = game.run_no_visual()

        # The fitness score is rows cleared
        scores_list.append(rows_cleared if pieces_dropped > 0 else 0.0)
            
    # Return the average of all trials
    return np.mean(scores_list)

def evaluate_population(population):
    """Gathers the fitness scores in parallel using all available CPU cores."""
    with multiprocessing.Pool(None) as pool: 
        fitness_scores = pool.map(run_game_simulation, population)
    return fitness_scores

def crossover(gene_a, gene_b):
    """Single-point crossover for two genotype arrays."""
    split_point = random.randint(1, len(gene_a) - 1)
    child_gene = np.concatenate((gene_a[:split_point], gene_b[split_point:]))
    return child_gene

def mutate(genotype, noise_sd):
    """Applies Multiplicative Gaussian mutation to a genotype array."""
    # Multiplicative noise: centered at 1 (no change), spread by noise_sd
    # Noise_sd should be small (e.g., 0.1 or 0.2)
    noise = np.random.normal(1, noise_sd, size=len(genotype))
    
    # Apply noise (Multiplicative Mutation)
    mutated = genotype * noise
    
    # Clip to [-1, 1] to keep weights bounded
    mutated = np.clip(mutated, -1, 1)
    return mutated


def select_and_reproduce(population, scores, pop_size, mutation_rate, noise_sd):
    """Selects the best (25% elitism/selection) and reproduces using rank-based selection."""
    
    # 1. Combine data and sort for selection (descending)
    scored_population = list(zip(population, scores))
    scored_population.sort(key=lambda x: x[1], reverse=True)
    
    # Keep the top half as parents (50% Elitism/Selection)
    num_parents = pop_size // 2
    parents = [item[0] for item in scored_population[:num_parents]]
    
    # 50% Elitism: Copy the top half directly
    new_population = [p.copy() for p in parents] 

    # Rank-Based Weighting (Best parent gets the highest weight)
    # The list is sorted, so we use the rank inverse for weights.
    weights = np.array(range(len(parents), 0, -1))
    
    # Soften the weights to prevent one agent from dominating (Cube Root)
    weights = np.power(weights, 1/3) 

    # 2. Reproduce until population size is reached (Crossover/Mutation)
    slots_needed = pop_size - len(new_population)
    
    for _ in range(slots_needed):
        # Select two parents based on softened rank weights
        parent_a, parent_b = random.choices(parents, weights=weights, k=2)
        
        # Crossover (Default to 100% Crossover Rate, similar to your simple plan)
        child_genotype = crossover(parent_a, parent_b)
        
        # Mutation
        if random.random() < mutation_rate:
            # Pass noise_sd to the fixed mutate function
            child_genotype = mutate(child_genotype, noise_sd) 
        
        new_population.append(child_genotype)
        
    return new_population

# --- Training Execution ---

print(f"Starting GA Training for {NUM_GENERATIONS} generations...")
print(f"Population size: {POP_SIZE}, Trials per Agent: {NUM_TRIALS}")

current_population = initialize_population(POP_SIZE, NUM_FEATURES)
history = {'max_fitness': [], 'avg_fitness': []}

for gen in range(NUM_GENERATIONS):
    print(f"\n--- Generation {gen + 1} ---")
        
    # 1. DATA GATHERING
    scores = evaluate_population(current_population)
        
    # Data Exploration (Summary)
    max_score = np.max(scores)
    avg_score = np.mean(scores)
    
    history['max_fitness'].append(max_score)
    history['avg_fitness'].append(avg_score)
        
    print(f"Max Score: {max_score:.2f}, Avg Score: {avg_score:.2f}")

    # 2. TRAINING (Evolution)
    if gen < NUM_GENERATIONS - 1:
        current_population = select_and_reproduce(
            current_population, scores, POP_SIZE, MUTATION_RATE, NOISE_SD
        )
            
# Find the best overall genotype from the final population
final_scores = evaluate_population(current_population)
best_genotype = current_population[np.argmax(final_scores)]

print("\n--- Training Complete ---")
print(f"Final Best Genotype: {best_genotype}")
