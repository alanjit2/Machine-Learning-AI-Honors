from copy import copy, deepcopy
import random
# from file_operation import *

"""
helper functions for file operations
"""
import json, os
from datetime import datetime

def get_best_model_weights():
    best_model_filepath = 'best_model.json'
    if os.path.exists(best_model_filepath):
        try:
            with open(best_model_filepath, 'r') as f:
                data = json.load(f)
                return data['weights']
        except:
            return None
    return None

def read_best_model():
    filepath = '/home/jupyter-259941/Machine-Learning-AI-Honors/Project_6/best_model.json'
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            print("Doesn't Exist")
            return None
    return None

def save_best_model(weights, fitness, generation, rows_cleared):
    filepath = 'best_model.json'
    current_best_model = read_best_model()

    is_best = False
    if current_best_model is None:
        is_best = True
        print(f" BEST SCORE - Rows: {rows_cleared:,} (First model saved)")
    elif rows_cleared > current_best_model.get('rows_cleared', 0):
        is_best = True
        old_best = current_best_model.get('rows_cleared', 0)
        improvement = rows_cleared - old_best
        print(f"\n BEST SCORE ")
        print(f"Current best score: {rows_cleared:,} rows (Gen {generation})")
    else:
        # old_best = current_best_model.get('rows_cleared', 0)
        print(f" Not saved current not better than previous")

    backup_filepath = f'model_gen_{generation}.json'
    backup_data = {
        'weights': weights,
        'fitness': fitness,
        'generation': generation,
        'rows_cleared': rows_cleared,
        'timestamp': datetime.now().isoformat()
    }
    with open(backup_filepath, 'w') as f:
        json.dump(backup_data, f, indent=2)
    print(f" Saved generation backup: {backup_filepath} (fitness: {fitness:.2f}, rows: {rows_cleared:,})")

    if is_best:
        with open(filepath, 'w') as f:
            json.dump(backup_data, f, indent=2)
        print(f" Updated best_model.json")


def load_training_history():
    filepath = 'training_history.json'
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return {'generations': [], 'best_fitness_per_gen': [], 'avg_fitness_per_gen': []}
    return {'generations': [], 'best_fitness_per_gen': [], 'avg_fitness_per_gen': []}


def save_training_history(history):
    training_filepath = 'training_history.json'
    with open(training_filepath, 'w') as f:
        json.dump(history, f, indent=2)

    if 'generation_details' in history:
        history_filepath = 'training_history_detailed.json'
        with open(history_filepath, 'w') as f:
            json.dump(history, f, indent=2)


"""
The best model is saved to: best_model.json and training history is saved to: training_history.json
Initialize Default weights for the model to random values
"""


class CUSTOM_AI_MODEL:
    DEFAULT_WEIGHTS = {
        'aggregate_height': random.random(),
        'lines_cleared': random.random(),
        'holes': random.random(),
        'bumpiness': random.random(),
        'max_height': random.random(),
        'wells': random.random(),
        'column_transitions': random.random(),
        'row_transitions': random.random(),
        'pit_depth': random.random(),
        'blocks_above_holes': random.random()
    }

    SEPARATOR_WIDTH = 70

    def __init__(self, weights=None):
        if weights is not None:
            self.weights = weights
        else:
            best_model_data = read_best_model()
            if best_model_data:
                self.weights = best_model_data['weights']
                self.generation = best_model_data.get('generation', 0)
                self.rows_cleared = best_model_data.get('rows_cleared', 0)
                self._print_model_info()

            else:
                self.weights = self.DEFAULT_WEIGHTS.copy()
                self.generation = 0
                self.rows_cleared = 0
                print("Using default weights (no saved model found)")

        self.fitness_scores = []
        self.avg_fitness = 0

    def _print_model_info(self):
        """Print information about the loaded model."""
        separator = '=' * self.SEPARATOR_WIDTH
        print(f"\n{separator}")
        print("LOADED BEST MODEL")
        print(f"{separator}")
        print(f"Generation: {self.generation}")
        print(f"Best Performance: {self.rows_cleared:,} rows cleared")
        print(f"Weights:")
        for key, value in self.weights.items():
            print(f"  {key:25s}: {value:8.4f}")
        print(f"{separator}\n")

    def get_best_move(self, board, tetriminos_piece, depth=1):
        optimal_position = -1000
        optimal_rotation = None
        highest_score = -1000


        # current_tetriminos_piece = tetriminos_piece
        rotated_piece = tetriminos_piece
        rotation_count = 0

        # for i in range(4):
        while rotation_count < 4:
            rotated_piece = rotated_piece.get_next_rotation()
            rotation_count += 1
            for column in range(board.width):
                try:
                    landing_row = board.drop_height(rotated_piece, column)
                except:
                    continue

                move_evaluation = self.evaluate_score(board, rotated_piece, column, landing_row)
                if move_evaluation > highest_score:
                    highest_score = move_evaluation
                    optimal_position = column
                    optimal_rotation = rotated_piece

        return optimal_position, optimal_rotation


    def extract_features(self, board):
        features = {}
        heights = self.get_column_heights(board)
        features['aggregate_height'] = sum(heights)
        features['lines_cleared'] = self.count_complete_lines(board)
        features['blocks_above_holes'] = self.count_blocks_above_holes(board, heights)

        features['bumpiness'] = self.evaluate_bumpiness(heights)
        features['wells'] = self.compute_well_score(heights)
        features['column_transitions'] = self.calculate_column_transitions(board, heights)
        features['row_transitions'] = self.compute_row_transitions(board)
        features['pit_depth'] = self.compute_max_pit_depth(heights)

        features['max_height'] = (max(heights) if heights else 0) - self.count_complete_lines(board)
        features['holes'] = self.count_holes(board, heights) * self.count_blocks_above_holes(board, heights)

        return features

    def get_column_heights(self, board):
        return [
            max([row_idx + 1 for row_idx in range(len(board)) if board[row_idx][col_idx]], default=0)
            for col_idx in range(len(board[0]))
        ]

    def count_complete_lines(self, board):
        return sum(1 for row in board if all(row))

    def count_holes(self, board, heights):
        total_holes = 0
        board_width = len(board[0])
        board_height = len(board)

        for col_idx in range(board_width):
            has_ceiling = False
            for row_idx in reversed(range(board_height)):
                cell_filled = board[row_idx][col_idx]
                if cell_filled:
                    has_ceiling = True
                elif has_ceiling:
                    total_holes += 1

        return total_holes

    def evaluate_bumpiness(self, heights):
        total_bumpiness = 0
        for idx in range(len(heights) - 1):
            height_difference = heights[idx] - heights[idx + 1]
            total_bumpiness = height_difference ** 2
        return total_bumpiness


    def evaluate_score(self, board, tetriminos_piece, x, y):
        board_copy = deepcopy(board.board)

        for offset in tetriminos_piece.body:
            try:
                row_idx = y + offset[1]
                col_idx = x + offset[0]
                board_copy[row_idx][col_idx] = True
            except:
                return float('-inf')

        extracted_features = self.extract_features(board_copy)
        evaluation_score = sum(weight * extracted_features[key]
                               for key, weight in self.weights.items())
        return evaluation_score


    def calculate_column_transitions(self, board, heights):
        change_count = 0
        num_columns = len(board[0])
        num_rows = len(board)

        for col_idx in range(num_columns):
            for row_idx in range(num_rows - 1):
                current_cell = board[row_idx][col_idx]
                next_cell = board[row_idx + 1][col_idx]
                if current_cell != next_cell:
                    change_count += 1

        return change_count

    def compute_row_transitions(self, board):
        change_count = 0
        num_rows = len(board)
        num_columns = len(board[0])

        for row_idx in range(num_rows):
            for col_idx in range(num_columns - 1):
                current_cell = board[row_idx][col_idx]
                adjacent_cell = board[row_idx][col_idx + 1]
                if current_cell != adjacent_cell:
                    change_count += 1

        return change_count

    def compute_well_score(self, heights):
        well_score = 0
        num_columns = len(heights)

        for col_idx in range(num_columns):
            # Left edge column
            if col_idx == 0:
                if num_columns > 1:
                    well_score += max(0, heights[1] - heights[0])
            # Right edge column
            elif col_idx == num_columns - 1:
                well_score += max(0, heights[-2] - heights[-1])
            # Middle columns
            else:
                depth_from_left = heights[col_idx - 1] - heights[col_idx]
                depth_from_right = heights[col_idx + 1] - heights[col_idx]

                if depth_from_left > 0 and depth_from_right > 0:
                    well_score += min(depth_from_left, depth_from_right)

        return well_score


    def compute_max_pit_depth(self, heights):
        if not heights:
            return 0

        maximum_pit_depth = 0
        num_columns = len(heights)

        for col_idx in range(num_columns):
            current_pit_depth = 0

            # Left edge
            if col_idx == 0:
                if num_columns > 1:
                    current_pit_depth = heights[1] - heights[0]
            # Right edge
            elif col_idx == num_columns - 1:
                current_pit_depth = heights[-2] - heights[-1]
            # Middle columns
            else:
                depth_from_left = heights[col_idx - 1] - heights[col_idx]
                depth_from_right = heights[col_idx + 1] - heights[col_idx]

                if depth_from_left > 0 and depth_from_right > 0:
                    current_pit_depth = min(depth_from_left, depth_from_right)

            maximum_pit_depth = max(maximum_pit_depth, current_pit_depth)

        return maximum_pit_depth


    def count_blocks_above_holes(self, board, heights):
        blocks_covering_holes = 0
        num_cols = len(board[0])

        for col_idx in range(num_cols):
            found_cavity = False

            for row_idx in range(len(board) - 1, -1, -1):
                cell_filled = board[row_idx][col_idx]
                is_in_stack = row_idx < heights[col_idx] - 1

                if not cell_filled and is_in_stack:
                    found_cavity = True
                elif found_cavity and cell_filled:
                    blocks_covering_holes += 1

        return blocks_covering_holes


def create_random_weights():
    return {
        'aggregate_height': random.uniform(-1.0, 0.0),
        'lines_cleared': random.uniform(0.0, 1.0),
        'holes': random.uniform(-1.0, 0.0),
        'bumpiness': random.uniform(-1.0, 0.0),
        'max_height': random.uniform(-1.0, 0.0),
        'wells': random.uniform(-1.0, 0.0),
        'column_transitions': random.uniform(-1.0, 0.0),
        'row_transitions': random.uniform(-1.0, 0.0),
        'pit_depth': random.uniform(-1.0, 0.0),
        'blocks_above_holes': random.uniform(-1.0, 0.0)
    }


def mutate_weights(weights, mutation_rate=0.1, mutation_scale=0.15, mutation_type='gaussian'):
    mutated = weights.copy()

    for key in mutated:
        if random.random() < mutation_rate:
            if mutation_type == 'gaussian':
                noise = random.gauss(0, mutation_scale)
                mutated[key] = mutated[key] * (1 + noise)
            elif mutation_type == 'uniform':
                noise = random.uniform(-mutation_scale, mutation_scale)
                mutated[key] = mutated[key] + noise
            elif mutation_type == 'adaptive':
                noise = random.gauss(0, mutation_scale * abs(mutated[key]))
                mutated[key] = mutated[key] + noise

            if key == 'lines_cleared':
                mutated[key] = max(0.0, min(2.0, mutated[key]))
            else:
                mutated[key] = max(-2.0, min(0.5, mutated[key]))

    return mutated


def crossover_weights(parent1_weights, parent2_weights):
    child_weights = {}
    for key in parent1_weights:
        if random.random() < 0.5:
            child_weights[key] = parent1_weights[key]
        else:
            child_weights[key] = parent2_weights[key]
    return child_weights


def evaluate_agent(agent, num_of_games=5, verbose=False, generation=None):
    from game import Game

    rows_cleared = 0
    total_tetriminos = 0
    best_rows = 0

    for game_num in range(num_of_games):
        game = Game("student", agent=agent, generation=generation, game_num=game_num + 1)
        pieces_dropped, rows_cleared = game.run_no_visual()

        rows_cleared += rows_cleared
        total_tetriminos += pieces_dropped
        best_rows = max(best_rows, rows_cleared)

        if verbose:
            print(f"Stats of the game {game_num + 1}/{num_of_games}: rows cleared : {rows_cleared} , pieces dropped : {pieces_dropped} ")

    avg_rows = rows_cleared / num_of_games
    avg_tetriminos = total_tetriminos / num_of_games

    # fitness = avg_rows + (avg_pieces * 0.1)
    fitness = avg_rows / (avg_tetriminos)

    return fitness, best_rows, avg_rows


def execute_generation(population_size=20, num_games=5, elite_count=4):
    print("\n" + "=" * 70)
    print("STARTING NEW GENERATION")
    print("=" * 70)

    history = load_training_history()
    current_gen = len(history['generations']) + 1

    if 'generation_details' not in history:
        history['generation_details'] = []
    if 'all_time_best_rows' not in history:
        history['all_time_best_rows'] = 0

    population = []

    best_model_weights = get_best_model_weights()

    if best_model_weights and current_gen > 1:
        print(f"Generation {current_gen}: Continuing from saved best model")

        population.append(CUSTOM_AI_MODEL(best_model_weights))

        for i in range(elite_count - 1):
            mutated = mutate_weights(best_model_weights, mutation_rate=0.1, mutation_scale=0.1, mutation_type='gaussian')
            population.append(CUSTOM_AI_MODEL(mutated))

        exploit_count = (population_size - elite_count) // 2
        for i in range(exploit_count):
            if i % 3 == 0: #  Most exploitive agents - small changes
                mutated = mutate_weights(best_model_weights, mutation_rate=0.25, mutation_scale=0.25, mutation_type='gaussian')
            elif i % 3 == 1: #  Combination of exploration and exploitive - adjust the small changes
                mutated = mutate_weights(best_model_weights, mutation_rate=0.3, mutation_scale=0.35, mutation_type='adaptive')
            else: #  Constant exploration
                mutated = mutate_weights(best_model_weights, mutation_rate=0.35, mutation_scale=0.30, mutation_type='uniform')
            population.append(CUSTOM_AI_MODEL(mutated))

        explore_count = population_size - elite_count - exploit_count
        for i in range(explore_count):
            if i % 2 == 0:
                mutated = create_random_weights()
            else:
                mutated = mutate_weights(best_model_weights, mutation_rate=0.5, mutation_scale=0.5, mutation_type='gaussian')
            population.append(CUSTOM_AI_MODEL(mutated))
    else:
        print(f"Running {current_gen} Gen : Creating initial random population")
        for i in range(population_size):
            if i == 0 and best_model_weights:
                population.append(CUSTOM_AI_MODEL(best_model_weights))
            else:
                population.append(CUSTOM_AI_MODEL(create_random_weights()))

    # print(f"\nEvaluating {population_size} agents ({num_games} games each)...")
    fitness_scores = []

    for idx, agent in enumerate(population):
        print(f"\nAgent {idx + 1}/{population_size}:")
        fitness, best_rows, avg_rows = evaluate_agent(agent, num_of_games=num_games, verbose=True, generation=current_gen)
        fitness_scores.append((fitness, best_rows, avg_rows, agent))
        print(f"Fitness: {fitness:.2f}, Best rows: {best_rows}, Avg rows: {avg_rows:.1f}")

    fitness_scores.sort(reverse=True, key=lambda x: x[0])

    fitness, best_rows, avg_rows, best_agent = fitness_scores[0]
    avg_fitness = sum(f[0] for f in fitness_scores) / len(fitness_scores)
    median_fitness = fitness_scores[len(fitness_scores) // 2][0]

    print("\n" + "=" * 70)
    print(f"Completed gen : {current_gen} ")
    # print("=" * 70)
    print(f"Best Agent  - Fitness: {fitness:.2f}, Best Game: {best_rows:,} rows")
    print(f"Avg Fitness - {avg_fitness:.2f}")
    print(f"Median Fitness - {median_fitness:.2f}")
    print(f"Top 5 Agents: {[f'{f[0]:.1f}' for f in fitness_scores[:5]]}")
    print(f"Top 5 Rows: {[f'{f[1]:,}' for f in fitness_scores[:5]]}")

    if best_rows > history['all_time_best_rows']:
        history['all_time_best_rows'] = best_rows

    save_best_model(best_agent.weights, fitness, current_gen, best_rows)

    history['generations'].append(current_gen)
    history['fitness'].append(fitness)
    history['avg_fitness'].append(avg_fitness)

    gen_detail = {
        'generation': current_gen,
        'fitness': fitness,
        'best_rows': best_rows,
        'avg_fitness': avg_fitness,
        'median_fitness': median_fitness,
        'top_5_fitness': [f[0] for f in fitness_scores[:5]],
        'top_5_rows': [f[1] for f in fitness_scores[:5]],
        'timestamp': datetime.now().isoformat()
    }
    history['generation_details'].append(gen_detail)

    save_training_history(history)

    return best_agent, fitness



if __name__ == "__main__":
    print("Training one generation...")
    execute_generation(population_size=20, num_games=5)