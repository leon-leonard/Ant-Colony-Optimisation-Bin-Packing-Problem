import numpy as np
import pandas as pd

# Function to create initial pheromone matrix, putting random amounts of pheromone on edges
def initialise_pheromone_matrix(num_items, num_bins):
    pheromone_matrix = np.random.rand(num_items, num_bins)

    return pheromone_matrix

# Function for ant to follow path in graph
def generate_ant_path(num_items, num_bins, pheromone_matrix, item_weights):
    # Create a list of empty bins to hold items
    bins = [[] for _ in range(num_bins)]

    # For each item, calculate the probability of placing it in each bin
    for i in range(num_items):
        pheromone_sum = sum(pheromone_matrix[i])
        probabilities = [pheromone_matrix[i][j] / pheromone_sum for j in range(num_bins)]
        cumulative_probabilities = [sum(probabilities[:j+1]) for j in range(len(probabilities))]
        random_probability = np.random.rand()
        # Select the bin whose cumulative probabilty is the first to exceed the random probability
        selected_bin = [prob for prob in cumulative_probabilities if prob >= random_probability][0]
        selected_bin_index = cumulative_probabilities.index(selected_bin)
        # Add the item to the selected bin
        bins[selected_bin_index].append(item_weights[i])

    return bins

# Function to calculate the total weight of items in each bin
def calculate_bin_weights(bins):
    bin_weights = [sum(items) for items in bins]

    return bin_weights

# Function to calculate the fitness score for a ant path solution
def calculate_fitness_score(bin_weights):
    fitness = max(bin_weights) - min(bin_weights)

    return fitness

# Function to evaporate the pheromone matrix
def evaporate_pheromone_matrix(pheromone_matrix, evaporation_rate):
    evaporated = pheromone_matrix * evaporation_rate

    return evaporated

# Function to update the pheromone matrix based on the fitness of the ant path solution
def update_pheromone_matrix(pheromone_matrix, bins, item_weights, fitness):
    # Look through each bin and increase the pheromone on the edge between the item and the bin
    for i in range(len(bins)):
        for item in bins[i]:
            item_index = np.where(item_weights == item)[0][0]
            pheromone_matrix[item_index][i] += 100 / fitness

    return pheromone_matrix

# Function to run the ant colony optimisation algorithm for each trial
def run_aco(num_items, bpp, num_ants, evaporation_rate, num_evaluations=10000):
    # Set parameters based on the BPP problem
    if bpp.lower() == 'bpp1':
        num_bins = 10
        item_weights = np.arange(1, 501)
    elif bpp.lower() == 'bpp2':
        num_bins = 50
        item_weights = (np.arange(1, 501) ** 2) / 2
    
    # Initialise pheromone matrix
    pheromones = initialise_pheromone_matrix(num_items, num_bins)
    evaluation = 0

    # Initialise variables to track the best solution
    best_fitness = np.inf
    best_solution = None
    best_evaluation = None

    while evaluation < num_evaluations:
        ant_solutions = []
        ant_fitnesses = []

        # For each ant, generate a path and calculate the fitness
        for ant in range(num_ants):
            bins = generate_ant_path(num_items, num_bins, pheromones, item_weights)
            bin_weights = calculate_bin_weights(bins)
            fitness = calculate_fitness_score(bin_weights)
            ant_solutions.append(bins)
            ant_fitnesses.append(fitness)

            # Update best solution if the current fitness is better
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = bins
                best_evaluation = evaluation

            evaluation += 1

        # Update pheromone matrix for each ant's solution
        for bins, fitness in zip(ant_solutions, ant_fitnesses):
            pheromones = update_pheromone_matrix(pheromones, bins, item_weights, fitness)

        # Apply pheromone evaporation after each ant's solution has updated the pheromone matrix
        pheromones = evaporate_pheromone_matrix(pheromones, evaporation_rate)

    return best_fitness, best_solution, best_evaluation

# Function to run multiple trials of the ACO algorithm 
def run_experiment(num_items, bpp, num_ants, evaporation_rate, bpp_df, num_trials=1):
    best_fitnesses = []
    for trial in range(num_trials):
        # Set random seed for reproducibility, so trials within the same experiment are random, but
        # trials between different experiments are comparable
        np.random.seed(trial)
        best_fitness, best_solution, best_iteration = run_aco(num_items, bpp, num_ants, evaporation_rate)
        best_fitnesses.append(best_fitness)

        trial_row = pd.DataFrame([[bpp, num_ants, evaporation_rate, trial, best_fitness]],
                                  columns=['bpp', 'p', 'e', 'trial', 'best_fitness'])
        bpp_df = pd.concat([bpp_df, trial_row], axis=0, ignore_index=True)
        print(f'Trial {trial+1}, Best Fitness: {best_fitness}')

    return best_fitnesses, bpp_df

def main():
    bpp_df = pd.DataFrame(columns=['bpp', 'p', 'e', 'trial', 'best_fitness'])

    # Run experiments for BPP1 with different ant (p) and evaporation (e) values
    print("Running experiments for BPP1 (b = 10)")
    print("p = 10, e = 0.60")
    best_fitnesses, bpp_df = run_experiment(500, 'bpp1', 10, 0.6, bpp_df)

    print("p = 10, e = 0.90")
    best_fitnesses, bpp_df = run_experiment(500, 'bpp1', 10, 0.9, bpp_df)

    print("p = 100, e = 0.60")
    best_fitnesses, bpp_df = run_experiment(500, 'bpp1', 100, 0.6, bpp_df)

    print("p = 100, e = 0.90")
    best_fitnesses, bpp_df = run_experiment(500, 'bpp1', 100, 0.9, bpp_df)
    
    # Run experiments for BPP2 with different configurations
    print("Running experiments for BPP2 (b = 50)")
    print("p = 10, e = 0.60")
    best_fitnesses, bpp_df = run_experiment(500, 'bpp2', 10, 0.6, bpp_df)

    print("p = 10, e = 0.90")
    best_fitnesses, bpp_df = run_experiment(500, 'bpp2', 10, 0.9, bpp_df)

    print("p = 100, e = 0.60")
    best_fitnesses, bpp_df = run_experiment(500, 'bpp2', 100, 0.6, bpp_df)

    print("p = 100, e = 0.90")
    best_fitnesses, bpp_df = run_experiment(500, 'bpp2', 100, 0.9, bpp_df)

    bpp_df.to_csv('bpp_df.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main()