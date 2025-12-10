import random
import math
from typing import List, Dict, Tuple, Generator
import copy
from tsp_utils import calculate_distance_km


class TravelingSalesmanGA:
    def __init__(self, cities: Dict[int, Tuple[float, float]]):
        """
        Initialize GA solver for TSP.
        
        Args:
            cities: Dictionary mapping city ID to (latitude, longitude) coordinates
        """
        self.cities = cities
        self.city_ids = list(cities.keys())
        self.num_cities = len(cities)
    
    def calculate_distance(self, city1: int, city2: int) -> float:
        """Calculate distance between two cities in kilometers using Haversine formula."""
        return calculate_distance_km(self.cities[city1], self.cities[city2])
    
    def calculate_tour_length(self, tour: List[int]) -> float:
        """Calculate total length of a tour."""
        if len(tour) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(tour)):
            current = tour[i]
            next_city = tour[(i + 1) % len(tour)]  # Wrap around to start
            total += self.calculate_distance(current, next_city)
        return total
    
    def create_random_tour(self) -> List[int]:
        """Create a random valid tour (permutation of all cities)."""
        tour = self.city_ids.copy()
        random.shuffle(tour)
        return tour
    
    def create_initial_population(self, population_size: int) -> List[List[int]]:
        """Create initial population of random tours."""
        return [self.create_random_tour() for _ in range(population_size)]
    
    def tournament_selection(self, population: List[List[int]], fitness_scores: List[float], 
                            tournament_size: int = 3) -> List[int]:
        """Select a parent using tournament selection."""
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        tournament.sort(key=lambda x: x[1])  # Sort by fitness (lower is better)
        return tournament[0][0].copy()
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order Crossover (OX) for TSP."""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        # Fill remaining positions from parent2
        parent2_remaining = [city for city in parent2 if city not in child[start:end]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = parent2_remaining[idx]
                idx += 1
        
        return child
    
    def partially_mapped_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Partially Mapped Crossover (PMX) for TSP."""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        # Create mapping
        mapping = {}
        for i in range(start, end):
            if parent2[i] not in child[start:end]:
                mapping[parent2[i]] = parent1[i]
        
        # Fill remaining positions
        for i in range(size):
            if child[i] is None:
                city = parent2[i]
                while city in mapping:
                    city = mapping[city]
                child[i] = city
        
        return child
    
    def crossover(self, parent1: List[int], parent2: List[int], crossover_type: str = "ox") -> List[int]:
        """Perform crossover between two parents."""
        if crossover_type == "ox":
            return self.order_crossover(parent1, parent2)
        else:
            return self.partially_mapped_crossover(parent1, parent2)
    
    def swap_mutation(self, tour: List[int], mutation_rate: float) -> List[int]:
        """Swap mutation: swap two random cities."""
        if random.random() < mutation_rate:
            tour = tour.copy()
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour
    
    def inversion_mutation(self, tour: List[int], mutation_rate: float) -> List[int]:
        """Inversion mutation: reverse a random segment."""
        if random.random() < mutation_rate:
            tour = tour.copy()
            start, end = sorted(random.sample(range(len(tour)), 2))
            tour[start:end+1] = reversed(tour[start:end+1])
        return tour
    
    def mutate(self, tour: List[int], mutation_rate: float, mutation_type: str = "swap") -> List[int]:
        """Apply mutation to a tour."""
        if mutation_type == "swap":
            return self.swap_mutation(tour, mutation_rate)
        else:
            return self.inversion_mutation(tour, mutation_rate)
    
    def _genetic_algorithm_core(self, population_size: int, generations: int, 
                                crossover_rate: float, mutation_rate: float,
                                elitism_count: int, tournament_size: int,
                                crossover_type: str, mutation_type: str):
        """Core GA algorithm without progress yielding."""
        # Initialize population
        population = self.create_initial_population(population_size)
        
        best_tour = None
        best_fitness = float('inf')
        fitness_history = []
        stop_reason = "max_generations"
        
        try:
            for generation in range(generations):
                # Calculate fitness (tour length - lower is better)
                fitness_scores = [self.calculate_tour_length(tour) for tour in population]
                
                # Track best solution
                current_best_idx = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
                current_best_fitness = fitness_scores[current_best_idx]
                
                if current_best_fitness < best_fitness:
                    best_fitness = current_best_fitness
                    best_tour = population[current_best_idx].copy()
                
                fitness_history.append(best_fitness)
                
                # Create new population
                new_population = []
                
                # Elitism: keep best individuals
                sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1])
                for i in range(elitism_count):
                    new_population.append(sorted_pop[i][0].copy())
                
                # Generate offspring
                while len(new_population) < population_size:
                    # Selection
                    parent1 = self.tournament_selection(population, fitness_scores, tournament_size)
                    parent2 = self.tournament_selection(population, fitness_scores, tournament_size)
                    
                    # Crossover
                    if random.random() < crossover_rate:
                        child = self.crossover(parent1, parent2, crossover_type)
                    else:
                        child = parent1.copy()
                    
                    # Mutation
                    child = self.mutate(child, mutation_rate, mutation_type)
                    
                    new_population.append(child)
                
                population = new_population
                
        except Exception as e:
            import traceback
            print(f"Error in _genetic_algorithm_core: {e}")
            print(traceback.format_exc())
            stop_reason = "error"
        
        # Ensure we have a valid tour
        if best_tour is None:
            if len(population) > 0:
                best_tour = population[0].copy()
                best_fitness = self.calculate_tour_length(best_tour)
            else:
                # Fallback: create a random tour
                best_tour = self.create_random_tour()
                best_fitness = self.calculate_tour_length(best_tour)
        
        # Ensure we have fitness history
        if len(fitness_history) == 0:
            fitness_history = [best_fitness]
        
        self._stop_reason = stop_reason
        return best_tour, best_fitness, fitness_history, stop_reason
    
    def genetic_algorithm(self, population_size: int, generations: int,
                         crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                         elitism_count: int = 2, tournament_size: int = 3,
                         crossover_type: str = "ox", mutation_type: str = "swap",
                         yield_progress: bool = False):
        """
        Run Genetic Algorithm for TSP.
        
        Args:
            population_size: Size of the population
            generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of best individuals to keep
            tournament_size: Size of tournament for selection
            crossover_type: "ox" (Order Crossover) or "pmx" (Partially Mapped Crossover)
            mutation_type: "swap" or "inversion"
            yield_progress: Whether to yield progress updates
        
        Returns:
            If yield_progress=False: (best_tour, best_fitness, fitness_history, stop_reason)
            If yield_progress=True: Generator yielding (current_best_tour, generation, fitness)
        """
        if not yield_progress:
            try:
                return self._genetic_algorithm_core(
                    population_size, generations, crossover_rate, mutation_rate,
                    elitism_count, tournament_size, crossover_type, mutation_type
                )
            except Exception as e:
                import traceback
                print(f"Error in genetic_algorithm: {e}")
                print(traceback.format_exc())
                # Return default values on error
                default_tour = self.create_random_tour()
                default_fitness = self.calculate_tour_length(default_tour)
                return default_tour, default_fitness, [default_fitness], "error"
        
        # Initialize population
        population = self.create_initial_population(population_size)
        
        best_tour = None
        best_fitness = float('inf')
        fitness_history = []
        stop_reason = "max_generations"
        
        for generation in range(generations):
            # Calculate fitness
            fitness_scores = [self.calculate_tour_length(tour) for tour in population]
            
            # Track best solution
            current_best_idx = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_tour = population[current_best_idx].copy()
            
            fitness_history.append(best_fitness)
            
            # Yield progress
            yield (best_tour.copy(), generation + 1, best_fitness)
            
            # Create new population
            new_population = []
            
            # Elitism
            sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1])
            for i in range(elitism_count):
                new_population.append(sorted_pop[i][0].copy())
            
            # Generate offspring
            while len(new_population) < population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores, tournament_size)
                parent2 = self.tournament_selection(population, fitness_scores, tournament_size)
                
                # Crossover
                if random.random() < crossover_rate:
                    child = self.crossover(parent1, parent2, crossover_type)
                else:
                    child = parent1.copy()
                
                # Mutation
                child = self.mutate(child, mutation_rate, mutation_type)
                
                new_population.append(child)
            
            population = new_population
        
        self._stop_reason = stop_reason
        self._final_result = (best_tour, best_fitness, fitness_history, stop_reason)

