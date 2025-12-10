import random
import math
from typing import List, Dict, Tuple, Generator
from tsp_utils import calculate_distance_km


class TravelingSalesmanSA:
    def __init__(self, cities: Dict[int, Tuple[float, float]]):
        """
        Initialize SA solver for TSP.
        
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
    
    def initial_config(self) -> List[int]:
        """Create initial random tour."""
        tour = self.city_ids.copy()
        random.shuffle(tour)
        return tour
    
    def calc_temp(self, iteration: int, initial_temp: float, cooling_rate: float = 0.95) -> float:
        """Calculate temperature at given iteration."""
        return initial_temp * (cooling_rate ** iteration)
    
    def swap_neighbors(self, tour: List[int]) -> List[int]:
        """Swap two random cities."""
        new_tour = tour.copy()
        i, j = random.sample(range(len(tour)), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour
    
    def reverse_segment(self, tour: List[int]) -> List[int]:
        """Reverse a random segment of the tour."""
        new_tour = tour.copy()
        start, end = sorted(random.sample(range(len(tour)), 2))
        new_tour[start:end+1] = reversed(new_tour[start:end+1])
        return new_tour
    
    def random_successor(self, current_tour: List[int]) -> List[int]:
        """Generate a random neighbor tour."""
        if random.random() < 0.5:
            return self.swap_neighbors(current_tour)
        else:
            return self.reverse_segment(current_tour)
    
    def _sim_anneal_core(self, iter_max: int, T: float, min_temp: float = 2.0, cooling_rate: float = 0.95):
        """Core SA algorithm without progress yielding."""
        xcurr = self.initial_config()
        xbest = xcurr.copy()
        
        temperature_history = []
        fitness_history = []
        stop_reason = "max_iterations"
        
        try:
            for i in range(1, iter_max + 1):
                Tc = self.calc_temp(i, T, cooling_rate)
                temperature_history.append(Tc)
                
                if Tc <= min_temp:
                    stop_reason = "temperature_threshold"
                    fitness_curr = self.calculate_tour_length(xcurr)
                    fitness_history.append(fitness_curr)
                    break
                
                xnext = self.random_successor(xcurr)
                
                fitness_curr = self.calculate_tour_length(xcurr)
                fitness_next = self.calculate_tour_length(xnext)
                delta_E = fitness_curr - fitness_next  # Negative delta means improvement
                
                fitness_history.append(fitness_curr)
                
                if delta_E > 0:  # Better solution
                    xcurr = xnext
                    if self.calculate_tour_length(xbest) > self.calculate_tour_length(xcurr):
                        xbest = xcurr.copy()
                elif delta_E < 0:  # Worse solution
                    prob = math.exp(delta_E / Tc) if Tc > 0 else 0
                    if random.random() < prob:
                        xcurr = xnext
                else:  # Same fitness
                    if random.random() < 0.5:
                        xcurr = xnext
                
        except Exception as e:
            import traceback
            print(f"Error in _sim_anneal_core: {e}")
            print(traceback.format_exc())
            stop_reason = "error"
        
        # Check if we found a solution after loop ends
        best_fitness = self.calculate_tour_length(xbest)
        if best_fitness == 0 and stop_reason == "max_iterations":
            stop_reason = "solution_found"
        
        self._stop_reason = stop_reason
        return xbest, self.calculate_tour_length(xbest), temperature_history, fitness_history, stop_reason
    
    def sim_anneal(self, iter_max: int, T: float, 
                   yield_progress: bool = False, min_temp: float = 2.0, cooling_rate: float = 0.95):
        """
        Run Simulated Annealing for TSP.
        
        Args:
            iter_max: Maximum number of iterations
            T: Initial temperature
            yield_progress: Whether to yield progress updates
            min_temp: Minimum temperature threshold
            cooling_rate: Cooling rate
        
        Returns:
            If yield_progress=False: (best_tour, best_fitness, temperature_history, fitness_history, stop_reason)
            If yield_progress=True: Generator yielding (current_tour, iteration, temperature, fitness)
        """
        if not yield_progress:
            return self._sim_anneal_core(iter_max, T, min_temp, cooling_rate)
        
        xcurr = self.initial_config()
        xbest = xcurr.copy()
        
        temperature_history = []
        fitness_history = []
        stop_reason = "max_iterations"
        
        for i in range(1, iter_max + 1):
            Tc = self.calc_temp(i, T, cooling_rate)
            temperature_history.append(Tc)
            
            if Tc <= min_temp:
                stop_reason = "temperature_threshold"
                fitness_curr = self.calculate_tour_length(xcurr)
                fitness_history.append(fitness_curr)
                
                yield (xcurr.copy(), i, Tc, fitness_curr)
                break
            
            xnext = self.random_successor(xcurr)
            
            fitness_curr = self.calculate_tour_length(xcurr)
            fitness_next = self.calculate_tour_length(xnext)
            delta_E = fitness_curr - fitness_next
            
            fitness_history.append(fitness_curr)
            
            yield (xcurr.copy(), i, Tc, fitness_curr)
            
            if delta_E > 0:
                xcurr = xnext
                if self.calculate_tour_length(xbest) > self.calculate_tour_length(xcurr):
                    xbest = xcurr.copy()
            elif delta_E < 0:
                prob = math.exp(delta_E / Tc) if Tc > 0 else 0
                if random.random() < prob:
                    xcurr = xnext
            else:
                if random.random() < 0.5:
                    xcurr = xnext
        
        # Check if we found a solution after loop ends
        best_fitness = self.calculate_tour_length(xbest)
        if best_fitness == 0 and stop_reason == "max_iterations":
            stop_reason = "solution_found"
        
        self._stop_reason = stop_reason
        self._final_result = (xbest, best_fitness, temperature_history, fitness_history, stop_reason)
    
    def sim_anneal_simple(self, iter_max: int, T: float, min_temp: float = 2.0, cooling_rate: float = 0.95):
        """Simple version without progress yielding."""
        return self._sim_anneal_core(iter_max, T, min_temp, cooling_rate)

