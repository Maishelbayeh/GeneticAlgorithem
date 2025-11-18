"""
Simulated Annealing solver for Graph Coloring Problem.
"""
import random
import math
from typing import List, Dict, Tuple, Generator


class GraphColoringSA:
    """
    Simulated Annealing solver for Graph Coloring Problem.
    """
    def __init__(self, graph: Dict[int, List[int]], num_colors: int):
        """
        Initialize the Graph Coloring Simulated Annealing solver.
        
        Args:
            graph: Dictionary representing adjacency list {vertex: [neighbors]}
            num_colors: Number of colors available
        """
        self.graph = graph
        self.num_colors = num_colors
        self.vertices = list(graph.keys())
        self.colors = list(range(num_colors))
        self.color_names = ['Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Orange', 'Cyan', 'Magenta', 'Pink', 'Brown']
    
    def initial_config(self, m: int) -> Dict[int, int]:
        """
        Initialize random coloring configuration.
        
        Args:
            m: Number of vertices (not used, kept for pseudocode compatibility)
        
        Returns:
            Dictionary mapping vertex to color index
        """
        return {vertex: random.choice(self.colors) for vertex in self.vertices}
    
    def calc_temp(self, iteration: int, initial_temp: float) -> float:
        """
        Calculate current temperature based on iteration.
        Uses exponential cooling schedule.
        
        Args:
            iteration: Current iteration number
            initial_temp: Initial temperature
        
        Returns:
            Current temperature
        """
        # Exponential cooling: T = T0 * (cooling_rate ^ iteration)
        cooling_rate = 0.95  # Can be made configurable
        return initial_temp * (cooling_rate ** iteration)
    
    def random_successor(self, current_coloring: Dict[int, int]) -> Dict[int, int]:
        """
        Generate a random successor by changing one vertex's color.
        
        Args:
            current_coloring: Current coloring configuration
        
        Returns:
            New coloring configuration
        """
        new_coloring = current_coloring.copy()
        vertex = random.choice(self.vertices)
        # Choose a different color (not the current one)
        available_colors = [c for c in self.colors if c != current_coloring[vertex]]
        if available_colors:
            new_coloring[vertex] = random.choice(available_colors)
        return new_coloring
    
    def count_conflicts(self, coloring: Dict[int, int]) -> int:
        """
        Count the number of conflicts (adjacent vertices with same color).
        
        Args:
            coloring: Coloring configuration
        
        Returns:
            Number of conflicts
        """
        conflicts = 0
        for vertex, neighbors in self.graph.items():
            for neighbor in neighbors:
                if coloring[vertex] == coloring[neighbor]:
                    conflicts += 1
        # Each conflict is counted twice (once per vertex), so divide by 2
        return conflicts // 2
    
    def sim_anneal(self, m: int, iter_max: int, T: float, 
                   yield_progress: bool = False, min_temp: float = 2.0):
        """
        Simulated Annealing algorithm based on the provided pseudocode.
        Can yield intermediate states for visualization.
        Stops when either max iterations reached or temperature drops below min_temp.
        
        Args:
            m: Number of vertices
            iter_max: Maximum number of iterations
            T: Initial temperature
            yield_progress: If True, yields (current_coloring, iteration, temperature, conflicts) at each step
            min_temp: Minimum temperature threshold (default: 2.0). Algorithm stops when temp <= min_temp
        
        Yields (if yield_progress=True):
            Tuple of (current_coloring, iteration, temperature, conflicts)
        
        Returns:
            If yield_progress=False: Tuple of (best_coloring, temperature_history, conflict_history, stop_reason)
            If yield_progress=True: Generator that yields progress, final result stored in instance
        """
        # Initialize
        xcurr = self.initial_config(m)
        xbest = xcurr.copy()
        
        # Track progress
        temperature_history = []
        conflict_history = []
        stop_reason = "max_iterations"  # Default stop reason
        
        for i in range(1, iter_max + 1):
            # Calculate current temperature
            Tc = self.calc_temp(i, T)
            temperature_history.append(Tc)
            
            # Check if temperature reached minimum threshold
            if Tc <= min_temp:
                stop_reason = "temperature_threshold"
                # Process one more iteration with current state before stopping
                conflicts_curr = self.count_conflicts(xcurr)
                conflict_history.append(conflicts_curr)
                
                if yield_progress:
                    yield (xcurr.copy(), i, Tc, conflicts_curr)
                
                break
            
            # Generate random successor
            xnext = self.random_successor(xcurr)
            
            # Calculate energy difference
            # Energy = number of conflicts (lower is better)
            # ΔE = conflicts_current - conflicts_next (positive means improvement)
            conflicts_curr = self.count_conflicts(xcurr)
            conflicts_next = self.count_conflicts(xnext)
            delta_E = conflicts_curr - conflicts_next
            
            # Track conflicts
            conflict_history.append(conflicts_curr)
            
            # Yield progress if requested
            if yield_progress:
                yield (xcurr.copy(), i, Tc, conflicts_curr)
            
            # Acceptance criteria
            if delta_E > 0:  # Improvement (fewer conflicts)
                xcurr = xnext
                if self.count_conflicts(xbest) > self.count_conflicts(xcurr):
                    xbest = xcurr.copy()
                    # Check if we found a perfect solution (0 conflicts)
                    if self.count_conflicts(xbest) == 0:
                        stop_reason = "solution_found"
                        conflict_history.append(conflicts_next)
                        if yield_progress:
                            yield (xbest.copy(), i, Tc, 0)
                        break
            elif delta_E < 0:  # Worse solution
                # Accept with probability e^(-ΔE/T)
                # Since ΔE is negative, -ΔE/T is positive
                prob = math.exp(delta_E / Tc) if Tc > 0 else 0
                if random.random() < prob:
                    xcurr = xnext
            else:  # delta_E == 0, same quality
                # Can accept to allow exploration
                if random.random() < 0.5:
                    xcurr = xnext
            
            # Check if current solution is perfect (0 conflicts)
            if self.count_conflicts(xcurr) == 0:
                xbest = xcurr.copy()
                stop_reason = "solution_found"
                if yield_progress:
                    yield (xbest.copy(), i, Tc, 0)
                break
        
        # Store stop reason
        self._stop_reason = stop_reason
        
        # Return final result
        if not yield_progress:
            return xbest, temperature_history, conflict_history, stop_reason
        else:
            # Store final result as attribute for access after generator completes
            self._final_result = (xbest, temperature_history, conflict_history, stop_reason)
    
    def sim_anneal_simple(self, m: int, iter_max: int, T: float, min_temp: float = 2.0) -> Tuple[Dict[int, int], List[float], List[int], str]:
        """
        Simple version that doesn't yield (for backward compatibility).
        """
        result = None
        for result in self.sim_anneal(m, iter_max, T, yield_progress=False, min_temp=min_temp):
            pass
        return result

