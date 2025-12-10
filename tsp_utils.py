"""
TSP utilities for visualization and distance calculation
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
import math


def calculate_distance_km(city1: Tuple[float, float], city2: Tuple[float, float]) -> float:
    """
    Calculate distance between two cities using Haversine formula (in kilometers).
    
    Args:
        city1: (latitude, longitude) of first city
        city2: (latitude, longitude) of second city
    
    Returns:
        Distance in kilometers
    """
    lat1, lon1 = city1
    lat2, lon2 = city2
    
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance


def visualize_tsp_tour(cities: Dict[int, Tuple[float, float]], 
                       tour: Optional[List[int]] = None,
                       city_names: Optional[Dict[int, str]] = None,
                       title: str = "TSP Tour Visualization",
                       show_labels: bool = True) -> plt.Figure:
    """
    Visualize TSP tour on a map.
    
    Args:
        cities: Dictionary mapping city ID to (latitude, longitude)
        tour: Optional tour (list of city IDs in order)
        city_names: Optional dictionary mapping city ID to name
        title: Title for the plot
        show_labels: Whether to show city labels
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extract coordinates
    lats = [coord[0] for coord in cities.values()]
    lons = [coord[1] for coord in cities.values()]
    city_ids = list(cities.keys())
    
    # Plot cities
    ax.scatter(lons, lats, c='red', s=200, zorder=5, edgecolors='black', linewidths=2)
    
    # Plot tour path if provided
    if tour and len(tour) > 1:
        # Draw the tour path
        tour_lons = [cities[city_id][1] for city_id in tour]
        tour_lats = [cities[city_id][0] for city_id in tour]
        
        # Close the tour (return to start)
        tour_lons.append(tour_lons[0])
        tour_lats.append(tour_lats[0])
        
        # Draw path
        ax.plot(tour_lons, tour_lats, 'b-', linewidth=2, alpha=0.6, zorder=3, label='Tour Path')
        
        # Highlight start/end point
        start_city = tour[0]
        start_lat, start_lon = cities[start_city]
        ax.scatter([start_lon], [start_lat], c='green', s=400, zorder=6, 
                  marker='*', edgecolors='black', linewidths=2, label='Start/End')
    
    # Add city labels
    if show_labels and city_names:
        for city_id, (lat, lon) in cities.items():
            city_name = city_names.get(city_id, f"City {city_id}")
            # Shorten long names for display
            if len(city_name) > 8:
                city_name = city_name[:8] + "..."
            ax.annotate(city_name, (lon, lat), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def calculate_tour_distance_km(cities: Dict[int, Tuple[float, float]], 
                                tour: List[int]) -> float:
    """
    Calculate total tour distance in kilometers.
    
    Args:
        cities: Dictionary mapping city ID to (latitude, longitude)
        tour: List of city IDs in order
    
    Returns:
        Total distance in kilometers
    """
    if len(tour) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(tour)):
        current = tour[i]
        next_city = tour[(i + 1) % len(tour)]  # Wrap around to start
        total += calculate_distance_km(cities[current], cities[next_city])
    
    return total


def generate_random_cities(num_cities: int, 
                          lat_range: Tuple[float, float] = (31.0, 32.5),
                          lon_range: Tuple[float, float] = (34.0, 35.5)) -> Dict[int, Tuple[float, float]]:
    """
    Generate random cities within specified ranges.
    
    Args:
        num_cities: Number of cities to generate
        lat_range: (min_lat, max_lat)
        lon_range: (min_lon, max_lon)
    
    Returns:
        Dictionary mapping city ID to (latitude, longitude)
    """
    import random
    cities = {}
    min_lat, max_lat = lat_range
    min_lon, max_lon = lon_range
    
    for i in range(num_cities):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        cities[i] = (lat, lon)
    
    return cities

