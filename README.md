# Palestinian Delivery TSP Application

A Streamlit application that solves the Traveling Salesman Problem (TSP) using Palestinian cities/regions. The app finds the optimal route to visit selected locations with minimum cost (distance) using two optimization algorithms: Simulated Annealing (SA) and Genetic Algorithm (GA).

## Features

- **50 Real Palestinian Regions**: Uses real coordinates for 50 Palestinian cities/regions
- **Flexible City Selection**: Choose all 50 cities or select specific cities to visit
- **Two Optimization Algorithms**: 
  - **Simulated Annealing (SA)**: Fast, single-solution approach
  - **Genetic Algorithm (GA)**: Population-based, better exploration
- **Real Distance Calculation**: Uses Haversine formula to calculate distances in kilometers
- **Live Visualization**: Shows the tour path on a map with live updates during optimization
- **Algorithm Comparison**: Compare SA and GA performance side by side
- **Configurable Parameters**: Full control over algorithm parameters
- **Progress Tracking**: Visualize algorithm progress with charts and metrics

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## How to Use

### 1. Select Cities
- **Option 1**: Check "Use All 50 Cities" to include all Palestinian regions
- **Option 2**: Uncheck and select specific cities from the dropdown list (minimum 2 cities required)

### 2. Choose Algorithm
- **Simulated Annealing (SA)**: Fast algorithm, good for quick results
- **Genetic Algorithm (GA)**: Better exploration, recommended for larger problems
- **Compare Both**: Run both algorithms and compare results side by side

### 3. Configure Parameters

#### Simulated Annealing Parameters:
- **Initial Temperature**: Higher values allow more exploration initially (default: 5000)
- **Minimum Temperature**: Stop threshold (default: 2.0)
- **Cooling Rate**: Rate at which temperature decreases (default: 0.95)
- **Max Iterations**: Maximum number of iterations (default: 1000)

#### Genetic Algorithm Parameters:
- **Population Size**: Number of solutions in each generation (default: 100, recommended: 100-200)
- **Number of Generations**: How many generations to evolve (default: 200, recommended: 200-500)
- **Crossover Rate**: Probability of combining two solutions (default: 0.8)
- **Mutation Rate**: Probability of random changes (default: 0.1)
- **Elitism Count**: Number of best solutions to keep (default: 2)
- **Tournament Size**: Size for tournament selection (default: 3)
- **Crossover Type**: Order Crossover (OX) or Partially Mapped Crossover (PMX)
- **Mutation Type**: Swap or Inversion

### 4. Run the Algorithm
- Click the run button
- Watch the route evolve on the map in real-time (if animation is enabled)
- Review final results, distance, execution time, and detailed tour information

## Algorithms

### Simulated Annealing (SA)

**How it works:**
- Starts with a random tour
- Gradually improves by accepting better solutions
- Sometimes accepts worse solutions (based on temperature) to escape local optima
- Temperature cools down gradually to reduce exploration of bad solutions

**Best for:**
- Quick results
- Small to medium problems (< 20 cities)
- When speed is more important than optimality

**Pros:**
- ✅ Fast execution
- ✅ Less memory usage
- ✅ Simple to understand

**Cons:**
- ⚠️ May get stuck in local optima
- ⚠️ Single solution path

### Genetic Algorithm (GA)

**How it works:**
- Starts with a population of random tours
- Evolves solutions through:
  - **Crossover**: Combines two tours to create a new one
  - **Mutation**: Randomly changes parts of a tour
  - **Selection**: Chooses best tours for next generation
- Uses elitism to preserve best solutions

**Best for:**
- Better quality solutions
- Larger problems (> 20 cities)
- When optimality is more important than speed

**Pros:**
- ✅ Better exploration of solution space
- ✅ Multiple solutions evolve together
- ✅ Better at escaping local optima

**Cons:**
- ⚠️ Slower execution
- ⚠️ Needs proper parameter tuning (population size, generations)

## When to Use SA vs GA?

| Scenario | Recommendation |
|----------|---------------|
| Quick results needed | **SA** |
| Small problem (< 20 cities) | **SA** |
| Large problem (> 20 cities) | **GA** (with proper tuning) |
| Best quality solution needed | **GA** (population_size=150-200, generations=300-500) |
| Limited computational resources | **SA** |
| Can wait for better results | **GA** |

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── tsp_sa_solver.py        # Simulated Annealing implementation for TSP
├── ga_solver.py            # Genetic Algorithm implementation for TSP
├── palestinian_cities.py   # 50 Palestinian cities/regions data
├── tsp_utils.py            # Utility functions for visualization and distance calculation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Requirements

- Python 3.8+
- streamlit >= 1.28.0
- matplotlib >= 3.7.0
- numpy >= 1.24.0

## Technical Details

### Distance Calculation
- Uses **Haversine formula** to calculate distances between cities
- Accounts for Earth's curvature
- Results in kilometers

### Visualization
- Uses matplotlib for route visualization
- Shows cities as points on a coordinate map
- Displays tour path as connected lines
- Highlights start/end point

### Performance Tips

**For SA:**
- Increase iterations for better results
- Adjust cooling rate: lower = faster but may miss solutions
- Higher initial temperature = more exploration

**For GA:**
- **Population size**: 100-200 recommended for good balance
- **Generations**: 200-500 for better quality
- **Crossover rate**: 0.7-0.9 typically works well
- **Mutation rate**: 0.05-0.15 prevents premature convergence

## Notes

- Distances are calculated using Haversine formula (accurate for Earth's curvature)
- The application displays routes on a coordinate map using geographic coordinates
- You can adjust all parameters to get the best performance for your specific problem
- GA typically needs more time but can find better solutions with proper tuning

## License

MIT License

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Acknowledgments

- Palestinian cities data with real geographic coordinates
- TSP algorithms: Simulated Annealing and Genetic Algorithm implementations
