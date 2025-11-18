# Graph Coloring with Simulated Annealing

A Streamlit application that solves the Graph Coloring Problem using the Simulated Annealing algorithm.

## Features

- **Interactive Graph Creation**: Create graphs manually or use example graphs
- **Simulated Annealing Algorithm**: Implements the SA algorithm based on the provided pseudocode
- **Configurable Parameters**: Adjust number of colors, initial temperature, and iterations
- **Visualization**: View the graph and its coloring with progress tracking
- **Real-time Progress**: Monitor temperature cooling and conflict reduction over iterations

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## How to Use

1. **Create a Graph**:
   - Choose "Manual Edge Entry" to input edges manually (format: `vertex1,vertex2`, one per line)
   - Or select from example graphs (K4, C5, W5, Bipartite)

2. **Configure Parameters**:
   - Set the number of colors available
   - Adjust initial temperature (higher = more exploration)
   - Set maximum iterations

3. **Run Algorithm**:
   - Click "Run Simulated Annealing" to start the optimization
   - View results including final coloring and conflict count
   - Check progress visualizations to see how the algorithm improves

## Algorithm Implementation

The implementation follows the provided pseudocode:
- `InitialConfig(m)`: Randomly assigns colors to all vertices
- `Calc_Temp(i, T)`: Calculates temperature using exponential cooling
- `Random-Successor(x_curr)`: Generates new state by changing one vertex's color
- Energy function: Number of conflicts (adjacent vertices with same color)
- Acceptance criteria: Accept improvements, accept worse solutions with probability `e^(-ΔE/T)`

## Requirements

- Python 3.8+
- streamlit
- networkx
- matplotlib
- numpy

