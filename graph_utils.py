"""
Graph utilities for creating and visualizing graphs.
"""
import random
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def create_graph_from_edges(edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """
    Create adjacency list from list of edges.
    
    Args:
        edges: List of (vertex1, vertex2) tuples
    
    Returns:
        Dictionary representing adjacency list
    """
    graph = {}
    for v1, v2 in edges:
        if v1 not in graph:
            graph[v1] = []
        if v2 not in graph:
            graph[v2] = []
        if v2 not in graph[v1]:
            graph[v1].append(v2)
        if v1 not in graph[v2]:
            graph[v2].append(v1)
    return graph


def generate_random_graph(num_nodes: int, edge_probability: float = 0.3) -> Dict[int, List[int]]:
    """
    Generate a random graph with specified number of nodes.
    
    Args:
        num_nodes: Number of vertices in the graph
        edge_probability: Probability of an edge between any two vertices
    
    Returns:
        Dictionary representing adjacency list
    """
    graph = {i: [] for i in range(num_nodes)}
    
    # Generate edges randomly
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_probability:
                graph[i].append(j)
                graph[j].append(i)
    
    return graph


def visualize_graph(graph: Dict[int, List[int]], coloring: Dict[int, int] = None, 
                   color_names: List[str] = None) -> plt.Figure:
    """
    Visualize the graph with coloring.
    
    Args:
        graph: Graph as adjacency list
        coloring: Optional coloring configuration
        color_names: List of color names
    
    Returns:
        Matplotlib figure
    """
    G = nx.Graph()
    
    # Add edges
    for vertex, neighbors in graph.items():
        for neighbor in neighbors:
            if not G.has_edge(vertex, neighbor):
                G.add_edge(vertex, neighbor)
    
    # Add isolated vertices
    for vertex in graph.keys():
        if vertex not in G:
            G.add_node(vertex)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Node colors
    if coloring and color_names:
        node_colors = [coloring.get(node, 0) for node in G.nodes()]
        color_map = plt.cm.get_cmap('tab10')
        node_color_list = [color_map(node_colors[i] % 10) for i in range(len(node_colors))]
    else:
        node_color_list = 'lightblue'
    
    # Draw
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_color_list,
            node_size=1000, font_size=12, font_weight='bold', 
            edge_color='gray', width=2)
    
    ax.set_title("Graph Visualization", fontsize=16, fontweight='bold')
    return fig

