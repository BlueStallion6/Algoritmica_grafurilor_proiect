import os
import random
import networkx as nx


def generate_random_graph(filename, n_vertices, n_edges, directed=False, weighted=False,
                          min_weight=1, max_weight=10, connected=True):
    """
    Generate a random graph and save it to a file

    Parameters:
    -----------
    filename : str
        The name of the file to save the graph to
    n_vertices : int
        Number of vertices
    n_edges : int
        Number of edges (must be valid for the number of vertices)
    directed : bool
        Whether the graph is directed
    weighted : bool
        Whether the graph has weights
    min_weight, max_weight : int
        Range of weights (if weighted is True)
    connected : bool
        Ensure the graph is connected
    """
    # Check if the number of edges is valid
    if directed:
        max_possible_edges = n_vertices * (n_vertices - 1)
    else:
        max_possible_edges = n_vertices * (n_vertices - 1) // 2

    if n_edges > max_possible_edges:
        print(f"Warning: Too many edges requested ({n_edges}) for {n_vertices} vertices.")
        print(f"Maximum possible edges for this graph type: {max_possible_edges}")
        n_edges = max_possible_edges

    # Create a NetworkX graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Add all nodes
    G.add_nodes_from(range(1, n_vertices + 1))

    # If we want a connected graph, start with a tree
    if connected and not directed:
        # Generate a random tree (spanning tree ensures connectivity)
        T = nx.random_tree(n=n_vertices)
        # Relabel nodes to start from 1
        mapping = {i: i + 1 for i in range(n_vertices)}
        T = nx.relabel_nodes(T, mapping)
        edges = list(T.edges())
        G.add_edges_from(edges)
        # Adjust remaining edges to add
        n_edges -= len(edges)

    # Add remaining random edges
    edges_added = 0
    max_attempts = n_edges * 10  # Avoid infinite loops
    attempts = 0

    while edges_added < n_edges and attempts < max_attempts:
        u = random.randint(1, n_vertices)
        v = random.randint(1, n_vertices)

        # Avoid self-loops
        if u == v:
            attempts += 1
            continue

        # Check if edge already exists
        if G.has_edge(u, v) or (not directed and G.has_edge(v, u)):
            attempts += 1
            continue

        # Add the edge
        if weighted:
            weight = random.randint(min_weight, max_weight)
            G.add_edge(u, v, weight=weight)
        else:
            G.add_edge(u, v)

        edges_added += 1

    # Write to file
    with open(filename, 'w') as f:
        # First line: n m d w
        f.write(f"{n_vertices} {G.number_of_edges()} {1 if directed else 0} {1 if weighted else 0}\n")

        # Write edges
        for u, v, data in G.edges(data=True):
            if weighted:
                f.write(f"{u} {v} {data.get('weight', 1)}\n")
            else:
                f.write(f"{u} {v}\n")

    print(f"Generated graph with {n_vertices} vertices and {G.number_of_edges()} edges.")
    print(f"Graph saved to '{filename}'")
    return G


# Examples of graph types
def generate_example_graphs():
    # Undirected, unweighted graph
    generate_random_graph("simple_graph.txt", 10, 15, directed=False, weighted=False)

    # Undirected, weighted graph
    generate_random_graph("weighted_graph.txt", 8, 12, directed=False, weighted=True)

    # Directed, unweighted graph
    generate_random_graph("directed_graph.txt", 7, 10, directed=True, weighted=False)

    # Directed, weighted graph (for flow problems)
    generate_random_graph("flow_network.txt", 6, 10, directed=True, weighted=True)

    # Bipartite graph
    # For a bipartite graph, we'll manually create one
    create_bipartite_graph("bipartite_graph.txt", 5, 5)  # 5 nodes on each side

    # Complete graph
    n = 5
    generate_random_graph("complete_graph.txt", n, n * (n - 1) // 2, directed=False, weighted=True)

    # Tree
    create_tree("tree_graph.txt")

    # Graph with Eulerian cycle
    create_eulerian_graph("eulerian_graph.txt")


def create_bipartite_graph(filename, n_left, n_right, edge_probability=0.4, weighted=True):
    """Create a bipartite graph with n_left + n_right nodes"""
    G = nx.bipartite.random_graph(n_left, n_right, edge_probability)

    # Relabel nodes to start from 1
    mapping = {i: i + 1 for i in range(n_left + n_right)}
    G = nx.relabel_nodes(G, mapping)

    # Add weights if needed
    if weighted:
        for u, v in G.edges():
            G[u][v]['weight'] = random.randint(1, 10)

    # Write to file
    with open(filename, 'w') as f:
        # First line: n m d w
        f.write(f"{n_left + n_right} {G.number_of_edges()} 0 {1 if weighted else 0}\n")

        # Write edges
        for u, v in G.edges():
            if weighted:
                f.write(f"{u} {v} {G[u][v]['weight']}\n")
            else:
                f.write(f"{u} {v}\n")

    print(f"Generated bipartite graph with {n_left + n_right} vertices and {G.number_of_edges()} edges.")
    print(f"Graph saved to '{filename}'")


def create_tree(filename, n_vertices=10, weighted=True):
    """Create a random tree with n_vertices"""
    T = nx.random_tree(n=n_vertices)

    # Relabel nodes to start from 1
    mapping = {i: i + 1 for i in range(n_vertices)}
    T = nx.relabel_nodes(T, mapping)

    # Add weights if needed
    if weighted:
        for u, v in T.edges():
            T[u][v]['weight'] = random.randint(1, 10)

    # Write to file
    with open(filename, 'w') as f:
        # First line: n m d w
        f.write(f"{n_vertices} {T.number_of_edges()} 0 {1 if weighted else 0}\n")

        # Write edges
        for u, v in T.edges():
            if weighted:
                f.write(f"{u} {v} {T[u][v]['weight']}\n")
            else:
                f.write(f"{u} {v}\n")

    print(f"Generated tree with {n_vertices} vertices.")
    print(f"Graph saved to '{filename}'")


def create_eulerian_graph(filename, n_vertices=8, weighted=True):
    """Create a graph with an Eulerian cycle"""
    # For a graph to have an Eulerian cycle, all vertices must have even degree

    # Start with a cycle
    G = nx.cycle_graph(n_vertices)

    # Relabel nodes to start from 1
    mapping = {i: i + 1 for i in range(n_vertices)}
    G = nx.relabel_nodes(G, mapping)

    # Add some random edges while maintaining even degree
    for _ in range(n_vertices // 2):
        u = random.randint(1, n_vertices)
        v = (u + 2) % n_vertices
        if v == 0:
            v = n_vertices
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    # Add weights if needed
    if weighted:
        for u, v in G.edges():
            G[u][v]['weight'] = random.randint(1, 10)

    # Write to file
    with open(filename, 'w') as f:
        # First line: n m d w
        f.write(f"{n_vertices} {G.number_of_edges()} 0 {1 if weighted else 0}\n")

        # Write edges
        for u, v in G.edges():
            if weighted:
                f.write(f"{u} {v} {G[u][v]['weight']}\n")
            else:
                f.write(f"{u} {v}\n")

    print(f"Generated Eulerian graph with {n_vertices} vertices.")
    print(f"Graph saved to '{filename}'")


if __name__ == "__main__":
    # Generate example graphs
    generate_example_graphs()

    # Generate a custom graph
    generate_random_graph(
        "custom_graph.txt",
        n_vertices=15,
        n_edges=30,
        directed=True,
        weighted=True,
        min_weight=1,
        max_weight=20
    )