import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import heapq
import random
import os


class GraphAnalysis:
    def __init__(self, filename):
        """
        Initialize the graph analysis with a filename
        """
        self.filename = filename
        self.graph = self.load_graph(filename)
        self.adjacency_list = {}
        self.adjacency_matrix = None
        self.edge_list = []
        self.incidence_matrix = None
        self.num_vertices = 0
        self.num_edges = 0
        self.is_directed = False
        self.is_weighted = False

    def load_graph(self, filename):
        """
        Load graph from file. The format is expected to be:
        First line: n m d w (n=vertices, m=edges, d=directed(0/1), w=weighted(0/1))
        Next m lines: u v [w] (u,v = vertices, w = optional weight)
        """
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist")

            with open(filename, 'r') as f:
                lines = f.readlines()

            if not lines:
                raise ValueError("File is empty")

            # Parse first line for metadata
            parts = lines[0].strip().split()
            if len(parts) != 4:
                raise ValueError(f"Invalid first line format. Expected 'n m d w' but got: {lines[0]}")

            n, m, d, w = map(int, parts)
            self.num_vertices = n
            self.num_edges = m
            self.is_directed = d == 1
            self.is_weighted = w == 1

            # Create a NetworkX graph
            if self.is_directed:
                G = nx.DiGraph()
            else:
                G = nx.Graph()

            # Add all vertices
            for i in range(1, n + 1):
                G.add_node(i)

            # Parse edges
            if len(lines) < m + 1:
                raise ValueError(f"Expected {m} edges, but only found {len(lines) - 1}")

            for i in range(1, m + 1):
                try:
                    edge_data = list(map(int, lines[i].strip().split()))
                    if self.is_weighted:
                        if len(edge_data) != 3:
                            raise ValueError(
                                f"Invalid edge format on line {i + 1}. Expected 'u v weight' for weighted graph")
                        u, v, weight = edge_data
                    else:
                        if len(edge_data) != 2:
                            raise ValueError(
                                f"Invalid edge format on line {i + 1}. Expected 'u v' for unweighted graph")
                        u, v = edge_data
                        weight = 1

                    # Check if vertices are valid
                    if u < 1 or u > n or v < 1 or v > n:
                        raise ValueError(f"Invalid vertex on line {i + 1}. Vertices must be between 1 and {n}")

                    G.add_edge(u, v, weight=weight)
                except Exception as e:
                    raise ValueError(f"Error parsing edge on line {i + 1}: {e}")

            return G
        except Exception as e:
            print(f"Error loading graph: {e}")
            print(f"Please check that your file '{filename}' follows the correct format:")
            print("First line: n m d w (n=vertices, m=edges, d=directed(0/1), w=weighted(0/1))")
            print("Following lines: u v [w] (edges with optional weight)")
            return None

    def create_representations(self):
        """
        Create different representations of the graph:
        - Adjacency list
        - Adjacency matrix (with weights if weighted)
        - Edge list
        - Incidence matrix
        """
        G = self.graph
        n = self.num_vertices

        # Make sure we have a valid graph
        if G is None:
            print("Error: Graph is None. Cannot create representations.")
            return

        if n <= 0:
            print("Error: Invalid number of vertices:", n)
            return

        # Create adjacency list
        self.adjacency_list = {node: [] for node in G.nodes()}
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            self.adjacency_list[u].append((v, weight))
            if not self.is_directed:
                self.adjacency_list[v].append((u, weight))

        # Create adjacency matrix (with size n+1 for 1-based indexing)
        self.adjacency_matrix = np.zeros((n + 1, n + 1))
        for u, v, data in G.edges(data=True):
            if u > n or v > n:
                print(f"Warning: Edge ({u}, {v}) has vertices out of range (1-{n})")
                continue
            weight = data.get('weight', 1)
            self.adjacency_matrix[u][v] = weight
            if not self.is_directed:
                self.adjacency_matrix[v][u] = weight

        # Create edge list
        self.edge_list = [(u, v, data.get('weight', 1)) for u, v, data in G.edges(data=True)]

        # Create incidence matrix
        if len(self.edge_list) > 0:  # Only create if we have edges
            if not self.is_directed:
                # For undirected graphs
                self.incidence_matrix = np.zeros((n + 1, len(self.edge_list)))
                for i, (u, v, _) in enumerate(self.edge_list):
                    if u <= n and i < self.incidence_matrix.shape[1]:  # Ensure indices are valid
                        self.incidence_matrix[u][i] = 1
                    if v <= n and i < self.incidence_matrix.shape[1]:  # Ensure indices are valid
                        self.incidence_matrix[v][i] = 1
            else:
                # For directed graphs
                self.incidence_matrix = np.zeros((n + 1, len(self.edge_list)))
                for i, (u, v, _) in enumerate(self.edge_list):
                    if u <= n and i < self.incidence_matrix.shape[1]:  # Ensure indices are valid
                        self.incidence_matrix[u][i] = -1  # Outgoing edge
                    if v <= n and i < self.incidence_matrix.shape[1]:  # Ensure indices are valid
                        self.incidence_matrix[v][i] = 1  # Incoming edge
        else:
            self.incidence_matrix = np.zeros((n + 1, 1))  # Empty matrix with proper dimensions

    def print_representations(self):
        """
        Print all graph representations
        """
        print("Graph Information:")
        print(f"Number of vertices: {self.num_vertices}")
        print(f"Number of edges: {self.num_edges}")
        print(f"Directed: {self.is_directed}")
        print(f"Weighted: {self.is_weighted}")

        print("\nAdjacency List:")
        for node, neighbors in self.adjacency_list.items():
            print(f"{node}: {neighbors}")

        print("\nAdjacency Matrix:")
        if self.adjacency_matrix is not None and self.adjacency_matrix.shape[0] > 1:
            try:
                # Make sure we don't try to access rows/columns that don't exist
                end_idx = min(self.num_vertices + 1, self.adjacency_matrix.shape[0])
                print(self.adjacency_matrix[1:end_idx, 1:end_idx])
            except Exception as e:
                print(f"Error displaying adjacency matrix: {e}")
                print(f"Matrix shape: {self.adjacency_matrix.shape}")
        else:
            print("Adjacency matrix not available")

        print("\nEdge List:")
        for edge in self.edge_list:
            print(edge)

        print("\nIncidence Matrix:")
        if self.incidence_matrix is not None and self.incidence_matrix.shape[0] > 1:
            try:
                # Make sure we don't try to access rows that don't exist
                end_idx = min(self.num_vertices + 1, self.incidence_matrix.shape[0])
                print(self.incidence_matrix[1:end_idx, :])
            except Exception as e:
                print(f"Error displaying incidence matrix: {e}")
                print(f"Matrix shape: {self.incidence_matrix.shape}")
        else:
            print("Incidence matrix not available")

    def visualize_graph(self):
        """
        Visualize the graph using NetworkX and Matplotlib
        """
        G = self.graph
        if G is None or G.number_of_nodes() == 0:
            print("Cannot visualize graph: Graph is empty or None")
            return

        try:
            # Create a layout for the graph
            try:
                pos = nx.spring_layout(G, seed=42)  # Use seed for consistency
            except:
                # Fallback to another layout if spring_layout fails
                pos = nx.kamada_kawai_layout(G)

            plt.figure(figsize=(10, 8))
            nx.draw(G, pos, with_labels=True, node_color='lightblue',
                    node_size=500, arrows=self.is_directed)

            if self.is_weighted:
                try:
                    edge_labels = {(u, v): d.get('weight', 1) for u, v, d in G.edges(data=True)}
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                except Exception as e:
                    print(f"Warning: Could not draw edge labels: {e}")

            plt.title("Graph Visualization")
            try:
                plt.savefig("graph_visualization.png")
                print("Graph visualization saved as 'graph_visualization.png'")
            except Exception as e:
                print(f"Warning: Could not save visualization: {e}")

            plt.show()
        except Exception as e:
            print(f"Error visualizing graph: {e}")
            print("Continuing with analysis without visualization...")

    # 1. Connectivity and Topological Sorting
    def analyze_connectivity(self):
        """
        Analyze the connectivity of the graph
        """
        G = self.graph
        results = {
            "is_connected": nx.is_connected(G) if not self.is_directed else None,
            "components": list(nx.connected_components(G)) if not self.is_directed else None,
            "strongly_connected": list(nx.strongly_connected_components(G)) if self.is_directed else None,
            "weakly_connected": list(nx.weakly_connected_components(G)) if self.is_directed else None,
            "bridges": list(nx.bridges(G)) if not self.is_directed else None,
            "articulation_points": list(nx.articulation_points(G)) if not self.is_directed else None
        }

        print("\nConnectivity Analysis:")
        if not self.is_directed:
            print(f"Is connected: {results['is_connected']}")
            print(f"Number of connected components: {len(results['components'])}")
            print(f"Connected components: {[list(comp) for comp in results['components']]}")
            print(f"Bridges (critical edges): {results['bridges']}")
            print(f"Articulation points (cut vertices): {results['articulation_points']}")
        else:
            print(f"Number of strongly connected components: {len(results['strongly_connected'])}")
            print(f"Strongly connected components: {[list(comp) for comp in results['strongly_connected']]}")
            print(f"Number of weakly connected components: {len(results['weakly_connected'])}")

        return results

    def topological_sort(self):
        """
        Perform topological sorting if the graph is directed and acyclic
        """
        G = self.graph

        print("\nTopological Sorting:")
        if not self.is_directed:
            print("Topological sorting is only applicable to directed graphs.")
            return None

        try:
            topo_order = list(nx.topological_sort(G))
            print(f"Topological order: {topo_order}")
            return topo_order
        except nx.NetworkXUnfeasible:
            print("The graph contains cycles, topological sorting is not possible.")
            return None

    # 2. Shortest Path Algorithms
    def shortest_paths(self, source=1):
        """
        Compute shortest paths from source to all other vertices
        """
        G = self.graph

        # Check if the graph has negative weights
        has_negative_weights = any(data.get('weight', 1) < 0 for _, _, data in G.edges(data=True))

        print(f"\nShortest Paths from vertex {source}:")

        # Dijkstra for non-negative weights
        if not has_negative_weights:
            print("Using Dijkstra's Algorithm (for non-negative weights):")
            try:
                distances = nx.single_source_dijkstra_path_length(G, source)
                paths = nx.single_source_dijkstra_path(G, source)

                for target, distance in distances.items():
                    if target != source:
                        print(f"To {target}: Distance = {distance}, Path = {paths[target]}")

                return {"distances": distances, "paths": paths}
            except nx.NetworkXNoPath:
                print("No path exists from the source to some vertices.")
                return None

        # Bellman-Ford for graphs with negative weights
        else:
            print("Using Bellman-Ford Algorithm (handles negative weights):")
            try:
                # NetworkX doesn't have a direct Bellman-Ford path function, so we'll implement it
                # First, get the distances
                distances = nx.single_source_bellman_ford_path_length(G, source)

                # Then reconstruct the paths
                paths = self._bellman_ford_paths(source)

                for target, distance in distances.items():
                    if target != source:
                        path = paths.get(target, [])
                        print(f"To {target}: Distance = {distance}, Path = {path}")

                return {"distances": distances, "paths": paths}
            except nx.NetworkXNegativeCycle:
                print("The graph contains a negative cycle, shortest paths are not well-defined.")
                return None

    def _bellman_ford_paths(self, source):
        """
        Helper function to compute Bellman-Ford paths
        """
        G = self.graph
        n = self.num_vertices

        # Initialize distances and predecessors
        dist = {node: float('inf') for node in G.nodes()}
        dist[source] = 0
        pred = {node: None for node in G.nodes()}

        # Relax edges repeatedly
        for _ in range(n - 1):
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 1)
                if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    pred[v] = u

        # Check for negative cycles
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                raise nx.NetworkXNegativeCycle("Graph contains a negative cycle")

        # Reconstruct paths
        paths = {}
        for node in G.nodes():
            if node == source:
                paths[node] = [source]
                continue

            if dist[node] == float('inf'):
                paths[node] = []
                continue

            path = [node]
            curr = node
            while curr != source:
                curr = pred[curr]
                path.append(curr)
            path.reverse()
            paths[node] = path

        return paths

    def floyd_warshall(self):
        """
        Compute all-pairs shortest paths using the Floyd-Warshall algorithm
        """
        G = self.graph

        print("\nAll-Pairs Shortest Paths (Floyd-Warshall):")
        try:
            # Get the shortest path lengths between all pairs of nodes
            path_lengths = dict(nx.floyd_warshall(G))

            # Print a sample of the results
            sample_size = min(5, self.num_vertices)
            sample_nodes = random.sample(list(G.nodes()), sample_size)

            print(f"Showing a sample of {sample_size} nodes:")
            for u in sample_nodes:
                for v in sample_nodes:
                    if u != v:
                        dist = path_lengths[u][v]
                        if dist == float('inf'):
                            print(f"No path from {u} to {v}")
                        else:
                            print(f"Shortest distance from {u} to {v}: {dist}")

            return path_lengths
        except:
            print("An error occurred while computing all-pairs shortest paths.")
            return None

    # 3. Minimum Spanning Tree
    def minimum_spanning_tree(self):
        """
        Compute the minimum spanning tree using Kruskal's or Prim's algorithm
        """
        G = self.graph

        if self.is_directed:
            print("\nMinimum Spanning Tree:")
            print("MST is only defined for undirected graphs.")
            return None

        print("\nMinimum Spanning Tree:")

        # Check if the graph is connected
        if not nx.is_connected(G):
            print("The graph is not connected, so a spanning tree does not exist.")
            return None

        # Compute MST using NetworkX (Kruskal's algorithm)
        mst = nx.minimum_spanning_tree(G)

        # Calculate total weight
        total_weight = sum(data.get('weight', 1) for _, _, data in mst.edges(data=True))

        print(f"MST edges: {list(mst.edges())}")
        print(f"Total MST weight: {total_weight}")

        # Visualize the MST
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)

        # Draw the original graph in light gray
        nx.draw(G, pos, with_labels=True, node_color='lightgray',
                node_size=500, edge_color='lightgray', width=1)

        # Draw the MST in blue
        nx.draw_networkx_edges(mst, pos, edge_color='blue', width=2)

        # Draw edge labels if weighted
        if self.is_weighted:
            edge_labels = {(u, v): d['weight'] for u, v, d in mst.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Minimum Spanning Tree")
        plt.savefig("mst_visualization.png")
        plt.show()

        return mst

    # 4. Eulerian and Hamiltonian Cycles
    def eulerian_analysis(self):
        """
        Analyze the graph for Eulerian paths and cycles
        """
        G = self.graph

        print("\nEulerian Analysis:")

        # Check if the graph is Eulerian (has an Eulerian cycle)
        is_eulerian = nx.is_eulerian(G)
        print(f"Is Eulerian (has Eulerian cycle): {is_eulerian}")

        # Check if the graph has an Eulerian path
        has_eulerian_path = nx.has_eulerian_path(G)
        print(f"Has Eulerian path: {has_eulerian_path}")

        # Find Eulerian path or cycle if it exists
        if is_eulerian:
            cycle = list(nx.eulerian_circuit(G))
            print(f"Eulerian cycle: {cycle}")
            return {"is_eulerian": True, "eulerian_cycle": cycle}
        elif has_eulerian_path:
            # For graphs with Eulerian paths, we need to find the start and end vertices
            if self.is_directed:
                # For directed graphs
                in_degree = G.in_degree()
                out_degree = G.out_degree()
                start = None
                end = None

                for node in G.nodes():
                    if out_degree[node] - in_degree[node] == 1:
                        start = node
                    elif in_degree[node] - out_degree[node] == 1:
                        end = node

                if start and end:
                    path = list(nx.eulerian_path(G, source=start))
                    print(f"Eulerian path from {start} to {end}: {path}")
                    return {"is_eulerian": False, "has_path": True, "eulerian_path": path}
            else:
                # For undirected graphs
                odd_degree_nodes = [node for node, degree in G.degree() if degree % 2 == 1]
                if len(odd_degree_nodes) == 2:
                    start, end = odd_degree_nodes
                    path = list(nx.eulerian_path(G, source=start))
                    print(f"Eulerian path from {start} to {end}: {path}")
                    return {"is_eulerian": False, "has_path": True, "eulerian_path": path}

        return {"is_eulerian": False, "has_path": has_eulerian_path}

    def hamiltonian_analysis(self):
        """
        Analyze the graph for Hamiltonian paths and cycles

        Note: Finding Hamiltonian paths/cycles is NP-complete. This uses a simple
        backtracking approach and may be slow for larger graphs.
        """
        G = self.graph
        n = self.num_vertices

        print("\nHamiltonian Analysis:")

        # For complete graphs, we know there's always a Hamiltonian cycle
        if nx.is_complete_graph(G):
            print("The graph is complete, so it has a Hamiltonian cycle.")
            hamiltonian_cycle = list(G.nodes()) + [list(G.nodes())[0]]
            print(f"A Hamiltonian cycle: {hamiltonian_cycle}")
            return {"has_cycle": True, "cycle": hamiltonian_cycle}

        # For larger graphs, we'll use Dirac's and Ore's theorems to check if a Hamiltonian cycle might exist
        if n >= 3:
            # Dirac's theorem: If minimum degree >= n/2, then G has a Hamiltonian cycle
            min_degree = min(dict(G.degree()).values())
            if min_degree >= n / 2:
                print("By Dirac's theorem, the graph has a Hamiltonian cycle (min degree >= n/2).")
                return {"has_cycle": True}

            # Ore's theorem: If sum of degrees of non-adjacent vertices >= n, then G has a Hamiltonian cycle
            for u in G.nodes():
                for v in G.nodes():
                    if u != v and not G.has_edge(u, v):
                        if G.degree(u) + G.degree(v) >= n:
                            print("By Ore's theorem, the graph has a Hamiltonian cycle.")
                            return {"has_cycle": True}

        # For smaller graphs, we can try to find a cycle using backtracking
        if n <= 20:  # Limit to small graphs to avoid long computations
            print("Attempting to find a Hamiltonian cycle using backtracking (may take time)...")
            cycle = self._find_hamiltonian_cycle()
            if cycle:
                print(f"Found Hamiltonian cycle: {cycle}")
                return {"has_cycle": True, "cycle": cycle}
            else:
                print("No Hamiltonian cycle found.")

                # Try to find a Hamiltonian path
                path = self._find_hamiltonian_path()
                if path:
                    print(f"Found Hamiltonian path: {path}")
                    return {"has_cycle": False, "has_path": True, "path": path}
                else:
                    print("No Hamiltonian path found.")
                    return {"has_cycle": False, "has_path": False}
        else:
            print("Graph is too large for exhaustive search of Hamiltonian cycles/paths.")
            return {"has_cycle": "unknown", "has_path": "unknown"}

    def _find_hamiltonian_cycle(self):
        """
        Helper function to find a Hamiltonian cycle using backtracking
        """
        G = self.graph
        n = self.num_vertices
        path = [1]  # Start with the first node
        visited = {1: True}

        def backtrack():
            if len(path) == n:
                # Check if there's an edge back to the starting vertex
                if G.has_edge(path[-1], path[0]):
                    return True
                return False

            for neighbor in G.neighbors(path[-1]):
                if neighbor not in visited:
                    visited[neighbor] = True
                    path.append(neighbor)
                    if backtrack():
                        return True
                    path.pop()
                    del visited[neighbor]
            return False

        if backtrack():
            path.append(path[0])  # Add the starting vertex to close the cycle
            return path
        return None

    def _find_hamiltonian_path(self):
        """
        Helper function to find a Hamiltonian path using backtracking
        """
        G = self.graph
        n = self.num_vertices

        # Try starting from each vertex
        for start in G.nodes():
            path = [start]
            visited = {start: True}

            def backtrack():
                if len(path) == n:
                    return True

                for neighbor in G.neighbors(path[-1]):
                    if neighbor not in visited:
                        visited[neighbor] = True
                        path.append(neighbor)
                        if backtrack():
                            return True
                        path.pop()
                        del visited[neighbor]
                return False

            if backtrack():
                return path
        return None

    # 5. Graph Matching
    def maximum_matching(self):
        """
        Find a maximum matching in the graph
        """
        G = self.graph

        print("\nMaximum Matching Analysis:")

        if self.is_directed:
            print("Matching algorithms are typically applied to undirected graphs.")
            return None

        # Check if the graph is bipartite
        is_bipartite = nx.is_bipartite(G)

        if is_bipartite:
            print("The graph is bipartite.")

            # Get the two sets of nodes in the bipartite graph
            X, Y = nx.bipartite.sets(G)

            # Find maximum bipartite matching
            matching = nx.bipartite.maximum_matching(G, top_nodes=X)

            # Convert to list of edges
            matching_edges = [(u, matching[u]) for u in matching if u in X]

            print(f"Maximum bipartite matching size: {len(matching_edges)}")
            print(f"Matching edges: {matching_edges}")

            return {"is_bipartite": True, "matching": matching_edges}
        else:
            print("The graph is not bipartite.")

            # Find maximum matching for general graphs
            matching = nx.max_weight_matching(G)
            matching_edges = list(matching)

            print(f"Maximum matching size: {len(matching_edges)}")
            print(f"Matching edges: {matching_edges}")

            return {"is_bipartite": False, "matching": matching_edges}

    # 6. Maximum Flow
    def maximum_flow(self, source=None, sink=None):
        """
        Compute the maximum flow in a network
        """
        G = self.graph

        print("\nMaximum Flow Analysis:")

        if not self.is_directed:
            print("Flow algorithms are typically applied to directed graphs.")
            return None

        if not source or not sink:
            # If source/sink not specified, try to guess reasonable values
            in_degree = G.in_degree()
            out_degree = G.out_degree()

            # Choose a node with maximum out-degree as source
            source_candidates = sorted(G.nodes(), key=lambda n: out_degree[n], reverse=True)
            source = source_candidates[0] if source_candidates else 1

            # Choose a node with maximum in-degree as sink
            sink_candidates = sorted(G.nodes(), key=lambda n: in_degree[n], reverse=True)
            sink_candidates = [n for n in sink_candidates if n != source]
            sink = sink_candidates[0] if sink_candidates else self.num_vertices

        print(f"Source node: {source}")
        print(f"Sink node: {sink}")

        try:
            # Compute maximum flow using Ford-Fulkerson algorithm
            flow_value, flow_dict = nx.maximum_flow(G, source, sink)

            print(f"Maximum flow value: {flow_value}")
            print("Flow on each edge:")
            for u in flow_dict:
                for v, flow in flow_dict[u].items():
                    if flow > 0:
                        print(f"Edge ({u}, {v}): Flow = {flow}")

            # Find the minimum cut
            cut_value, partition = nx.minimum_cut(G, source, sink)
            reachable, non_reachable = partition

            print(f"Minimum cut value: {cut_value}")
            print(f"Nodes reachable from source: {reachable}")
            print(f"Nodes not reachable from source: {non_reachable}")

            return {
                "flow_value": flow_value,
                "flow_dict": flow_dict,
                "cut_value": cut_value,
                "partition": partition
            }
        except nx.NetworkXError as e:
            print(f"Error computing maximum flow: {e}")
            return None

    # 7. Graph Coloring
    def graph_coloring(self):
        """
        Compute a graph coloring
        """
        G = self.graph

        print("\nGraph Coloring Analysis:")

        if self.is_directed:
            print("Coloring algorithms are typically applied to undirected graphs.")
            return None

        # Compute a greedy coloring
        coloring = nx.greedy_color(G)

        # Count the number of colors used
        num_colors = max(coloring.values()) + 1

        print(f"Greedy coloring uses {num_colors} colors.")
        print(f"Coloring: {coloring}")

        # For bipartite graphs, we know the chromatic number is 2
        is_bipartite = nx.is_bipartite(G)
        if is_bipartite:
            print("The graph is bipartite, so its chromatic number is 2.")

        # Visualize the coloring
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)

        # Create a list of colors for visualization
        color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Map node colors
        node_colors = [color_map[coloring[node] % len(color_map)] for node in G.nodes()]

        # Draw the graph with node colors
        nx.draw(G, pos, with_labels=True, node_color=node_colors,
                node_size=500, edge_color='gray')

        plt.title("Graph Coloring")
        plt.savefig("coloring_visualization.png")
        plt.show()

        return {"num_colors": num_colors, "coloring": coloring, "is_bipartite": is_bipartite}

    def run_all_analyses(self):
        """
        Run all graph analyses
        """
        if self.graph is None:
            print("Error: No graph loaded. Cannot perform analysis.")
            return

        print("=== Creating Graph Representations ===")
        self.create_representations()
        self.print_representations()

        print("\n=== Visualizing Graph ===")
        self.visualize_graph()

        print("\n=== Running Graph Analyses ===")

        # 1. Connectivity and Topological Sorting
        self.analyze_connectivity()
        self.topological_sort()

        # 2. Shortest Path
        self.shortest_paths()
        self.floyd_warshall()

        # 3. Minimum Spanning Tree
        self.minimum_spanning_tree()

        # 4. Eulerian and Hamiltonian Cycles
        self.eulerian_analysis()
        self.hamiltonian_analysis()

        # 5. Graph Matching
        self.maximum_matching()

        # 6. Maximum Flow
        self.maximum_flow()

        # 7. Graph Coloring
        self.graph_coloring()

        print("\n=== Analysis Complete ===")
        print("Results have been printed and visualizations saved to disk.")


# Example usage
if __name__ == "__main__":
    # Example with a file
    filename = "graph_data.txt"

    # Create a sample graph file if it doesn't exist
    if not os.path.exists(filename):
        print(f"Creating sample graph file '{filename}'...")
        with open(filename, 'w') as f:
            # Format: n m d w (n=vertices, m=edges, d=directed, w=weighted)
            f.write("6 8 0 1\n")  # 6 vertices, 8 edges, undirected, weighted
            # Format: u v [w] (u,v = vertices, w = optional weight)
            f.write("1 2 3\n")
            f.write("1 3 5\n")
            f.write("2 3 2\n")
            f.write("2 4 6\n")
            f.write("3 4 1\n")
            f.write("3 5 4\n")
            f.write("4 5 8\n")
            f.write("4 6 7\n")

    # Analyze the graph
    graph_analysis = GraphAnalysis(filename)
    graph_analysis.run_all_analyses()