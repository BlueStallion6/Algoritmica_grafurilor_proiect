import os
import sys

try:
    from graph_generator import generate_random_graph, generate_example_graphs
except ImportError:
    print("Error: Could not import from graph_generator.py")
    print("Make sure graph_generator.py is in the same directory as this script.")
    sys.exit(1)

try:
    from graph_analysis import GraphAnalysis
except ImportError:
    print("Error: Could not import from graph_analysis.py")
    print("Make sure graph_analysis.py is in the same directory as this script.")
    sys.exit(1)


def create_sample_graph_file(filename, graph_type="simple"):
    """Create a sample graph file based on the specified type"""
    if graph_type == "simple":
        # Simple undirected, unweighted graph
        generate_random_graph(filename, 6, 8, directed=False, weighted=False)
    elif graph_type == "weighted":
        # Weighted graph
        generate_random_graph(filename, 6, 8, directed=False, weighted=True)
    elif graph_type == "directed":
        # Directed graph
        generate_random_graph(filename, 6, 10, directed=True, weighted=False)
    elif graph_type == "flow":
        # Flow network
        generate_random_graph(filename, 6, 10, directed=True, weighted=True)
    elif graph_type == "all":
        # Generate all example graphs
        generate_example_graphs()
        return "Example graphs generated: simple_graph.txt, weighted_graph.txt, directed_graph.txt, flow_network.txt, bipartite_graph.txt, complete_graph.txt, tree_graph.txt, eulerian_graph.txt"
    else:
        return f"Unknown graph type: {graph_type}"

    return f"Sample graph file created: {filename}"


def main():
    """Main function to run the graph analysis program"""
    try:
        # Check if a filename was provided as a command-line argument
        if len(sys.argv) > 1:
            filename = sys.argv[1]

            # Check if the file exists
            if not os.path.exists(filename):
                print(f"Error: File '{filename}' not found.")
                return
        else:
            # No filename provided, ask the user what to do
            print("Graph Analysis Application")
            print("==========================")
            print("1. Analyze an existing graph file")
            print("2. Create a sample graph file")
            print("3. Generate example graphs")
            print("4. Create a simple test graph")
            print("5. Exit")

            choice = input("Enter your choice (1-5): ")

            if choice == "1":
                filename = input("Enter the path to the graph file: ")
                if not os.path.exists(filename):
                    print(f"Error: File '{filename}' not found.")
                    return
            elif choice == "2":
                filename = input("Enter the name for the new graph file: ")
                graph_type = input("Enter graph type (simple, weighted, directed, flow): ")
                result = create_sample_graph_file(filename, graph_type)
                print(result)

                # Check if the file was created successfully before analyzing
                if os.path.exists(filename):
                    continue_analysis = input("Would you like to analyze this graph now? (y/n): ")
                    if continue_analysis.lower() != 'y':
                        return
                else:
                    print(f"Error: Could not create file '{filename}'")
                    return
            elif choice == "3":
                result = create_sample_graph_file("", "all")
                print(result)

                # Ask which file to analyze
                filename = input("Enter the name of the file to analyze (e.g., simple_graph.txt): ")
                if not os.path.exists(filename):
                    print(f"Error: File '{filename}' not found.")
                    return
            elif choice == "4":
                # Create a very simple test graph
                filename = "test_graph.txt"
                with open(filename, 'w') as f:
                    f.write("5 7 0 1\n")  # 5 vertices, 7 edges, undirected, weighted
                    f.write("1 2 3\n")
                    f.write("1 3 4\n")
                    f.write("1 4 2\n")
                    f.write("2 3 5\n")
                    f.write("2 5 1\n")
                    f.write("3 4 6\n")
                    f.write("4 5 7\n")
                print(f"Created test graph file '{filename}'")

                continue_analysis = input("Would you like to analyze this graph now? (y/n): ")
                if continue_analysis.lower() != 'y':
                    return
            elif choice == "5":
                print("Exiting program.")
                return
            else:
                print("Invalid choice.")
                return

        # Analyze the graph
        print(f"Analyzing graph from file: '{filename}'")
        graph_analysis = GraphAnalysis(filename)

        if graph_analysis.graph is None:
            print("Error: Could not load graph. Analysis aborted.")
            return

        graph_analysis.run_all_analyses()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()