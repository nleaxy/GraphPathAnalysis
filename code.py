import sys
import heapq
import timeit
import networkx as nx
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QInputDialog, QLabel, QComboBox, QMessageBox, QTableWidget,
                             QTableWidgetItem, QGroupBox, QSizePolicy)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import deque

class GraphWidget(QWidget):
    """
    Main widget for graph path analysis application.
    
    This application provides a GUI for creating directed graphs and finding
    shortest paths using various algorithms including Dijkstra, Bellman-Ford,
    Floyd-Warshall, Johnson, Levit, and Yen's algorithms.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Path Analysis Tool")
        self.setGeometry(100, 100, 1200, 900)

        # Main layout for the entire application
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Initialize the directed graph using NetworkX
        self.graph = nx.DiGraph()

        # Create all UI components
        self._create_graph_controls()
        self._create_algorithm_controls()
        self._create_visualization_group()
        self._create_results_group()
        self._create_complexity_group()

        # Add all components to main layout
        main_layout.addWidget(self.graph_controls_group)
        main_layout.addWidget(self.algorithm_controls_group)
        main_layout.addWidget(self.visualization_group)
        main_layout.addWidget(self.results_group)
        main_layout.addWidget(self.complexity_group)

        # Initialize matplotlib axes for graph visualization
        self.ax = self.canvas.figure.subplots()

    def _create_graph_controls(self):
        """Create the graph manipulation control panel."""
        self.graph_controls_group = QGroupBox("Graph Management")
        layout = QHBoxLayout()

        # Button to add new vertices to the graph
        self.add_vertex_button = QPushButton("➕ Add Vertex")
        self.add_vertex_button.clicked.connect(self.add_vertex)
        
        # Button to add edges between vertices
        self.add_edge_button = QPushButton("↔️ Add Edge")
        self.add_edge_button.clicked.connect(self.add_edge)
        
        # Button to visualize the current graph
        self.show_graph_button = QPushButton("🖼️ Show Graph")
        self.show_graph_button.clicked.connect(self.show_graph)

        layout.addWidget(self.add_vertex_button)
        layout.addWidget(self.add_edge_button)
        layout.addWidget(self.show_graph_button)
        self.graph_controls_group.setLayout(layout)

    def _create_algorithm_controls(self):
        """Create the algorithm selection and execution control panel."""
        self.algorithm_controls_group = QGroupBox("Algorithm Management")
        layout = QHBoxLayout()

        # Dropdown menu for algorithm selection
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "Dijkstra",
            "Bellman-Ford",
            "Floyd-Warshall",
            "Johnson",
            "Levit",
            "Yen"
        ])
        
        # Button to execute the selected algorithm
        self.run_algorithm_button = QPushButton("⚡ Find Path")
        self.run_algorithm_button.clicked.connect(self.run_algorithm)
        
        # Button to analyze complexity of all algorithms
        self.analyze_button = QPushButton("📊 Complexity Analysis")
        self.analyze_button.clicked.connect(self.analyze_complexity)

        layout.addWidget(self.algorithm_combo, 3)
        layout.addWidget(self.run_algorithm_button, 1)
        layout.addWidget(self.analyze_button, 1)
        self.algorithm_controls_group.setLayout(layout)

    def _create_visualization_group(self):
        """Create the graph visualization panel."""
        self.visualization_group = QGroupBox("Graph Visualization")
        layout = QVBoxLayout()
        
        # Label showing current graph state
        self.graph_label = QLabel("Current graph state:")
        
        # Matplotlib canvas for graph visualization
        self.canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout.addWidget(self.graph_label)
        layout.addWidget(self.canvas)
        self.visualization_group.setLayout(layout)

    def _create_results_group(self):
        """Create the results display panel for path finding results."""
        self.results_group = QGroupBox("Path Finding Results")
        layout = QVBoxLayout()
        
        results_sub = QHBoxLayout()
        
        # Labels to display shortest distance and path
        self.distance_label = QLabel("Shortest distance: ")
        self.path_label = QLabel("Path: ")
        
        results_sub.addWidget(self.distance_label)
        results_sub.addWidget(self.path_label)
        
        layout.addLayout(results_sub)
        self.results_group.setLayout(layout)

    def _create_complexity_group(self):
        """Create the performance analysis panel."""
        self.complexity_group = QGroupBox("Performance Analysis")
        layout = QVBoxLayout()
        
        # Table to display algorithm performance comparison
        self.complexity_table = QTableWidget()
        self.complexity_table.setColumnCount(4)
        self.complexity_table.setHorizontalHeaderLabels(
            ["Algorithm", "Time (ms)", "Theoretical Complexity", "Nodes/Edges"])
        
        layout.addWidget(self.complexity_table)
        self.complexity_group.setLayout(layout)

    def add_vertex(self):
        """Add a new vertex to the graph through user input dialog."""
        vertex, ok = QInputDialog.getText(self, 'Add Vertex', 'Enter vertex name:')
        if ok and vertex:
            self.graph.add_node(vertex)
            self.update_graph_display()

    def add_edge(self):
        """Add a weighted edge between two vertices through user input dialogs."""
        if len(self.graph.nodes()) < 2:
            QMessageBox.warning(self, "Error", "Need at least 2 vertices")
            return

        vertices = list(self.graph.nodes())
        
        # Get source vertex
        from_vertex, ok1 = QInputDialog.getItem(self, 'Add Edge', 'From vertex:', vertices, 0, False)
        # Get destination vertex
        to_vertex, ok2 = QInputDialog.getItem(self, 'Add Edge', 'To vertex:', vertices, 0, False)

        if ok1 and ok2 and from_vertex and to_vertex:
            # Get edge weight
            weight, ok3 = QInputDialog.getDouble(self, 'Add Edge', 'Edge weight:', 1.0, -1000.0, 1000.0, 2)
            if ok3:
                self.graph.add_edge(from_vertex, to_vertex, weight=weight)
                self.update_graph_display()

    def show_graph(self):
        """Visualize the current graph using matplotlib."""
        self.ax.clear()
        
        # Generate node positions using spring layout
        pos = nx.spring_layout(self.graph)
        
        # Draw the graph with nodes, edges, and labels
        nx.draw(self.graph, pos, ax=self.ax, with_labels=True, node_color='lightblue',
                node_size=500, font_size=10, arrows=True)
        
        # Add edge weight labels
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, ax=self.ax)
        
        # Refresh the canvas
        self.canvas.draw()

    def update_graph_display(self):
        """Update the text display showing current graph state."""
        self.graph_label.setText(f"Vertices: {list(self.graph.nodes())}\nEdges: {list(self.graph.edges(data='weight'))}")

    def dijkstra(self, start, end):
        """
        Implement Dijkstra's algorithm for finding shortest path.
        
        Time Complexity: O((V + E) log V)
        Space Complexity: O(V)
        
        Args:
            start: Starting vertex
            end: Destination vertex
            
        Returns:
            tuple: (shortest_distance, shortest_path)
        """
        if start not in self.graph or end not in self.graph:
            raise ValueError("Vertices do not exist in graph")

        # Priority queue for processing vertices
        heap = []
        heapq.heappush(heap, (0, start))
        
        # Initialize distances and previous vertex tracking
        distances = {node: float('inf') for node in self.graph.nodes()}
        distances[start] = 0
        previous = {node: None for node in self.graph.nodes()}

        while heap:
            current_dist, current_node = heapq.heappop(heap)

            # Early termination when destination is reached
            if current_node == end:
                break

            # Skip if we've already found a better path
            if current_dist > distances[current_node]:
                continue

            # Relax neighboring edges
            for neighbor, data in self.graph[current_node].items():
                weight = data['weight']
                distance = current_dist + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(heap, (distance, neighbor))

        if distances[end] == float('inf'):
            raise ValueError("Path does not exist")

        # Reconstruct path from destination to source
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        return distances[end], path

    def bellman_ford(self, start, end):
        """
        Implement Bellman-Ford algorithm for finding shortest path.
        Can handle negative edge weights and detect negative cycles.
        
        Time Complexity: O(V * E)
        Space Complexity: O(V)
        
        Args:
            start: Starting vertex
            end: Destination vertex
            
        Returns:
            tuple: (shortest_distance, shortest_path)
        """
        if start not in self.graph or end not in self.graph:
            raise ValueError("Vertices do not exist in graph")

        # Initialize distances and previous vertex tracking
        distances = {node: float('inf') for node in self.graph.nodes()}
        distances[start] = 0
        previous = {node: None for node in self.graph.nodes()}

        # Relax edges V-1 times
        for _ in range(len(self.graph.nodes()) - 1):
            for u, v, data in self.graph.edges(data=True):
                weight = data['weight']
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    previous[v] = u

        # Check for negative cycles
        for u, v, data in self.graph.edges(data=True):
            if distances[u] + data['weight'] < distances[v]:
                raise ValueError("Graph contains negative weight cycle")

        if distances[end] == float('inf'):
            raise ValueError("Path does not exist")

        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        return distances[end], path

    def floyd_warshall(self, start, end):
        """
        Implement Floyd-Warshall algorithm for all-pairs shortest paths.
        
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        
        Args:
            start: Starting vertex
            end: Destination vertex
            
        Returns:
            tuple: (shortest_distance, shortest_path)
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        node_index = {node: i for i, node in enumerate(nodes)}

        # Initialize distance matrix
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0

        # Initialize next vertex matrix for path reconstruction
        next_node = [[None] * n for _ in range(n)]

        # Fill initial distances from edges
        for u, v, data in self.graph.edges(data=True):
            i, j = node_index[u], node_index[v]
            dist[i][j] = data['weight']
            next_node[i][j] = v

        # Floyd-Warshall main algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]

        start_idx, end_idx = node_index[start], node_index[end]
        if dist[start_idx][end_idx] == float('inf'):
            raise ValueError("Path does not exist")

        # Reconstruct path
        path = [start]
        current = start
        while current != end:
            current = next_node[node_index[current]][end_idx]
            if current is None:
                raise ValueError("Path does not exist")
            path.append(current)

        return dist[start_idx][end_idx], path

    def johnson(self, start, end):
        """
        Implement Johnson's algorithm for all-pairs shortest paths.
        Uses Bellman-Ford to detect negative cycles and reweight edges,
        then applies Dijkstra's algorithm.
        
        Time Complexity: O(V^2 log V + V*E)
        Space Complexity: O(V^2)
        
        Args:
            start: Starting vertex
            end: Destination vertex
            
        Returns:
            tuple: (shortest_distance, shortest_path)
        """
        # Create temporary graph with additional source vertex
        temp_graph = self.graph.copy()
        new_node = "temp_johnson_node"
        while new_node in temp_graph.nodes():
            new_node += "_"

        temp_graph.add_node(new_node)
        # Add edges from new source to all vertices with weight 0
        for node in self.graph.nodes():
            temp_graph.add_edge(new_node, node, weight=0)

        # Run Bellman-Ford from the new source to get reweighting function
        h = {node: 0 for node in temp_graph.nodes()}
        for _ in range(len(temp_graph.nodes()) - 1):
            updated = False
            for u, v, data in temp_graph.edges(data=True):
                if h[v] > h[u] + data['weight']:
                    h[v] = h[u] + data['weight']
                    updated = True
            if not updated:
                break

        # Check for negative cycles
        for u, v, data in temp_graph.edges(data=True):
            if h[v] > h[u] + data['weight']:
                raise ValueError("Graph contains negative weight cycle")

        # Remove temporary node
        temp_graph.remove_node(new_node)

        # Create reweighted graph
        reweighted_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            new_weight = data['weight'] + h[u] - h[v]
            reweighted_graph.add_edge(u, v, weight=new_weight)

        # Run Dijkstra on reweighted graph
        distances = {}
        previous = {}

        heap = []
        heapq.heappush(heap, (0, start))
        distances[start] = 0
        previous[start] = None

        while heap:
            current_dist, current_node = heapq.heappop(heap)

            if current_node == end:
                break

            if current_dist > distances.get(current_node, float('inf')):
                continue

            for neighbor, data in reweighted_graph[current_node].items():
                weight = data['weight']
                distance = current_dist + weight

                if distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(heap, (distance, neighbor))

        if end not in distances:
            raise ValueError("Path does not exist")

        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous.get(current)
        path.reverse()

        # Convert back to original weights
        original_distance = distances[end] + h[end] - h[start]

        return original_distance, path

    def levit(self, start, end):
        """
        Implement Levit's algorithm (SLF - Shortest Label First).
        Modification of Bellman-Ford with three sets optimization.
        
        Time Complexity: O(V * E) average case
        Space Complexity: O(V)
        
        Args:
            start: Starting vertex
            end: Destination vertex
            
        Returns:
            tuple: (shortest_distance, shortest_path)
        """
        if start not in self.graph or end not in self.graph:
            raise ValueError("Vertices do not exist in graph")

        INF = float('inf')
        distances = {node: INF for node in self.graph.nodes()}
        distances[start] = 0
        previous = {node: None for node in self.graph.nodes()}

        # Three sets for Levit's algorithm
        M0 = set()  # Vertices already processed
        M1 = deque([start])  # Vertices to be processed
        M2 = deque()  # Vertices that need reprocessing

        while M1 or M2:
            current = M1.popleft() if M1 else M2.popleft()

            if current == end:
                break

            M0.add(current)

            # Relax all outgoing edges
            for neighbor, data in self.graph[current].items():
                weight = data['weight']
                new_dist = distances[current] + weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current

                    # Move neighbor to appropriate set
                    if neighbor in M0:
                        M2.appendleft(neighbor)
                    elif neighbor in M1:
                        pass  # Already in M1
                    else:
                        M1.append(neighbor)

        if distances[end] == INF:
            raise ValueError("Path does not exist")

        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        return distances[end], path

    def yen(self, start, end, K=3):
        """
        Implement Yen's algorithm for K-shortest paths.
        Currently returns only the shortest path (K=1).
        
        Time Complexity: O(K*V*(E + V log V))
        Space Complexity: O(V)
        
        Args:
            start: Starting vertex
            end: Destination vertex
            K: Number of shortest paths to find (not fully implemented)
            
        Returns:
            tuple: (shortest_distance, shortest_path)
        """
        if start not in self.graph or end not in self.graph:
            raise ValueError("Vertices do not exist in graph")

        # For now, just return the shortest path using Dijkstra
        distance, path = self.dijkstra(start, end)
        return distance, path

    def run_algorithm(self):
        """Execute the selected algorithm on user-specified start and end vertices."""
        if len(self.graph.nodes()) < 2:
            QMessageBox.warning(self, "Error", "Need at least 2 vertices")
            return

        vertices = list(self.graph.nodes())
        
        # Get start and end vertices from user
        start_vertex, ok1 = QInputDialog.getItem(self, 'Start Vertex', 'Select start vertex:', vertices, 0, False)
        end_vertex, ok2 = QInputDialog.getItem(self, 'End Vertex', 'Select end vertex:', vertices, 0, False)

        if not (ok1 and ok2):
            return

        algorithm = self.algorithm_combo.currentText()

        try:
            # Measure execution time
            start_time = timeit.default_timer()

            # Execute selected algorithm
            if algorithm == "Dijkstra":
                distance, path = self.dijkstra(start_vertex, end_vertex)
                complexity = "O((V + E) log V)"
            elif algorithm == "Bellman-Ford":
                distance, path = self.bellman_ford(start_vertex, end_vertex)
                complexity = "O(V * E)"
            elif algorithm == "Floyd-Warshall":
                distance, path = self.floyd_warshall(start_vertex, end_vertex)
                complexity = "O(V^3)"
            elif algorithm == "Johnson":
                distance, path = self.johnson(start_vertex, end_vertex)
                complexity = "O(V^2 log V + V*E)"
            elif algorithm == "Levit":
                distance, path = self.levit(start_vertex, end_vertex)
                complexity = "O(V * E)"
            elif algorithm == "Yen":
                distance, path = self.yen(start_vertex, end_vertex)
                complexity = "O(K*V*(E + V log V))"
            else:
                raise ValueError("Unknown algorithm")

            exec_time = (timeit.default_timer() - start_time) * 1000

            # Update results display
            self.distance_label.setText(f"Shortest distance: {distance}")
            self.path_label.setText(f"Path: {' -> '.join(path)}")
            self.update_complexity_table(algorithm, exec_time, complexity)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def analyze_complexity(self):
        """Run all algorithms and compare their performance."""
        if len(self.graph.nodes()) < 2:
            QMessageBox.warning(self, "Error", "Need at least 2 vertices")
            return

        vertices = list(self.graph.nodes())
        start_vertex = vertices[0]
        end_vertex = vertices[-1]

        results = []

        # Test all algorithms
        for algorithm in ["Dijkstra", "Bellman-Ford", "Floyd-Warshall", "Johnson", "Levit", "Yen"]:
            try:
                start_time = timeit.default_timer()

                # Execute algorithm
                if algorithm == "Dijkstra":
                    self.dijkstra(start_vertex, end_vertex)
                    complexity = "O((V + E) log V)"
                elif algorithm == "Bellman-Ford":
                    self.bellman_ford(start_vertex, end_vertex)
                    complexity = "O(V * E)"
                elif algorithm == "Floyd-Warshall":
                    self.floyd_warshall(start_vertex, end_vertex)
                    complexity = "O(V^3)"
                elif algorithm == "Johnson":
                    self.johnson(start_vertex, end_vertex)
                    complexity = "O(V^2 log V + V*E)"
                elif algorithm == "Levit":
                    self.levit(start_vertex, end_vertex)
                    complexity = "O(V * E)"
                elif algorithm == "Yen":
                    self.yen(start_vertex, end_vertex)
                    complexity = "O(K*V*(E + V log V))"

                exec_time = (timeit.default_timer() - start_time) * 1000

                results.append({
                    "Algorithm": algorithm,
                    "Time (ms)": f"{exec_time:.4f}",
                    "Theoretical Complexity": complexity,
                    "Nodes/Edges": f"{len(self.graph.nodes())}/{len(self.graph.edges())}"
                })

            except Exception as e:
                results.append({
                    "Algorithm": algorithm,
                    "Time (ms)": "Error",
                    "Theoretical Complexity": complexity,
                    "Nodes/Edges": f"{len(self.graph.nodes())}/{len(self.graph.edges())}"
                })

        self.show_complexity_results(results)

    def update_complexity_table(self, algorithm, exec_time, complexity):
        """Add a single algorithm result to the complexity table."""
        row_count = self.complexity_table.rowCount()
        self.complexity_table.insertRow(row_count)

        self.complexity_table.setItem(row_count, 0, QTableWidgetItem(algorithm))
        self.complexity_table.setItem(row_count, 1, QTableWidgetItem(f"{exec_time:.4f}"))
        self.complexity_table.setItem(row_count, 2, QTableWidgetItem(complexity))
        self.complexity_table.setItem(row_count, 3,
                                    QTableWidgetItem(f"{len(self.graph.nodes())}/{len(self.graph.edges())}"))

    def show_complexity_results(self, results):
        """Display all algorithm performance results in the table."""
        self.complexity_table.setRowCount(0)

        for i, result in enumerate(results):
            self.complexity_table.insertRow(i)
            self.complexity_table.setItem(i, 0, QTableWidgetItem(result["Algorithm"]))
            self.complexity_table.setItem(i, 1, QTableWidgetItem(result["Time (ms)"]))
            self.complexity_table.setItem(i, 2, QTableWidgetItem(result["Theoretical Complexity"]))
            self.complexity_table.setItem(i, 3, QTableWidgetItem(result["Nodes/Edges"]))

if __name__ == "__main__":
    # Create and run the application
    app = QApplication(sys.argv)
    widget = GraphWidget()
    widget.show()
    sys.exit(app.exec_())
