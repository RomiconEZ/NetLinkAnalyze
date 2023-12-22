from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydantic import BaseModel


class Node(BaseModel):
    """
    Node class with a numerical value for comparison operations.
    """

    number: int  # Numeric value of the node

    def __lt__(self, other: "Node"):
        return self.number < other.number

    def __eq__(self, other: "Node"):
        return self.number == other.number


class Edge(BaseModel):
    """
    Edge class representing a connection between two nodes with a timestamp.

    Attributes:
        start_node (Node): The starting node of the edge.
        end_node (Node): The ending node of the edge.
        timestamp (int): The timestamp associated with the edge.
    """

    start_node: Node
    end_node: Node
    timestamp: int

    def __lt__(self, other: "Edge"):
        return max(self.end_node, self.start_node) < max(other.end_node, other.start_node)

    def __eq__(self, other: "Edge"):
        return (
            self.start_node == other.start_node
            and self.end_node == other.end_node
            and self.timestamp == other.timestamp
        )

    def get_max_node(self):
        return max(self.end_node, self.start_node)


class TemporalGraph:
    """
    A graph structure representing a temporal network of edges.

    The edges in this graph are time-stamped, allowing for the representation
    of dynamic relationships between nodes over time.

    Attributes:
        edge_list (list[Edge]): A list of edges in the graph.
    """

    edge_list: list[Edge]

    def __init__(self, path: str, n_skip_lines: int = 2):
        """
        Initializes a TemporalGraph by reading edge data from a file.

        The data file is expected to have edges in a specific format, where
        each line represents an edge with start node, end node, and timestamp.
        Self-loops (edges where start node equals end node) are ignored.

        Args:
            path (str): The file path to read the edge data from.
            n_skip_lines (int): The number of lines to skip at the beginning of the file.
        """
        self.edge_list = list()
        with open(path) as raw_data:
            # Skip the first two lines of the file (usually headers or metadata)
            for _ in range(n_skip_lines):
                raw_data.readline()

            list_of_items = raw_data.read().split("\n")
            list_of_items.pop(-1)  # Remove the last empty item if present
            for item in list_of_items:
                item = item.split(" ")
                if int(item[0]) == int(item[1]):  # Ignore self-loops
                    continue
                self.edge_list.append(
                    Edge(
                        start_node=Node(number=int(item[0])),
                        end_node=Node(number=int(item[1])),
                        timestamp=int(item[-1]),
                    )
                )

        self.edge_list.sort(key=lambda x: x.timestamp)

    def get_static_graph(self, left_border: float, right_border: float) -> "StaticGraph":
        """
        Creates a static subgraph from the temporal graph.

        This subgraph is a slice of the temporal graph, determined by the
        provided left and right indices as fractions of the total number
        of edges.

        Args:
            left_border (float): Left index as a fraction of total edges.
            right_border (float): Right index as a fraction of total edges.

        Returns:
            StaticGraph: A static graph containing the edges within the specified range.
        """
        left_index = int(left_border * len(self.edge_list))
        right_index = int(right_border * len(self.edge_list))
        sg = StaticGraph()
        for i in range(left_index, right_index):
            sg.add_edge(self.edge_list[i])
        return sg

    def get_max_timestamp(self):
        return max(self.edge_list, key=lambda x: x.timestamp).timestamp

    def get_min_timestamp(self):
        return min(self.edge_list, key=lambda x: x.timestamp).timestamp


@dataclass
class StaticGraph:
    """
    Class representing a static graph with nodes and edges.

    The graph is represented as an adjacency dictionary of dictionaries.
    Each node is a key in the outer dictionary, and its value is another
    dictionary containing adjacent nodes as keys and a list of timestamps
    as values.

    Attributes:
        num_of_edge (int): Number of edges in the graph.
        num_of_node (int): Number of nodes in the graph.
        adjacency_dict_of_dicts (dict): Adjacency dictionary of dictionaries.
        largest_connected_component (Optional[StaticGraph]): Largest connected component of the graph.
        number_of_connected_components (Optional[int]): Number of connected components in the graph.
    """

    num_of_edge: int = 0
    num_of_node: int = 0
    # Adjacency matrix
    adjacency_dict_of_dicts: dict[int, dict[int, [int]]] = None
    largest_connected_component: Optional["StaticGraph"] = None
    number_of_connected_components: Optional[int] = None

    def __init__(self):
        """Initialize the static graph with empty structures for nodes and edges."""
        self.adjacency_dict_of_dicts = dict()
        self.largest_connected_component = None
        self.number_of_connected_components = None

    def add_node(self, node: Node) -> int:
        """
        Add a new node to the graph.

        Args:
            node (Node): The node to be added.

        Returns:
            int: The updated number of nodes in the graph.
        """
        self.adjacency_dict_of_dicts[node.number] = dict()
        self.num_of_node += 1
        return self.num_of_node

    def add_edge(self, edge: Edge) -> int:
        """
        Add a new edge to the graph. If the nodes of the edge do not exist,
        they are added to the graph.

        Args:
            edge (Edge): The edge to be added.

        Returns:
            int: The updated number of edges in the graph.
        """
        start_node_number = edge.start_node.number
        end_node_number = edge.end_node.number
        # Add nodes if they don't exist
        if start_node_number not in self.adjacency_dict_of_dicts.keys():
            self.add_node(edge.start_node)
        if end_node_number not in self.adjacency_dict_of_dicts.keys():
            self.add_node(edge.end_node)
        # Ensure start_node_number is less than end_node_number
        if start_node_number > end_node_number:
            start_node_number, end_node_number = end_node_number, start_node_number
        # Add edge with timestamp, avoiding duplicates
        if end_node_number not in self.adjacency_dict_of_dicts[start_node_number].keys():
            self.adjacency_dict_of_dicts[start_node_number][end_node_number] = [edge.timestamp]
            self.adjacency_dict_of_dicts[end_node_number][start_node_number] = []
            self.num_of_edge += 1
        elif edge.timestamp not in self.adjacency_dict_of_dicts[start_node_number][end_node_number]:
            self.adjacency_dict_of_dicts[start_node_number][end_node_number] += [edge.timestamp]
            self.num_of_edge += 1
        return self.num_of_edge

    def count_vertices(self) -> int:
        """Return the number of vertices in the graph."""
        return self.num_of_node

    def get_adjacency_dict_of_dicts(self) -> dict:
        """Return the adjacency dictionary of dictionaries representing the graph."""
        return self.adjacency_dict_of_dicts

    def count_edges(self) -> int:
        """Return the number of edges in the graph."""
        return self.num_of_edge

    def density(self) -> float:
        """
        Calculate and return the density of the graph.

        Density is defined as the ratio of the number of edges to the maximum
        possible number of edges in a graph with the same number of vertices.
        """
        cnt_vert: int = self.count_vertices()
        if cnt_vert > 1:
            return self.count_edges() / (cnt_vert * (cnt_vert - 1))
        else:
            return 0.0

    def __find_size_of_connected_component(self, used: dict, start_vertice) -> int:
        """
        Traverse the connected component starting from 'start_vertice' and calculate its size.

        Args:
            used (dict): Dictionary to mark visited vertices.
            start_vertice: Starting vertex for the traversal.

        Returns:
            int: The size of the connected component.
        """
        queue = list()
        used[start_vertice] = True
        queue.append(start_vertice)

        size = 0

        while len(queue) > 0:
            v = queue.pop(0)
            size += 1
            for to in self.adjacency_dict_of_dicts[v].keys():
                if to not in used.keys():
                    used[to] = True
                    queue.append(to)

        return size

    def __find_largest_connected_component(self, used: dict, start_vertice):
        """
        Traverse and record the largest connected component as a separate graph.

        Args:
            used (dict): Dictionary to mark visited vertices.
            start_vertice: Starting vertex for the traversal.
        """
        queue = list()
        used[start_vertice] = True
        queue.append(start_vertice)

        while len(queue) > 0:
            v = queue.pop(0)
            for to in self.adjacency_dict_of_dicts[v].keys():
                if to not in used.keys():
                    used[to] = True
                    queue.append(to)

                min_node = min(v, to)
                max_node = max(v, to)
                edge_ts = self.adjacency_dict_of_dicts[min_node][max_node]
                for i in edge_ts:
                    start_node = Node(number=min_node)
                    end_node = Node(number=max_node)
                    timestamp = i
                    self.largest_connected_component.add_edge(
                        Edge(start_node=start_node, end_node=end_node, timestamp=timestamp)
                    )

    def __update_number_of_connected_components_and_largest_connected_component(self):
        """
        Run DFS from each unvisited vertex to find weakly connected components.
        Also calculates the number of these components and the largest component.

        Args:
            None.

        Returns:
            None.
        """
        used = dict()
        vertice: int = 0
        self.number_of_connected_components = 0
        max_component_size: int = 0
        for v in self.adjacency_dict_of_dicts.keys():
            if v not in used.keys():
                self.number_of_connected_components += 1
                component_size = self.__find_size_of_connected_component(used, v)
                if component_size > max_component_size:
                    max_component_size = component_size
                    vertice = v

        # Update vertex visits for processing the maximum power of weak connectivity components
        used.clear()

        # Found the maximum power component of weak connectivity, write it in the field
        self.largest_connected_component = StaticGraph()
        self.__find_largest_connected_component(used, vertice)

    @staticmethod
    def __floyd_warshall_algorithm(graph: "StaticGraph") -> dict[int, dict[int, int]]:
        """
        Implement the Floyd-Warshall algorithm to find the shortest paths in a graph.

        Args:
            graph (StaticGraph): The graph on which the algorithm is to be applied.

        Returns:
            dict[int, dict[int, int]]: A dictionary representing the shortest paths between nodes.
        """
        shortest_paths = dict()

        # Initialize shortest paths with direct connections
        for i in graph.adjacency_dict_of_dicts.keys():
            shortest_paths[i] = dict()
            for j in graph.adjacency_dict_of_dicts[i].keys():
                shortest_paths[i][j] = 1

        # Update shortest paths through intermediate vertices
        for k in shortest_paths.keys():
            for i in shortest_paths[k].keys():
                for j in shortest_paths[k].keys():
                    if j not in shortest_paths[i]:
                        shortest_paths[i][j] = shortest_paths[i][k] + shortest_paths[k][j]
                    else:
                        shortest_paths[i][j] = min(shortest_paths[i][j], shortest_paths[i][k] + shortest_paths[k][j])

        return shortest_paths

    def get_largest_connected_component(self) -> "StaticGraph":
        """
        If not already found, find the largest weakly connected component.

        Returns:
            StaticGraph: The largest weakly connected component.
        """
        if self.largest_connected_component is None:
            self.__update_number_of_connected_components_and_largest_connected_component()

        return self.largest_connected_component

    def get_number_of_connected_components(self) -> int:
        """
        If not already found, find the number of weakly connected components.

        Returns:
            int: Number of weakly connected components.
        """
        if self.largest_connected_component is None:
            self.__update_number_of_connected_components_and_largest_connected_component()

        return self.number_of_connected_components

    def share_of_vertices(self) -> float:
        """
        Calculate the proportion of vertices in the largest connected component.

        Returns:
            float: Proportion of vertices in the largest connected component.
        """
        return self.get_largest_connected_component().count_vertices() / self.count_vertices()

    def get_radius(self, graph: "StaticGraph") -> int:
        """
        Calculate the radius of the graph.

        The radius is the minimum eccentricity of any vertex in the graph.
        Eccentricity of a vertex is the greatest distance between that vertex and any other vertex in the graph.

        Args:
            graph (StaticGraph): The graph for which the radius is to be calculated.

        Returns:
            int: The radius of the graph.
        """
        sample_graph: StaticGraph = graph

        # Floyd-Warshell algorithm
        shortest_paths = self.__floyd_warshall_algorithm(sample_graph)

        radius = np.inf
        for i in shortest_paths.keys():
            eccentricity = 0
            for j in shortest_paths[i].keys():
                eccentricity = max(eccentricity, shortest_paths[i][j])
            if eccentricity > 0:
                radius = min(radius, eccentricity)
        return radius

    def get_diameter(self, graph: "StaticGraph") -> int:
        """
        Calculate the diameter of the graph.

        The diameter is the greatest distance between any pair of vertices in the graph.

        Args:
            graph (StaticGraph): The graph for which the diameter is to be calculated.

        Returns:
            int: The diameter of the graph.
        """
        sample_graph: StaticGraph = graph

        # Floyd-Warshell algorithm
        shortest_paths = self.__floyd_warshall_algorithm(sample_graph)

        diameter = 0
        for i in shortest_paths.keys():
            for j in shortest_paths[i].keys():
                diameter = max(diameter, shortest_paths[i][j])
        return diameter

    def percentile_distance(self, graph: "StaticGraph", percentile: int = 90) -> int:
        """
        Calculate a specific percentile of the distance distribution in the graph.

        Args:
            graph (StaticGraph): The graph for which the distances are calculated.
            percentile (int): The percentile to calculate (between 0 and 100).

        Returns:
            int: The calculated percentile distance.
        """
        sample_graph: StaticGraph = graph
        shortest_paths = self.__floyd_warshall_algorithm(sample_graph)
        dists = []
        for i in shortest_paths.keys():
            for j in shortest_paths[i].keys():
                dists.append(shortest_paths[i][j])
        dists.sort()
        return dists[int(percentile / 100 * (len(dists) - 1))]

    def average_cluster_factor(self) -> float:
        """
        Calculate the average clustering coefficient for the largest connected component in the graph.

        The clustering coefficient for a vertex quantifies
        how close its neighbors are to being a complete graph (clique).

        Returns:
            float: The average clustering coefficient for the largest connected component.
        """
        cnt_verts = self.get_largest_connected_component().count_vertices()
        result = 0
        for i in self.get_largest_connected_component().adjacency_dict_of_dicts.keys():
            i_degree = len(self.get_largest_connected_component().adjacency_dict_of_dicts[i])
            if i_degree < 2:
                continue
            l_u = 0
            for j in self.get_largest_connected_component().adjacency_dict_of_dicts[i].keys():
                for k in self.get_largest_connected_component().adjacency_dict_of_dicts[i].keys():
                    if k in self.get_largest_connected_component().adjacency_dict_of_dicts[j]:
                        l_u += 1

            result += l_u / (i_degree * (i_degree - 1))
        return result / cnt_verts

    def assortative_factor(self) -> float:
        """
        Calculate the assortativity coefficient of the graph.

        Assortativity measures the similarity of connections in the graph with respect to the node degree.
        It indicates whether high-degree nodes tend to connect with other high-degree nodes (assortative mixing)
        or low-degree nodes (disassortative mixing).
        A positive assortativity coefficient indicates a preference
        for high-degree nodes to attach to other high-degree nodes,
        while a negative coefficient indicates the opposite.

        Returns:
            float: The assortativity coefficient of the graph.
        """
        re = 0
        r1 = 0
        r2 = 0
        r3 = 0
        for u in self.get_largest_connected_component().adjacency_dict_of_dicts.keys():
            u_degree = len(self.get_largest_connected_component().adjacency_dict_of_dicts[u])
            r1 += u_degree
            r2 += u_degree**2
            r3 += u_degree**3
            for v in self.get_largest_connected_component().adjacency_dict_of_dicts[u].keys():
                v_degree = len(self.get_largest_connected_component().adjacency_dict_of_dicts[v])
                re += u_degree * v_degree

        return (re * r1 - (r2 * r2)) / (r3 * r1 - (r2 * r2))


@dataclass
class SelectApproach:
    """
    Class to select a subgraph from a given graph using different sampling approaches.

    Attributes:
        start_node1_number (Optional[int]): The starting node number for the snowball sampling method.
        start_node2_number (Optional[int]): An additional starting node number for the snowball sampling method.
    """

    start_node1_number: Optional[int]
    start_node2_number: Optional[int]

    def __init__(self, s_node1_number: int = None, s_node2_number: int = None):
        """
        Initialize the selection approach.

        If both starting nodes are provided and their order is reversed, they are swapped to maintain consistency.

        Args:
            s_node1_number (int, optional): The first starting node number for snowball sampling.
            s_node2_number (int, optional): The second starting node number for snowball sampling.
        """
        if s_node1_number is not None and s_node2_number is not None:
            if s_node1_number > s_node2_number:
                s_node1_number, s_node2_number = s_node2_number, s_node1_number
        self.start_node1_number = s_node1_number
        self.start_node2_number = s_node2_number

    def snowball_sample(self, graph: StaticGraph) -> StaticGraph:
        """
        Perform snowball sampling on the graph.

        Starting from one or two nodes, it expands to include neighbors of these nodes, up to a specified limit.

        Args:
            graph (StaticGraph): The original graph from which a subgraph is to be sampled.

        Returns:
            StaticGraph: The sampled subgraph.
        """
        queue = list()
        start_node1_number = self.start_node1_number
        start_node2_number = self.start_node2_number

        # Add two vertices to the queue for BFS
        queue.append(start_node1_number)
        queue.append(start_node2_number)
        cnt_verts = graph.count_vertices()

        size = min(500, cnt_verts)
        # A new graph that should result
        sample_graph = StaticGraph()

        used = dict()
        used[start_node1_number] = True
        used[start_node2_number] = True

        size -= 2

        while len(queue) > 0:  # BFS
            v = queue.pop(0)

            for i in graph.adjacency_dict_of_dicts[v]:
                if i not in used.keys():
                    if size > 0:
                        used[i] = True
                        size -= 1
                        queue.append(i)
                    else:
                        continue

                min_node = min(v, i)
                max_node = max(v, i)

                edge_ts = graph.adjacency_dict_of_dicts[min_node][max_node]
                for j in edge_ts:
                    start_node = Node(number=min_node)
                    end_node = Node(number=max_node)
                    timestamp = j
                    sample_graph.add_edge(Edge(start_node=start_node, end_node=end_node, timestamp=timestamp))

        return sample_graph

    @staticmethod
    def random_selected_vertices(graph: StaticGraph) -> StaticGraph:
        """
        Perform random vertex sampling on the graph.

        Randomly selects a specified number of vertices and their associated edges to create a subgraph.

        Args:
            graph (StaticGraph): The original graph from which a subgraph is to be sampled.

        Returns:
            StaticGraph: The sampled subgraph.
        """
        # Set of remaining vertices
        remaining_vertices = list(graph.adjacency_dict_of_dicts.keys())
        size = min(500, graph.count_vertices())
        # A new graph that should result
        sample_graph = StaticGraph()

        for _ in range(size):
            # Select a new vertex to add to the graph
            new_vertice = remaining_vertices[np.random.randint(0, len(remaining_vertices))]

            remaining_vertices.remove(new_vertice)
            sample_graph.add_node(Node(number=new_vertice))
            for vertice in sample_graph.adjacency_dict_of_dicts.keys():
                # If the vertices are adjacent in the original graph, then add the edges
                if new_vertice in graph.adjacency_dict_of_dicts[vertice].keys():
                    min_node = min(vertice, new_vertice)
                    max_node = max(vertice, new_vertice)

                    edge_ts = graph.adjacency_dict_of_dicts[min_node][max_node]
                    for i in edge_ts:
                        start_node = Node(number=min_node)
                        end_node = Node(number=max_node)
                        timestamp = i
                        sample_graph.add_edge(Edge(start_node=start_node, end_node=end_node, timestamp=timestamp))

        return sample_graph

    def __call__(self, graph: StaticGraph):
        """
        Execute the selected sampling method on the graph.

        Chooses between snowball sampling and random vertex sampling based on the provided starting nodes.

        Args:
            graph (StaticGraph): The original graph from which a subgraph is to be sampled.

        Returns:
            StaticGraph: The sampled subgraph.
        """
        if self.start_node1_number is None:
            return self.random_selected_vertices(graph)
        return self.snowball_sample(graph)
