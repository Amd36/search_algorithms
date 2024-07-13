import heapq
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict

class Queue:
    def __init__(self):
        self.elements = deque()
        self.size = -1

    def __iter__(self):
        return iter(self.elements)

    def enqueue(self, element):
        self.elements.append(element)
        self.size += 1

    def dequeue(self):
        if not self.elements:
            print("The queue is empty")
        else:
            return self.elements.popleft()
        
    def top(self):
        if not self.elements:
            print("The queue is empty")
        else:
            return self.elements[0]
    
    def show(self):
        print(self.elements)

class MinHeap(Queue):  # Inherits from Queue for enqueue and dequeue operations
    def heapify_up(self, index):
        parent_index = (index - 1) // 2

        # Compares the cost of child with its parent, if less than swaps; loops until root node is reached
        while index > 0 and self.elements[index][1] < self.elements[parent_index][1]:
            # Swap elements if the current element is greater than its parent
            self.elements[index], self.elements[parent_index] = self.elements[parent_index], self.elements[index]
            index = parent_index
            parent_index = (index - 1) // 2

    def heapify_down(self, index):
        n = len(self.elements)
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        smallest = index

        # Compares the cost of the parent with its child, if less then swaps
        if left_child_index < n and self.elements[left_child_index][1] < self.elements[smallest][1]:
            smallest = left_child_index

        if right_child_index < n and self.elements[right_child_index][1] < self.elements[smallest][1]:
            smallest = right_child_index

        if smallest != index:
            # Swap elements if the current element is smaller than its children
            self.elements[index], self.elements[smallest] = self.elements[smallest], self.elements[index]
            self.heapify_down(smallest) 

    def insert(self, value):
        # Insert value at the end of the heap
        self.enqueue(value)
        # Perform heapify-up to maintain heap property
        self.heapify_up(len(self.elements) - 1)

    def extract_min(self):
        if not self.elements:
            print("Heap is empty")
            return None
        min_value = self.elements[0]
        # Replace root with the last element
        self.elements[0] = self.elements[-1]
        self.dequeue()  # Remove last element
        # Perform heapify-down to maintain heap property
        self.heapify_down(0)
        return min_value

# Return the path from the parent dictionary
def return_path(dict, source, goal):
    if goal == source:
        return [goal]
    else:
        path = return_path(dict, source, dict[goal])
        path.append(goal)
        return path

def UCS_path(graph, source, goal):
    heap = MinHeap()
    cost = defaultdict(list)
    parent = defaultdict(list)
    visited = []
    for i in graph.nodes:
        parent[i] = None 
        cost[i] = float('inf')  # Initialize the cost of each node to infinity
    heap.insert((source, 0))

    while heap:
        node = heap.extract_min()

        if node[0] == goal:
            return (return_path(parent, source, goal), cost[goal])

        if node[0] not in visited:
            visited.append(node[0])

            for neighbor in graph.neighbors(node[0]):
                g = graph[node[0]][neighbor]['weight'] + node[1]

                if g <= cost[neighbor]:
                    parent[neighbor] = node[0]
                    cost[neighbor] = g

                heap.insert((neighbor, g))

def UCS_traversal(graph, source, goal):
    heap = MinHeap()
    cost = defaultdict(list)
    traversal = []
    visited = []
    for i in graph.nodes:
        cost[i] = float('inf')  # Initialize the cost of each node to infinity
    heap.insert((source, 0))

    while heap:
        node = heap.extract_min()
        traversal.append(node[0])

        if node[0] == goal:
            return traversal

        if node[0] not in visited:
            visited.append(node[0])

            for neighbor in graph.neighbors(node[0]):
                g = graph[node[0]][neighbor]['weight'] + node[1]

                if g <= cost[neighbor]:
                    cost[neighbor] = g

                heap.insert((neighbor, g))

if __name__ == "__main__":
    # Define nodes and edges for the graph
    nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'z']
    edges = [
        ('a', 'b', 4),
        ('a', 'c', 3),
        ('b', 'a', 4),
        ('b', 'e', 12),
        ('b', 'f', 5),
        ('c', 'a', 3),
        ('c', 'd', 7),
        ('c', 'e', 10),
        ('d', 'c', 7),
        ('d', 'e', 2),
        ('e', 'b', 12),
        ('e', 'c', 10),
        ('e', 'd', 2),
        ('e', 'z', 5),
        ('f', 'b', 5),
        ('f', 'z', 16)
    ]

    # Create a networkx graph
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)

    # Run UCS to find the least-cost path from 'a' to 'z'
    path, cost = UCS_path(G, 'a', 'z')

    # Print the result
    print(f"Path: {path}")
    print(f"Cost: {cost}")

    # Draw the graph
    pos = nx.spring_layout(G)

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Highlight the path
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)

    # Display the graph with title showing the total cost
    plt.title(f"Graph for UCS Algorithm (Total Cost: {cost})")
    plt.show()
