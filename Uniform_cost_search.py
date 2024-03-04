from collections import deque

class Queue:
    def __init__(self):
        self.elements = deque()
        self.size = -1

    def __iter__(self):
        return iter(self.elements)

    def enqueue(self, element):
        self.elements.append(element)
        self.size+=1

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

class Graph:
    def __init__(self, nodes):
        from collections import defaultdict
        self.graph = defaultdict(list)
        for i in nodes:
            self.graph[i] = []

    def get_graph(self):
        return dict(self.graph)

    def add_edge(self, start, end, cost):
        if (start not in self.graph) and (end not in self.graph):
            print("Both nodes are not found in graph")
        elif start not in self.graph:
            print(start + " not found in graph")
        elif end not in self.graph:
            print(end + " not found in graph")
        else:
            self.graph[start].append((end, cost))

    def add_vertex(self, vertex):
        self.graph[vertex] = []

class MinHeap(Queue):  # Inherits from Queue for enqueue and dequeue operations
    def heapify_up(self, index):
        parent_index = (index - 1) // 2

        #compares the cost of child with its parent, if less than swaps; loops until root node is reached
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

        #compares the cost of the parent with its child, if less then swaps
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

#return the path from the parent dictionary
def return_path(dict, source, goal):
    if goal == source:
        return [goal]
    else:
        path = return_path(dict, source, dict[goal])
        path.append(goal)
        return path

def UCS_path(graph, source, goal):
    from collections import defaultdict
    heap = MinHeap()
    cost = defaultdict(list)
    parent = defaultdict(list)
    visited = []
    # queue = Queue()
    for i in graph:
        parent[i] = None 
        cost[i] = 100000 #initialize the cost of each node to infinity
    heap.insert((source,0))

    while heap:
        node = heap.extract_min()

        # print(f"visited = {visited}")
        # for i in parent:
        #     print(f"parent[{i}] = {parent[i]}")

        if node[0] == goal:
            return (return_path(parent, 'a', 'z'), cost[goal])

        if node[0] not in visited:
            #update the visited list
            visited.append(node[0])

            #insert the adjacent vertices to the heap
            for i in graph[node[0]]:
                #Calculate cost
                g = i[1] + node[1]

                #check the cost and update the parent list
                if g<=cost[i[0]]:
                    parent[i[0]] = node[0]
                    cost[i[0]] = g

                #Insert into the fringe
                heap.insert((i[0], g))

def UCS_traversal(graph, source, goal):
    from collections import defaultdict
    heap = MinHeap()
    cost = defaultdict(list)
    traversal = []
    visited = []
    # queue = Queue()
    for i in graph:
        cost[i] = 100000 #initialize the cost of each node to infinity
    heap.insert((source,0))

    while heap:
        node = heap.extract_min()
        traversal.append(node[0])
        # print(f"visited = {visited}")

        if node[0] == goal:
            return traversal

        if node[0] not in visited:
            #update the visited list
            visited.append(node[0])

            #insert the adjacent vertices to the heap
            for i in graph[node[0]]:
                #Calculate cost
                g = i[1] + node[1]

                #check the cost and update the parent list
                if g<=cost[i[0]]:
                    cost[i[0]] = g

                #Insert into the fringe
                heap.insert((i[0], g))


if __name__ == "__main__":
    nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'z']

    graph = Graph(nodes)

    graph.add_edge('a', 'b', 4)
    graph.add_edge('a', 'c', 3)
    graph.add_edge('b', 'a', 4)
    graph.add_edge('b', 'e', 12)
    graph.add_edge('b', 'f', 5)
    graph.add_edge('c', 'a', 3)
    graph.add_edge('c', 'd', 7)
    graph.add_edge('c', 'e', 10)
    graph.add_edge('d', 'c', 7)
    graph.add_edge('d', 'e', 2)
    graph.add_edge('e', 'b', 12)
    graph.add_edge('e', 'c', 10)
    graph.add_edge('e', 'd', 2)
    graph.add_edge('e', 'z', 5)
    graph.add_edge('f', 'b', 5)
    graph.add_edge('f', 'z', 16)

    graph = graph.get_graph()

    print(UCS_traversal(graph, 'a', 'z'))
    print(UCS_path(graph, 'a', 'z'))