from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


class ChatGraph:
    def __init__(self, model_name, similarity_threshold=0.3, max_nodes=15):
        self.nodes = {}
        self.edges = []
        self.similarity_threshold = similarity_threshold
        self.max_nodes = max_nodes
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        return self.model.encode(text)

    def add_message(self, msg_id, text, reply_to=None):
        if len(self.nodes) >= self.max_nodes:
            self.prune_nodes()

        self.nodes[msg_id] = text

        if reply_to and reply_to in self.nodes:
            self.edges.append((reply_to, msg_id))

        self.create_similarity_edges(msg_id, text)

    def create_similarity_edges(self, msg_id, new_text):
        new_embedding = self.get_embedding(new_text)
        for existing_id, existing_text in self.nodes.items():
            if existing_id == msg_id:
                continue
            similarity = 1 - cosine(new_embedding, self.get_embedding(existing_text))
            if similarity > self.similarity_threshold:
                self.edges.append((existing_id, msg_id))

    def prune_nodes(self):
        """Elimina los nodos con menos conexiones cuando se supera el máximo."""
        node_degrees = {node: 0 for node in self.nodes}
        for edge in self.edges:
            node_degrees[edge[0]] += 1
            node_degrees[edge[1]] += 1

        sorted_nodes = sorted(node_degrees, key=node_degrees.get)
        nodes_to_remove = sorted_nodes[: len(self.nodes) - self.max_nodes + 1]

        self.nodes = {k: v for k, v in self.nodes.items() if k not in nodes_to_remove}
        self.edges = [
            (a, b)
            for a, b in self.edges
            if a not in nodes_to_remove and b not in nodes_to_remove
        ]

    def visualize_graph(self):
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node, label=self.nodes[node])
        for edge in self.edges:
            G.add_edge(edge[0], edge[1])

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, "label")
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=labels,
            node_color="lightblue",
            edge_color="gray",
        )
        plt.show()

    def get_concatenated_texts(self):
        """Concatena frases conectadas para su análisis posterior."""
        groups = []
        visited = set()

        for node in self.nodes:
            if node not in visited:
                group = self.dfs_collect(node, visited)
                concatenated_text = " CONNECTED ".join(self.nodes[n] for n in group)
                groups.append(concatenated_text)

        return groups

    def dfs_collect(self, node, visited):
        """Realiza DFS para agrupar frases conectadas."""
        stack = [node]
        group = []

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                group.append(current)
                stack.extend(
                    [n2 for n1, n2 in self.edges if n1 == current]
                    + [n1 for n1, n2 in self.edges if n2 == current]
                )

        return group


graph = ChatGraph(model_name="all-mpnet-base-v2")

graph.add_message(1, "There were many women at that party")
graph.add_message(2, "The sky is blue")
graph.add_message(3, "Yes, they were all sluts")
graph.add_message(4, "All they wanted was dick")
graph.add_message(5, "I like spaghetti")
graph.add_message(6, "The weather is very cold today")
graph.add_message(7, "Yes, it has been raining all day")
graph.add_message(8, "Women don't know how to drive")
graph.add_message(9, "That's an absurd stereotype")
graph.add_message(10, "I hate when people don't respect traffic lights")
graph.add_message(11, "Yesterday I was almost run over because of that")
graph.add_message(12, "I love dogs")
graph.add_message(13, "I prefer cats")
graph.add_message(14, "I don't know what to do today")
graph.add_message(15, "We could go to the movies")
graph.add_message(16, "Women only care about money")
graph.add_message(17, "That's not true, many women work and are independent")

concatenated_texts = graph.get_concatenated_texts()
print(concatenated_texts)
