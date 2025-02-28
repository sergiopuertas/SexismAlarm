from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


class ChatGraph:
    def __init__(self, model_name, similarity_threshold=0.5, max_nodes=50):
        self.nodes = {}
        self.edges = []
        self.similarity_threshold = similarity_threshold
        self.max_nodes = max_nodes
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classificator = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   num_labels=2)  # Ajusta num_labels según el problema

        self.nli = pipeline("text-classification", model=self.classificator,tokenizer = self.tokenizer,  framework = "pt")

    def get_embedding(self, text):
        return self.model.encode(text)

    def is_addition(self, reply_text, original_text):
        # Si los textos vienen como listas, conviértelos a string.
        if isinstance(original_text, list):
            original_text = " ".join(original_text)
        if isinstance(reply_text, list):
            reply_text = " ".join(reply_text)

        # Verifica que sean cadenas.
        if not isinstance(reply_text, str) or not isinstance(original_text, str):
            raise ValueError(
                f"Error: Ambos textos deben ser str, pero recibí {type(reply_text)} y {type(original_text)}")

        try:
            if not original_text or not reply_text:
                return False

            results = self.nli(
                (original_text, reply_text),
                top_k=None,
                truncation=True,
                padding=True
            )[0]
            entailment_score = next((res["score"] for res in results if res["label"] == "ENTAILMENT"), 0)
            contradiction_score = next((res["score"] for res in results if res["label"] == "CONTRADICTION"), 0)

            return entailment_score > contradiction_score

        except Exception as e:
            print(f"Error en is_addition: {e}")
            return False

    def add_message(self, msg_id, text, reply_to=None):
        if len(self.nodes) >= self.max_nodes:
            self.prune_nodes()

        self.nodes[msg_id] = text

        if reply_to and reply_to in self.nodes:
            original_text = self.nodes[reply_to]
            if self.is_addition(text, original_text):
                self.edges.append((reply_to, msg_id))
            else:
                print(f"Mensaje {msg_id} en oposición a {reply_to}. No se conecta.")

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
