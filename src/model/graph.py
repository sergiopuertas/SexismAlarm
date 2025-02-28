from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import networkx as nx
import matplotlib.pyplot as plt
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class ChatGraph:
    def __init__(self, model_name, similarity_threshold=0.5, max_nodes=50):
        self.nodes = {}
        self.edges = []
        self.similarity_threshold = similarity_threshold
        self.max_nodes = max_nodes
        self.model = SentenceTransformer(model_name)
        self.classificator = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')
        self.clastokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        self.classificator.eval()

    def get_embedding(self, text):
        return self.model.encode(text)

    def is_addition(self, reply_text, original_text):
        if isinstance(original_text, list):
            original_text = " ".join(original_text)
        if isinstance(reply_text, list):
            reply_text = " ".join(reply_text)

        if not original_text or not reply_text:
            return False

        try:
            tokens = self.clastokenizer(original_text,reply_text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                logits = self.classificator(**tokens).logits
            print(logits)
            results = torch.argmax(logits, dim=-1).item()

            return False if results == 0 else True

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
        else: self.create_similarity_edges(msg_id,text)


    def create_similarity_edges(self, msg_id, new_text):
        new_embedding = self.get_embedding(new_text)
        for existing_id, existing_text in self.nodes.items():
            if existing_id == msg_id:
                continue
            similarity = 1 - cosine(new_embedding, self.get_embedding(existing_text))
            if similarity > self.similarity_threshold:
                self.edges.append((existing_id, msg_id))

    def prune_nodes(self):
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
        groups = []
        visited = set()

        for node in self.nodes:
            if node not in visited:
                direct_connections = self.get_direct_connections(node)
                direct_connections.append(node)
                visited.update(direct_connections)

                concatenated_text = " CONNECTED ".join(self.nodes[n] for n in direct_connections)
                groups.append(concatenated_text)

        return groups

    def get_direct_connections(self, node):
        """Devuelve los nodos directamente conectados a un nodo dado."""
        return [n2 if n1 == node else n1 for n1, n2 in self.edges if node in (n1, n2)]

