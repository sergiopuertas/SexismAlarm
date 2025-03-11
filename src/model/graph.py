from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

class ChatGraph:
    def __init__(self, model_name = 'cross-encoder/nli-deberta-v3-large', max_nodes=10):
        self.sim_threshold = 0.2
        self.nodes = {}
        self.edges = []
        self.max_nodes = max_nodes
        self.nli_model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name,use_fast = False)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)

        # Modelo para embeddings contextuales
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        self.nli_model.eval()

    def analyze_relation(self, text_a, text_b):
        """Analiza la relación semántica entre dos textos con comprensión contextual"""
        # Etiquetas del modelo: 0: contradicción, 1: entailment, 2: neutral
        inputs = self.tokenizer(text_a, text_b, return_tensors='pt', truncation=True)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        return probs

    def contextual_similarity(self, text_a, text_b):
        """Calcula similitud semántica con contexto conversacional"""
        embeddings = self.embedding_model.encode([text_a, text_b])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    def is_addition(self, reply_text, original_text):
        if not original_text or not reply_text:
            return False

        try:
            # Análisis de relación semántica
            probs = self.analyze_relation(original_text, reply_text)

            if probs.argmax() == 1:  # Entailment
                print(f"Entailment detectado: {probs}")
                return True
            elif probs.argmax() == 0:  # Contradicción
                print(f"Contradicción detectada: {probs}")
                return False
            else:  # Neutral
                similarity = self.contextual_similarity(original_text, reply_text)
                print(f"Similitud neutral: {similarity}")
                return similarity > self.sim_threshold and self.context_relevance_check(reply_text, original_text)

        except Exception as e:
            print(f"Error en is_addition: {e}")
            return False

    def context_relevance_check(self, text_a, text_b):
        """Verifica relevancia contextual usando embeddings de conversación completa"""
        context_embedding = self.embedding_model.encode(" ".join(self.nodes.values()))
        text_a_embed = self.embedding_model.encode(text_a)
        text_b_embed = self.embedding_model.encode(text_b)

        # Calcula similitud con el contexto general
        sim_a = util.pytorch_cos_sim(context_embedding, text_a_embed).item()
        sim_b = util.pytorch_cos_sim(context_embedding, text_b_embed).item()

        return (sim_a + sim_b) / 2 > 0.6

    def add_message(self, msg_id, text, reply_to=None):
        self.nodes[msg_id] = text

        if reply_to is not None and reply_to in self.nodes:
            parent_text = self.nodes[reply_to]
            if self.is_addition(text, parent_text):
                self.edges.append((reply_to, msg_id))
        else:
            previous_keys = list(self.nodes.keys())
            if msg_id in previous_keys:
                previous_keys.remove(msg_id)
            last_three = previous_keys[-3:] if previous_keys else []
            for existing_id in last_three:
                existing_text = self.nodes[existing_id]
                if self.is_addition(text, existing_text):
                    self.edges.append((existing_id, msg_id))

        return self.get_concatenated_texts()

    def get_concatenated_texts(self):
        groups = []
        visited = set()

        # Ordenar los nodos por orden de aparición (asumiendo que el ID del mensaje es incremental)
        sorted_nodes = sorted(self.nodes.keys())

        for node in sorted_nodes:
            if node not in visited:
                # Obtener conexiones directas
                connections = self.get_direct_connections(node)
                connections.append(node)

                # Ordenar las conexiones por orden de aparición
                sorted_connections = sorted(connections, key=lambda x: x)

                # Marcar como visitados
                visited.update(sorted_connections)

                # Concatenar los textos en orden
                concatenated_text = " CONNECTED ".join(self.nodes[n] for n in sorted_connections)
                groups.append(concatenated_text)

        return groups

    def get_direct_connections(self, node):
        return [n2 if n1 == node else n1 for n1, n2 in self.edges if node in (n1, n2)]