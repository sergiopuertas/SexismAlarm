import discord
import torch
from sentence_transformers import SentenceTransformer
from src.model.graph import ChatGraph
from src.model.model import LSTMModel
from language_conversion import process_lang
from src.data_preparation.data_cleaning import clean_txt

TOKEN = "TOKEN"
GUILD_ID = "GUILD_ID"
CHANNEL_ID = "CHANNEL_ID"

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
client = discord.Client(intents=intents)


class ConversationHandler:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.chat_graph = ChatGraph(model_name=model_name)
        self.model_emb_dim = 200
        self.vocab = self.load_vocab("model/vocab.pt")
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model = self.load_model(
            "model/model_trained_cutreattention.pth", self.vocab
        )

    def add_message(self, msg_id, text, reply_to=None):
        self.chat_graph.add_message(msg_id, text, reply_to)

        concatenated_texts = self.chat_graph.get_concatenated_texts()

        for concatenated_text in concatenated_texts:
            if text in concatenated_text:
                prediction = self.classify_message(concatenated_text)
                return prediction

        return None

    def classify_message(self, text):
        indices = [self.vocab.get(token, 0) for token in text.split()]

        input_tensor = (
            torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            output = self.lstm_model(input_tensor)

        prediction = torch.sigmoid(output).squeeze().item()
        return prediction

    def load_vocab(self, vocab_path):
        vocab = torch.load(vocab_path)  # Cargamos el vocabulario guardado
        return vocab

    def load_model(self, model_path, vocab):
        model = LSTMModel(self.vocab, embedding_dim=self.model_emb_dim)
        model.load_state_dict(
            torch.load(model_path, map_location=self.device)["model_state_dict"]
        )
        model.to(self.device)
        model.eval()  # Set model to evaluation mode
        return model


conversation_handler = ConversationHandler()


@client.event
async def on_ready():
    print(f"Bot conectado como {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    msg_id = message.id
    text = message.content
    reply_to = message.reference.message_id if message.reference else None

    text = process_lang(text)
    text = clean_txt(text, "../data_preparation/abb_dict.txt")

    prediction = conversation_handler.add_message(msg_id, text, reply_to)

    if prediction is not None and prediction > 0.5:
        await message.reply("⚠️ **Este mensaje ha sido clasificado como Sexista**.")


client.run(TOKEN)
