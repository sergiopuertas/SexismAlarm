import discord
import os
import torch
import requests
from sentence_transformers import SentenceTransformer
from model.graph import ChatGraph
from model.model import LSTMModel
from dotenv import load_dotenv
from data_preparation.data_cleaning import load_abbreviations, clean_txt

load_dotenv()
TOKEN = os.getenv('TOKEN')

intents = discord.Intents.default()
intents.messages = True
intents.members = True
intents.guilds = True
intents.message_content = True
client = discord.Client(intents=intents)


class ConversationHandler:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        self.chat_graph = ChatGraph(model_name=model_name)
        self.model_emb_dim = 200
        self.vocab = self.load_vocab("model/vocab_v2.pt")
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model = self.load_model(
            "model/model_trained_v2.pth"
        )

    def add_message(self, msg_id, text, reply_to=None):
        self.chat_graph.add_message(msg_id, text, reply_to)

        concatenated_texts = self.chat_graph.get_concatenated_texts()
        print(concatenated_texts)
        for concatenated_text in concatenated_texts:
            if text in concatenated_text:
                if len(concatenated_text.split()) == 1:
                    word = concatenated_text[0].lower()
                    if word in sexist_words:
                        return 1
                    else:
                        return 0

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
        vocab = torch.load(vocab_path)
        return vocab

    def load_model(self, model_path):
        model = LSTMModel(self.vocab, embedding_dim=self.model_emb_dim)
        model.load_state_dict(
            torch.load(model_path, map_location=self.device)["model_state_dict"]
        )
        model.to(self.device)
        model.eval()
        return model

sexist_words = open("deployment/sexist_words.txt").read().split()
abbreviations = load_abbreviations("data_preparation/abb_dict.txt")

conversation_handler = ConversationHandler()

def translate_text(text, target_language='en'):
    url = "https://api.mymemory.translated.net/get"
    params = {
        'q': text,
        'langpair': f'es|{target_language}'
    }
    response = requests.get(url, params=params)
    return response.json()['responseData']['translatedText']

def on_message(text, reply_to):

    text = translate_text(text)
    text = clean_txt(text, abbreviations)

    prediction = conversation_handler.add_message(id+1, text, reply_to)

    if prediction is not None and prediction > 0.5:
         print("⚠️ **Este mensaje ha sido clasificado como Sexista**.")


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
    print(message.reference)
    text = translate_text(text)
    text = clean_txt(text, abbreviations)
    prediction = conversation_handler.add_message(msg_id, text, reply_to)
    print(prediction)
    if prediction is not None and prediction > 0.5:
        await message.reply("⚠️ **Este mensaje ha sido clasificado como Sexista**.")


client.run(TOKEN)
