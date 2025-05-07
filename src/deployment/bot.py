import discord
import os
import torch
import requests
from graph import ChatGraph
from model import LSTMModel
from dotenv import load_dotenv
from data_cleaning import clean_message
from googletrans import Translator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv()
TOKEN = os.getenv('TOKEN')

intents = discord.Intents.default()
intents.messages = True
intents.members = True
intents.guilds = True
intents.message_content = True
client = discord.Client(intents=intents)

conversation_handlers = {}
active_channels = set()
paused_channels = set()

sexist_words = open("sexist_words.txt").read().split()

class ConversationHandler:
    def __init__(self):
        self.chat_graph = ChatGraph()
        self.vocab = self.load_vocab("vocab.pt")
        self.lstm_model = self.load_model("model_trained.pth")

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
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.lstm_model(input_tensor)
        return torch.sigmoid(output).squeeze().item()

    def load_vocab(self, vocab_path):
        return torch.load(vocab_path)

    def load_model(self, model_path):
        model = LSTMModel(self.vocab)
        model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    def reset_graph(self):
        """Reinicia el grafo de conversación"""
        self.chat_graph = ChatGraph()


def get_conversation_handler(channel_id):
    if channel_id not in conversation_handlers:
        conversation_handlers[channel_id] = ConversationHandler()
    return conversation_handlers[channel_id]


def translate_text(text):
    translator = Translator()
    try:
        detected = translator.detect(text)

        if detected.lang == 'en':
            return text

        translation = translator.translate(text, dest='en')
        return translation.text
    except Exception as e:
        print(f"Error de traducción: {e}")
        return text


@client.event
async def on_ready():
    print(f"Bot conectado como {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    channel_id = message.channel.id
    content = message.content

    # Comandos de administración
    if content.startswith('!'):
        if content == "!hello":
            await message.channel.send("👋 Hello! Initializing sexism detector...")
            active_channels.add(channel_id)
            if paused_channels is not None : paused_channels.discard(channel_id)
            get_conversation_handler(channel_id)
            await message.channel.send("✅ Bot has been initialized in the channel")
            return

        elif content == "!stop":
            if channel_id in conversation_handlers:
                del conversation_handlers[channel_id]
            active_channels.discard(channel_id)
            paused_channels.discard(channel_id)
            await message.channel.send("🛑 Bot stopped completely. All data removed.")
            return

        elif content == "!pause":
            if channel_id not in active_channels:
                await message.channel.send("ℹ️ Bot not active in this channel. Use !hello to initialize.")
                return

            paused_channels.add(channel_id)
            await message.channel.send("⏸️ Bot en pausa. Los mensajes no serán analizados pero los datos se conservan.")
            return

        elif content == "!resume":
            if channel_id not in active_channels:
                await message.channel.send("ℹ️ El bot no está activo en este canal. Usa !iniciar primero.")
                return

            if channel_id in paused_channels:
                paused_channels.discard(channel_id)
                await message.channel.send("▶️ Bot reanudado. Los mensajes serán analizados nuevamente.")
            else:
                await message.channel.send("ℹ️ El bot no estaba en pausa.")
            return

        elif content == "!clear":
            if channel_id in conversation_handlers:
                handler = conversation_handlers[channel_id]
                handler.reset_graph()
                await message.channel.send("🔄 Grafo de conversación reiniciado.")
            else:
                await message.channel.send("ℹ️ Primero inicia el bot con !iniciar")
            return

        elif content == "!help":
            help_msg = """
            **Available commands:**
            !hello - Activates the bot in the channel
            !pause - Pauses the analysis without deleting data
            !resume - Resumes the analysis after a pause
            !stop - Completely stops the bot and deletes all data
            !clear - Resets the conversation graph
            !help - Displays the help prompt
            """
            await message.channel.send(help_msg)
            return

    # Procesamiento de mensajes normales
    if channel_id not in active_channels or channel_id in paused_channels:
        return

    handler = get_conversation_handler(channel_id)

    try:
        msg_id = message.id
        reply_to = message.reference.message_id if message.reference else None
        text = translate_text(message.content)
        text = clean_message(text)

        print(f"Mensaje recibido: {text}")

        prediction = handler.add_message(msg_id, text, reply_to)

        print(f"Predicción del mensaje: {prediction}")

        if prediction is not None and prediction > 0.5:
            await message.reply("⚠️ **Este mensaje ha sido clasificado como Sexista**.")

    except Exception as e:
        print(f"Error procesando mensaje: {e}")
        await message.channel.send("❌ Error procesando el mensaje.")


client.run(TOKEN)