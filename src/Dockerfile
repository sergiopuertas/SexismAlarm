# Usamos Python slim para imagen ligera
FROM python:3.10-slim

# Variables de entorno para evitar warnings de pip
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Creamos directorio de la app
WORKDIR /app

# Copiamos sólo lo necesario primero (para cache de capas)
COPY deployment/requirements.txt /app/

# Instalamos dependencias
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')" \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copiamos el código y los pesos del modelo
COPY deployment/bot.py model/graph.py model/model.py data_preparation/data_cleaning.py /app/
COPY data_preparation/abb_dict.txt deployment/sexist_words.txt /app/
COPY model/Versions/V2/vocab.pt model/Versions/V2/model_trained.pth /app/
COPY model/glove.twitter.27B.200d.txt /app/

# Exponemos puerto si tu bot usa HTTP (no es obligatorio para Discord bots)
# EXPOSE 8000

# Carga variables desde .env si lo montas como volumen
# ENTRYPOINT carga dotenv automáticamente en tu código
ENTRYPOINT ["python", "bot.py"]
