# Sexism Alarm Bot 🤖🚨
<div align="center">

<img src="logo.jpeg" alt="Sexism Alarm Bot" width="400" />

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)]((https://www.apache.org/licenses/LICENSE-2.0))
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=flat&logo=discord&logoColor=white)](https://discord.com/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
</div>

<div align="center">
A Discord bot that detects and alerts sexist messages using a custom trained AI model. 
Packaged with Docker, you can deploy it on any machine without worrying about dependencies.
</div>

## 📋 Table of Contents
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Using Docker Compose](#-using-docker-compose)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Features

- **Smart Detection**: Identifies sexist content in Discord messages
- **Custom AI Model**: Built on BiLSTM + Self-Attention architecture
- **Advanced NLP**: Uses sentiment analysis and NLI classification with Hugging Face
- **Multilingual Support**: Automatic translation of responses with `googletrans`
- **Easy Deployment**: Simple setup with Docker & Docker Compose

## 🛠️ Requirements

- [Docker >= 20.10](https://docs.docker.com/get-docker/)
- [Docker Compose >= 1.29](https://docs.docker.com/compose/install/)
- (Optional) Git to clone the repository

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sexism-alarm.git
   cd sexism-alarm
   ```

2. Create a `.env` file in the root directory with your Discord token and optional variables:
   ```
   DISCORD_TOKEN=your_token_here
   ```

3. Build and run the bot:
   ```bash
   docker-compose up -d
   ```

## 📝 Configuration

### .env
Store your `DISCORD_TOKEN` and any other environment variables here.

### requirements.txt
Defines the Python dependencies needed for inference.

### Dockerfile
Builds an image with everything needed (models, NLTK corpora, Transformers cache).

### docker-compose.yml
Orchestrates the service in a simple way.

## 🐳 Using Docker Compose

Start the bot with a single command (builds the image if it has changed):

```bash
docker-compose up -d
```

To view logs:
```bash
docker-compose logs -f
```

To stop the bot:
```bash
docker-compose down
```

## 💻 Development

To run the bot in development mode:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the bot locally
python bot.py
```

## 🤝 Contributing

1. Fork this repository
2. Create a branch (`git checkout -b feature/my-improvement`)
3. Make your changes and commit (`git commit -m "Add new feature"`)
4. Push to your branch (`git push origin feature/my-improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ by your-username</sub>
</div>
