# Sexism Alarm Bot 🤖🚨

Un bot de Discord que detecta y alerta mensajes sexistas usando un modelo de IA entrenado.  
Empaquetado con Docker para que puedas desplegarlo en cualquier máquina sin preocuparte por dependencias.

---

## 📋 Índice

- [Características](#-características)  
- [Requisitos](#-requisitos)  
- [Instalación](#-instalación)  
- [Configuración](#-configuración)  
- [Uso con Docker Compose](#-uso-con-docker-compose)  
- [Desarrollo](#-desarrollo)  
- [Contribuir](#-contribuir)  
- [Licencia](#-licencia)  

---

## 🚀 Características

- Detecta contenido sexista en mensajes de Discord.  
- Basado en un modelo BiLSTM + Self-Attention personalizado.  
- Usa análisis de sentimientos y clasificación NLI con Hugging Face.  
- Traducción automática de respuestas con `googletrans`.  
- Fácil de desplegar gracias a Docker & Docker Compose.  

---

## 🛠️ Requisitos

- [Docker >= 20.10](https://docs.docker.com/get-docker/)  
- [Docker Compose >= 1.29](https://docs.docker.com/compose/install/)  
- (Opcional) Git para clonar el repo  

---

## ⚙️ Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/sexism-alarm.git
   cd sexism-alarm
'''
2. Crea un fichero .env en la raíz con tu token de Discord y, opcionalmente, otras variables:
DISCORD_TOKEN=tu_token_aquí

## 📝 Configuración

### .env
Guarda aquí tu DISCORD_TOKEN y cualquier otra variable.

### requirements.txt
Define las dependencias Python necesarias para inference.

### Dockerfile
Construye la imagen con todo lo necesario (modelos, corpora NLTK, cache de Transformers).

### docker-compose.yml
Orquesta el servicio de forma sencilla.

## 🐳 Uso con Docker Compose
Levanta el bot con un solo comando (construye la imagen si ha cambiado):

## 🤝 Contribuir
Haz un fork de este repositorio.
Crea una rama (git checkout -b feature/mi-mejora).
Realiza tus cambios y commitea (git commit -m "Añade nueva funcionalidad").
Envía un Pull Request.

## 📄 Licencia
Este proyecto está bajo la licencia MIT. Consulta el fichero LICENSE para más detalles.





