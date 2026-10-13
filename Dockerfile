# Image Docker pour l'API chatbot_rag


# Image de base Python légère
FROM python:3.11-slim

# Répertoire de travail dans le container
WORKDIR /app

# Copie des dépendances en premier (optimise le cache Docker)
COPY requirements_docker.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copie de tout le projet
COPY . .

# Port exposé par l'API FastAPI
EXPOSE 8000

# Commande de lancement de l'API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]