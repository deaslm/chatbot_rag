# Chatbot RAG — Puls-Events

Assistant intelligent de recommandation d'événements culturels basé sur RAG.

## Installation
1. Cloner le dépôt depuis GitHub
2. Créer l'environnement virtuel : `python -m venv .venv`
3. Activer : `source .venv/Scripts/activate`
4. Installer les dépendances : `pip install -r requirements.txt`
5. Copier le modèle de config : `cp .env.example .env`
   puis ouvrir `.env` et renseigner votre clé API Mistral

## Structure
- `data/` : données brutes et nettoyées
- `scripts/` : scripts d'ingestion et vectorisation
- `api/` : API FastAPI
- `tests/` : tests unitaires
- `docs/` : rapport technique