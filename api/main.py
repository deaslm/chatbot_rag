# API REST FastAPI exposant le système RAG
# Endpoints : /health, /ask, /rebuild

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess

from api.rag_chain import repondre

# Initialisation de l'application FastAPI
# La doc Swagger est accessible sur http://localhost:8000/docs
app = FastAPI(
    title=" API chatbot_rag",
    description="API de recommandation d'événements culturels à Rennes",
    version="1.0.0"
)

# Modèle de données pour la requête /ask
class Question(BaseModel):
    question: str  # la question posée par l'utilisateur

# Modèle de données pour la réponse
class Reponse(BaseModel):
    question: str
    reponse: str

@app.get("/health")
def health():
    """Vérifie que l'API fonctionne correctement."""
    return {"status": "ok", "message": "API opérationnelle"}

@app.post("/ask", response_model=Reponse)
def ask(body: Question):
    """
    Reçoit une question et retourne une réponse générée par le RAG.
    Le système cherche les événements pertinents dans FAISS puis génère une réponse avec Mistral via Ollama.
    """
    # Vérification que la question n'est pas vide
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")
    
    # Appel au système RAG
    reponse = repondre(body.question)
    return Reponse(question=body.question, reponse=reponse)

@app.post("/rebuild")
def rebuild():
    """
    Reconstruit l'index FAISS à partir des données nettoyées. 
    Il sert de mise à jour quand de nouvelles données sont disponibles.
    Attention : endpoint sensible — à protéger en production.
    """
    try:
        # Relance les scripts de récupération events, nettoyage et indexation
        subprocess.run(["python", "scripts/recuperation_events.py"], check=True)
        subprocess.run(["python", "scripts/nettoyage_data.py"], check=True)
        subprocess.run(["python", "scripts/construction_index_faiss.py"], check=True)
        return {"status": "ok", "message": "Index FAISS reconstruit avec succès"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la reconstruction : {str(e)}")