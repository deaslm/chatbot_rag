# Cœur du système RAG : connecte FAISS et Mistral (via Ollama en local)

import faiss
import pickle
import numpy as np
import os
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Chargement des variables d'environnement (.env)
load_dotenv()

# Chemins vers l'index FAISS et les métadonnées
CHEMIN_INDEX = "faiss_index/index.faiss"
CHEMIN_META = "faiss_index/metadonnees.pkl"

# Modèle d'embedding — le meme que celui utilisé pour construire l'index
MODELE_EMBEDDING = "all-MiniLM-L6-v2"

# Nombre de chunks pertinents à récupérer depuis FAISS
NB_RESULTATS = 5

def charger_index():
    """Charge l'index FAISS et les métadonnées depuis le disque."""
    index = faiss.read_index(CHEMIN_INDEX)
    with open(CHEMIN_META, "rb") as f:
        metadonnees = pickle.load(f)
    return index, metadonnees

def rechercher_chunks(question, index, metadonnees, modele):
    """
    Vectorise la question et cherche les chunks les plus proches dans FAISS.
    """
    # Vectorisation de la question avec le même modèle que l'index
    vecteur_question = modele.encode([question]).astype("float32")

    # Recherche dans FAISS
    distances, indices = index.search(vecteur_question, NB_RESULTATS)

    # Récupération des métadonnées des chunks trouvés
    resultats = [metadonnees[i] for i in indices[0] if i < len(metadonnees)]
    return resultats

def construire_prompt(question, chunks):
    """
    Construit le prompt envoyé à Mistral. On injecte les chunks pertinents comme contexte.
    """
    contexte = "\n\n".join([
        f"Événement : {c['titre']}\n"
        f"Date : {c['date_debut']} → {c['date_fin']}\n"
        f"Lieu : {c['adresse']}, {c['ville']}\n"
        f"Description : {c['texte']}\n"
        f"Lien : {c['url']}"
        for c in chunks
    ])

    prompt = f"""Tu es un assistant culturel spécialisé dans les événements de Rennes.
Réponds à la question de l'utilisateur en te basant sur les événements ci-dessous.
Si aucun événement ne correspond, dis-le clairement.

Événements disponibles :
{contexte}

Question : {question}
Réponse :"""
    return prompt

def repondre(question):
    """
    Fonction principale du RAG :
    1. Cherche les chunks pertinents dans FAISS
    2. Construit le prompt avec le contexte
    3. Envoie à Mistral via Ollama et retourne la réponse
    """
    # Chargement de l'index et du modèle
    index, metadonnees = charger_index()
    modele = SentenceTransformer(MODELE_EMBEDDING)

    # Recherche des chunks pertinents
    chunks = rechercher_chunks(question, index, metadonnees, modele)

    # Construction du prompt
    prompt = construire_prompt(question, chunks)

    # Appel à Mistral via Ollama
    response = requests.post(
        "http://host.docker.internal:11434/api/chat",
        json={
            "model": "mistral",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False  # on veut la réponse complète d'un coup
        }
    )

    # Gestion des erreurs
    data = response.json()
    if "message" not in data:
        return f"Erreur Ollama : {data}"

    return data["message"]["content"]

if __name__ == "__main__":
    questions = [
        "Quels concerts sont prévus à Rennes ?",
        "Y a-t-il des expositions à Rennes ?",
        "Quels événements ont lieu en septembre 2026 ?",
        "Y a-t-il des événements sportifs à Rennes ?",
        "Y a-t-il des événements gratuits ?"
    ]
    for q in questions:
        print(f"\nQuestion : {q}")
        print(f"Réponse : {repondre(q)}")
        print("-"*50)