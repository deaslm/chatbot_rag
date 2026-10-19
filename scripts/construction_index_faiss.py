# Vectorise les événements et construit l'index FAISS via LangChain

import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

# Modèle d'embedding
MODELE_EMBEDDING = "all-MiniLM-L6-v2"

# Taille des chunks
TAILLE_CHUNK = 512

def decouper_en_chunks(texte, taille=TAILLE_CHUNK):
    """Découpe un texte long en morceaux de taille fixe."""
    if len(texte) <= taille:
        return [texte]
    return [texte[i:i+taille] for i in range(0, len(texte), taille)]

def construire_index():
    """
    Charge les données nettoyées, crée des Documents LangChain,
    vectorise avec HuggingFaceEmbeddings et sauvegarde l'index FAISS.
    """
    # Chargement des données nettoyées
    df = pd.read_csv("data/processed/events_rennes_nettoyes.csv")
    print(f"✅ {len(df)} événements chargés")

    # Chargement du modèle d'embedding
    print(f"\nChargement du modèle {MODELE_EMBEDDING}...")
    embeddings = HuggingFaceEmbeddings(model_name=MODELE_EMBEDDING)

    # Création des Documents LangChain (texte + métadonnées associées) pour stocker le contenu vectorisable
    documents = []
    print("Création des Documents LangChain...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Découpage du texte en chunks
        chunks = decouper_en_chunks(str(row["texte_complet"]))

        for chunk in chunks:
            doc = Document(
                page_content=chunk,                  # texte à vectoriser
                metadata={                           # infos associées retournées lors de la recherche
                    "uid": row["uid"],
                    "titre": row["title_fr"],
                    "date_debut": row["firstdate_begin"],
                    "date_fin": row["lastdate_end"],
                    "horaires": row["daterange_fr"],
                    "ville": row["location_city"],
                    "adresse": row["location_address"],
                    "url": row["canonicalurl"]
                }
            )
            documents.append(doc)

    print(f"{len(documents)} Documents LangChain créés")

    # Construction de l'index FAISS
    # FAISS.from_documents() vectorise automatiquement tous les Documents et construit l'index 
    print("Construction de l'index FAISS...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Sauvegarde de l'index
    os.makedirs("faiss_index", exist_ok=True)
    vectorstore.save_local("faiss_index")
    print("Index sauvegardé dans faiss_index/")
    print(f"{vectorstore.index.ntotal} vecteurs indexés")

if __name__ == "__main__":
    construire_index()