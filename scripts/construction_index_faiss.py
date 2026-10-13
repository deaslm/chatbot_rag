# Vectorise les événements et construit l'index FAISS

import pandas as pd
import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODELE_EMBEDDING = "all-MiniLM-L6-v2"       # modèle open-source, vecteurs de dim 384
TAILLE_CHUNK = 512                          # nb caractères max par chunk

def decouper_en_chunks(texte, taille=TAILLE_CHUNK):
    """Découpe un texte long en morceaux de taille fixe."""
    if len(texte) <= taille:
        return [texte]
    return [texte[i:i+taille] for i in range(0, len(texte), taille)]

def construire_index():
    # Chargement données
    df = pd.read_csv("data/processed/events_rennes_nettoyes.csv")
    modele = SentenceTransformer(MODELE_EMBEDDING)

    # Découpage en chunks + sauvegarde métadonnées associées
    chunks, metadonnees = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        for chunk in decouper_en_chunks(str(row["texte_complet"])):
            chunks.append(chunk)
            metadonnees.append({
                "uid": row["uid"], "titre": row["title_fr"],
                "date_debut": row["firstdate_begin"], "date_fin": row["lastdate_end"],
                "horaires": row["daterange_fr"], "ville": row["location_city"],
                "adresse": row["location_address"], "url": row["canonicalurl"],
                "texte": chunk
            })

    # Vectorisation
    vecteurs = np.array(modele.encode(chunks, show_progress_bar=True)).astype("float32")
    print(f"{vecteurs.shape[0]} vecteurs de dimension {vecteurs.shape[1]}")

    # Construction index FAISS (IndexFlatL2 = recherche par distance euclidienne: lus la distance est petite, plus les textes sont similaires)
    index = faiss.IndexFlatL2(vecteurs.shape[1])
    index.add(vecteurs)

    # Sauvegarde index + métadonnées
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, "faiss_index/index.faiss")
    with open("faiss_index/metadonnees.pkl", "wb") as f:
        pickle.dump(metadonnees, f)

    print(f"Index FAISS : {index.ntotal} vecteurs indexés")
    print("Sauvegardé dans faiss_index/")

if __name__ == "__main__":
    construire_index()