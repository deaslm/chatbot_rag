# Tests unitaires validant :
# 1. L'indexation des données (FAISS via LangChain)
# 2. Les performances du système
# 3. Les métriques d'évaluation

import sys
import os
import time
import pandas as pd
import pytest
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluation_reponses import similarite_cosinus, exact_match_thematique

CHEMIN_INDEX = "faiss_index"
CHEMIN_CSV = "data/processed/events_rennes_nettoyes.csv"
MODELE = "all-MiniLM-L6-v2"

# =============================================
# 1. TESTS D'INDEXATION
# =============================================

def test_csv_non_vide():
    """Vérifie que le CSV nettoyé contient des données."""
    df = pd.read_csv(CHEMIN_CSV)
    assert len(df) > 0
    print(f"CSV contient {len(df)} événements")

def test_csv_sans_nan_texte():
    """Vérifie qu'il n'y a pas de NaN dans texte_complet."""
    df = pd.read_csv(CHEMIN_CSV)
    assert df["texte_complet"].isnull().sum() == 0
    print("Pas de NaN dans texte_complet")

def test_index_faiss_existe():
    """Vérifie que les fichiers FAISS LangChain sont bien présents."""
    assert os.path.exists(os.path.join(CHEMIN_INDEX, "index.faiss"))
    assert os.path.exists(os.path.join(CHEMIN_INDEX, "index.pkl"))
    print("Fichiers FAISS LangChain présents")

def test_index_faiss_non_vide():
    """Vérifie que l'index FAISS contient des vecteurs."""
    embeddings = HuggingFaceEmbeddings(model_name=MODELE)
    vectorstore = FAISS.load_local(
        CHEMIN_INDEX, embeddings,
        allow_dangerous_deserialization=True
    )
    assert vectorstore.index.ntotal > 0
    print(f"Index FAISS contient {vectorstore.index.ntotal} vecteurs")

def test_metadonnees_coherentes():
    """Vérifie que les métadonnées sont bien associées aux vecteurs."""
    embeddings = HuggingFaceEmbeddings(model_name=MODELE)
    vectorstore = FAISS.load_local(
        CHEMIN_INDEX, embeddings,
        allow_dangerous_deserialization=True
    )
     
    nb_docs = len(vectorstore.docstore._dict)              # docstore contient les métadonnées
    assert nb_docs == vectorstore.index.ntotal
    print(f"{nb_docs} métadonnées pour {vectorstore.index.ntotal} vecteurs")

# =============================================
# 2. TESTS DE PERFORMANCES
# =============================================

def test_recherche_retourne_5_resultats():
    """Vérifie que FAISS retourne bien 5 résultats via LangChain."""
    embeddings = HuggingFaceEmbeddings(model_name=MODELE)
    vectorstore = FAISS.load_local(
        CHEMIN_INDEX, embeddings,
        allow_dangerous_deserialization=True
    )
    # similarity_search pour chercher dans FAISS
    resultats = vectorstore.similarity_search("événement gratuit", k=5)
    assert len(resultats) == 5
    print("FAISS retourne bien 5 résultats")

def test_temps_recherche_acceptable():
    """Vérifie que la recherche FAISS prend moins de 2 secondes."""
    embeddings = HuggingFaceEmbeddings(model_name=MODELE)
    vectorstore = FAISS.load_local(
        CHEMIN_INDEX, embeddings,
        allow_dangerous_deserialization=True
    )
    debut = time.time()
    vectorstore.similarity_search("événement gratuit", k=5)
    duree = time.time() - debut
    assert duree < 2.0, f"Recherche trop lente : {duree:.2f}s"
    print(f"Temps de recherche : {duree:.3f}s")

# =============================================
# 3. TESTS MÉTRIQUES D'ÉVALUATION
# =============================================

def test_similarite_cosinus_identique():
    """Deux textes identiques doivent avoir similarité = 1."""
    modele = HuggingFaceEmbeddings(model_name=MODELE)
    from sentence_transformers import SentenceTransformer
    st_modele = SentenceTransformer(MODELE)
    score = similarite_cosinus("événement gratuit", "événement gratuit", st_modele)
    assert score == 1.0
    print(f"Similarité textes identiques : {score}")

def test_similarite_cosinus_differents():
    """Deux textes différents doivent avoir similarité < 1."""
    from sentence_transformers import SentenceTransformer
    st_modele = SentenceTransformer(MODELE)
    score = similarite_cosinus("concert de jazz", "emploi recrutement", st_modele)
    assert score < 1.0
    print(f"Similarité textes différents : {score}")

def test_exact_match_theme_correct():
    """Vérifie que l'exact match détecte le bon thème."""
    score = exact_match_thematique(
        "Quels concerts sont prévus ?",
        "Il y a un concert de jazz et un orchestre"
    )
    assert score > 0
    print(f"Exact match thématique : {score}")

if __name__ == "__main__":
    print("Lancement des tests unitaires...\n")
    print("--- INDEXATION ---")
    test_csv_non_vide()
    test_csv_sans_nan_texte()
    test_index_faiss_existe()
    test_index_faiss_non_vide()
    test_metadonnees_coherentes()
    print("\n--- PERFORMANCES ---")
    test_recherche_retourne_5_resultats()
    test_temps_recherche_acceptable()
    print("\n--- MÉTRIQUES ---")
    test_similarite_cosinus_identique()
    test_similarite_cosinus_differents()
    test_exact_match_theme_correct()
    print("Tous les tests unitaires sont passés !")