# Tests unitaires validant :
# 1. L'indexation des données (FAISS + métadonnées)
# 2. Les performances du système (temps de réponse, résultats)
# 3. Les métriques d'évaluation (similarité cosinus, exact match)

import sys
import os
import time
import faiss
import pickle
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluation_reponses import similarite_cosinus, exact_match_thematique

CHEMIN_INDEX = "faiss_index/index.faiss"
CHEMIN_META = "faiss_index/metadonnees.pkl"
CHEMIN_CSV = "data/processed/events_rennes_nettoyes.csv"
MODELE = "all-MiniLM-L6-v2"

# =============================================
# TESTS D'INDEXATION
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
    """Vérifie que les fichiers FAISS sont bien présents."""
    assert os.path.exists(CHEMIN_INDEX)
    assert os.path.exists(CHEMIN_META)
    print("Fichiers FAISS présents")

def test_index_faiss_non_vide():
    """Vérifie que l'index FAISS contient des vecteurs."""
    index = faiss.read_index(CHEMIN_INDEX)
    assert index.ntotal > 0
    print(f"Index FAISS contient {index.ntotal} vecteurs")

def test_metadonnees_coherentes():
    """Vérifie que le nombre de métadonnées correspond au nombre de vecteurs."""
    index = faiss.read_index(CHEMIN_INDEX)
    with open(CHEMIN_META, "rb") as f:
        metadonnees = pickle.load(f)
    assert len(metadonnees) == index.ntotal
    print(f"{len(metadonnees)} métadonnées pour {index.ntotal} vecteurs")

# =============================================
# TESTS DE PERFORMANCES
# =============================================

def test_recherche_retourne_5_resultats():
    """Vérifie que FAISS retourne bien 5 résultats."""
    index = faiss.read_index(CHEMIN_INDEX)
    modele = SentenceTransformer(MODELE)
    vecteur = modele.encode(["concerts à Rennes"]).astype("float32")
    distances, indices = index.search(vecteur, 5)
    assert len(indices[0]) == 5
    print("FAISS retourne bien 5 résultats")

def test_temps_recherche_acceptable():
    """Vérifie que la recherche FAISS prend moins de 2 secondes."""
    index = faiss.read_index(CHEMIN_INDEX)
    modele = SentenceTransformer(MODELE)
    vecteur = modele.encode(["concerts à Rennes"]).astype("float32")
    debut = time.time()
    index.search(vecteur, 5)
    duree = time.time() - debut
    assert duree < 2.0, f"Recherche trop lente : {duree:.2f}s"
    print(f"Temps de recherche : {duree:.3f}s")

# =============================================
# TESTS MÉTRIQUES D'ÉVALUATION
# =============================================

def test_similarite_cosinus_identique():
    """Deux textes identiques doivent avoir similarité = 1."""
    modele = SentenceTransformer(MODELE)
    score = similarite_cosinus("concert à Rennes", "concert à Rennes", modele)
    assert score == 1.0
    print(f"Similarité textes identiques : {score}")

def test_similarite_cosinus_differents():
    """Deux textes différents doivent avoir similarité < 1."""
    modele = SentenceTransformer(MODELE)
    score = similarite_cosinus("concert de jazz", "forum emploi", modele)
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
    print("🧪 Lancement des tests unitaires...\n")
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