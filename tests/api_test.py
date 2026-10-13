# Tests fonctionnels / d'intégration de l'API FastAPI
# Vérifie que les endpoints répondent correctement / que tous les composants (FastAPI + rag_chain + FAISS + Ollama) fonctionnent ensemble

import requests

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Vérifie que l'API est opérationnelle."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("/health OK")

def test_ask_valide():
    """Vérifie qu'une question valide retourne une réponse."""
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": "Quels concerts sont prévus à Rennes ?"}
    )
    assert response.status_code == 200
    assert "reponse" in response.json()
    assert len(response.json()["reponse"]) > 0
    print("/ask question valide OK")

def test_ask_vide():
    """Vérifie qu'une question vide retourne une erreur 400."""
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": ""}
    )
    assert response.status_code == 400
    print("/ask question vide -> erreur 400 OK")

def test_ask_hors_sujet():
    """Vérifie que le RAG gère une question hors sujet."""
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": "Quel est le prix d'une baguette ?"}
    )
    assert response.status_code == 200
    print("/ask question hors sujet OK")

if __name__ == "__main__":
    print("Lancement des tests fonctionnels...\n")
    test_health()
    test_ask_valide()
    test_ask_vide()
    test_ask_hors_sujet()
    print("Tous les tests sont passés !")