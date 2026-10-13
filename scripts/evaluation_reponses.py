# Évalue les réponses RAG avec 3 métriques :
# 1. Similarité cosinus (sens global)
# 2. Exact Match thématique (bon thème abordé ?)
# 3. Classification manuelle (jugement humain)

import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODELE = "all-MiniLM-L6-v2"

# Mots clés par thème pour vérifier si la réponse aborde le bon sujet
THEMES = {
    "concert": ["concert", "musique", "orchestre"],
    "exposition": ["exposition", "expo", "galerie"],
    "sport": ["sport", "vélo", "tournoi"],
    "gratuit": ["gratuit", "libre"],
    "septembre": ["septembre", "2026"]
}

def similarite_cosinus(ref, gen, modele):
    # Mesure la proximité sémantique entre référence et réponse générée
    v = modele.encode([ref, gen])
    return round(float(cosine_similarity([v[0]], [v[1]])[0][0]), 3)

def exact_match_thematique(question, gen):
    # Détecte le thème de la question et vérifie s'il est bien abordé
    gen_lower = gen.lower()
    for theme, mots in THEMES.items():
        if theme in question.lower():
            # Proportion de mots clés du thème présents dans la réponse
            return round(sum(m in gen_lower for m in mots) / len(mots), 3)
    return 0.0

def evaluer():
    df = pd.read_csv("docs/jeu_de_test.csv")
    modele = SentenceTransformer(MODELE)
    resultats, sims, ems = [], [], []

    for _, l in df.iterrows():
        sim = similarite_cosinus(l["reponse_reference"], l["reponse_generee"], modele)
        em = exact_match_thematique(l["question"], l["reponse_generee"])
        sims.append(sim); ems.append(em)
        print(f"Q : {l['question']}")
        print(f"  Similarité : {sim} | Exact Match : {em} | Manuel : {l['evaluation']}\n")
        resultats.append({
            "question": l["question"],
            "evaluation_manuelle": l["evaluation"],
            "score_similarite": sim,
            "score_exact_match": em,
            "commentaire": l["commentaire"]
        })

    # Bilan global
    print("="*50)
    print(f"Similarité moyenne  : {round(np.mean(sims), 3)}")
    print(f"Exact Match moyen   : {round(np.mean(ems), 3)}")
    print(f"Correctes           : {sum(1 for r in resultats if r['evaluation_manuelle'] == 'correcte')}/5")
    print(f"Partiellement       : {sum(1 for r in resultats if r['evaluation_manuelle'] == 'partiellement correcte')}/5")
    print(f"Incorrectes         : {sum(1 for r in resultats if r['evaluation_manuelle'] == 'incorrecte')}/5")

    # Sauvegarde rapport
    with open("docs/rapport_evaluation.json", "w", encoding="utf-8") as f:
        json.dump({
            "score_similarite_moyen": round(np.mean(sims), 3),
            "score_exact_match_moyen": round(np.mean(ems), 3),
            "correctes": sum(1 for r in resultats if r['evaluation_manuelle'] == 'correcte'),
            "partiellement_correctes": sum(1 for r in resultats if r['evaluation_manuelle'] == 'partiellement correcte'),
            "incorrectes": sum(1 for r in resultats if r['evaluation_manuelle'] == 'incorrecte'),
            "detail": resultats
        }, f, ensure_ascii=False, indent=4)
    print("Rapport sauvegardé dans docs/rapport_evaluation.json")

if __name__ == "__main__":
    evaluer()