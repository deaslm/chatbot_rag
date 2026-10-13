# Nettoie les données brutes et prépare le texte pour la vectorisation

import pandas as pd
import os
import json

def nettoyer_evenements():
    # Chargement des données brutes
    df = pd.read_csv("data/raw/events_rennes.csv")
    
    # Calcul des NaN avant nettoyage pour le rapport
    nan_avant = df.isnull().sum()
    nan_avant = nan_avant[nan_avant > 0].to_dict()
    nb_brut = len(df)

    # Suppression de category (100% vide)
    df = df.drop(columns=["category"])

    # Suppression des lignes sans uid
    df = df.dropna(subset=["uid"])

    # Remplacement des NaN par chaîne vide
    colonnes_texte = ["title_fr", "description_fr", "longdescription_fr", "keywords_fr"]
    df[colonnes_texte] = df[colonnes_texte].fillna("")

    # Création du texte complet pour la vectorisation
    df["texte_complet"] = (
        df["title_fr"] + " " +
        df["description_fr"] + " " +
        df["longdescription_fr"] + " " +
        df["keywords_fr"]
    ).str.strip()

    # Suppression des événements sans texte
    df = df[df["texte_complet"].str.len() > 0]

    # Sauvegarde CSV nettoyé
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/events_rennes_nettoyes.csv", index=False, encoding="utf-8")

    # Sauvegarde rapport nettoyage JSON
    rapport = {
        "evenements_bruts": nb_brut,
        "evenements_nettoyes": len(df),
        "evenements_supprimes": nb_brut - len(df),
        "nan_avant_nettoyage": nan_avant,
        "nan_apres_nettoyage": int(df.isnull().sum().sum()),
        "colonnes_supprimees": ["category"],
        "colonnes_nan_remplaces_par_chaine_vide": colonnes_texte,
        "nouvelle_colonne_creee": "texte_complet"
    }
    with open("data/processed/rapport_nettoyage.json", "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=4)

    print(f"{len(df)} événements nettoyés sauvegardés")
    print("Rapport sauvegardé dans data/processed/rapport_nettoyage.json")
    return df

if __name__ == "__main__":
    nettoyer_evenements()