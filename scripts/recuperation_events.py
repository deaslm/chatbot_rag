# Récupère les événements culturels de Rennes via l'API OpenAgenda

import requests      
import pandas as pd  
import os            
from dotenv import load_dotenv  
from tqdm import tqdm           

# Charge les variables du fichier .env (ex: clé API)
load_dotenv()

# Configuration de l'API 
BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records/"

# Clé API lue depuis .env
API_KEY = os.getenv("OPENDATA_API_KEY", "")


PARAMS = {
    "lang": "fr",
    "limit": 100,                                                           # max autorisé par l'API par requête
    "where": 'location_city:"Rennes" AND lastdate_end >= "2025-10-06"'      # filtre sur Rennes + 1 an d'historique et événements à venir
}

# Champs utiles qu'on garde
CHAMPS_UTILES = [
    "uid",                  # identifiant unique
    "title_fr",             # titre de l'événement
    "description_fr",       # description courte → sera vectorisée
    "longdescription_fr",   # description longue → sera vectorisée
    "keywords_fr",          # mots clés
    "firstdate_begin",      # date de début
    "daterange_fr",         # résumé horaires
    "lastdate_end",         # date de fin
    "location_city",        # ville
    "location_address",     # adresse
    "canonicalurl",         # lien vers l'événement
    "category"              # type d'événement
]

def recuperer_evenements(max_events=500):
    """
    Récupère tous les événements par pages de 100 (pagination)
    jusqu'à max_events. Retourne un DataFrame Pandas.
    """
    events = []   # liste qui accumule tous les événements
    offset = 0    # position de départ, 100/page

    with tqdm(total=max_events, desc="Récupération en cours") as barre:
        while offset < max_events:
            # On met à jour l'offset pour récupérer la page suivante
            PARAMS["offset"] = offset

            # Appel HTTP GET vers l'API
            response = requests.get(BASE_URL, params=PARAMS)

            # Si erreur (ex: 404, 500), on arrête
            if response.status_code != 200:
                print(f"Erreur API : {response.status_code}")
                break

            # On extrait les résultats du JSON retourné
            resultats = response.json().get("results", [])

            # Si la page est vide, on a tout récupéré
            if not resultats:
                print("Toutes les données ont été récupérées.")
                break

            # Pour chaque événement, on garde seulement les champs utiles
            events += [{c: e.get(c) for c in CHAMPS_UTILES} for e in resultats]

            # On avance de 100 pour la prochaine page
            offset += 100
            barre.update(len(resultats))

    return pd.DataFrame(events)


if __name__ == "__main__":
    df = recuperer_evenements()

    # Création du dossier data/raw/ s'il n'existe pas
    os.makedirs("data/raw", exist_ok=True)

    # Sauvegarde en CSV
    df.to_csv("data/raw/events_rennes.csv", index=False, encoding="utf-8")
    print(f"{len(df)} événements sauvegardés dans data/raw/events_rennes.csv")