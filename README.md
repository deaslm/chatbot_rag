# Chatbot RAG — Puls-Events

# Rapport technique – Assistant intelligent de recommandation d’événements culturels

## 1. Objectifs du projet

### Contexte  
Puls-Events développe une plateforme de recommandations culturelles personnalisées. L'entreprise souhaite tester un chatbot intelligent capable de répondre à des questions sur les événements culturels, en s'appuyant sur un système RAG (Retrieval-Augmented Generation).

### Problématique  
Système RAG - Comment permettre à un utilisateur d'interroger en langage naturel une base d'événements culturels et obtenir une réponse pértinente ?    
RAG répond à ce besoin en combinant :  
- La **recherche vectorielle** (FAISS) pour retrouver les événements pertinents
- La **génération de texte** (Mistral via Ollama) pour formuler une réponse naturelle

### Objectif du POC  
Démontrer la faisabilité technique d'un chatbot RAG capable de :
- Répondre à des questions sur les événements culturels de la zone géo. choisie
- S'appuyer sur des données réelles issues de l'API OpenAgenda
- Exposer le système via une API exploitable par les équipes produit et marketing

### Périmètre  
- **Zone géographique** : Rennes (Bretagne)
- **Période** : 1 an d'historique + événements à venir (depuis octobre 2025)
- **Volume** : 500 événements récupérés, 483 après nettoyage
- **Source** : API OpenAgenda


## 2. Architecture du système

### Schéma

**PIPELINE D'INDEXATION :**  
API OpenAgenda -> Nettoyage (Pandas) -> Chunking (512 car.) -> Embeddings (all-MiniLM-L6-v2) -> Index FAISS(IndexFlatL2)

**PIPELINE DE REQUETE :**  
Question -> Embedding -> Recherche FAISS (top 5 chunks) -> Prompt enrichi -> Mistral (Ollama) -> Réponse

**API FASTAPI :**  
GET /health ; POST /ask ; POST /rebuild

### Technologies utilisées  
Récupération données / Appels API OpenAgenda :  Python requests  
Nettoyage / Prétraitement des données : Pandas  
Embeddings / Vectorisation des textes : sentence-transformers  
Base vectorielle / Recherche par similarité : FAISS (IndexFlatL2)   
LLM / Génération de réponses : Mistral via Ollama  
Orchestration / Pipeline RAG : LangChain  
API / Exposition du système : FastAPI Uvicorn  
Conteneurisation / Déploiement local : Docker  


## 3. Préparation et vectorisation des données

### Source de données  
API OpenAgenda avec les filtres :  
- `location_city:"Rennes"`
- `lastdate_end >= "2025-10-06"`
- `limit=100` par page (pagination automatique)

### Nettoyage  
500 événements bruts -> 483 événements nettoyés  
- 500 valeurs manquantes dans `category` -> colonne supprimée (100% vide)
- 17 événements sans titre ni description -> supprimés
- 4 colonnes où les nan ont été remplacés par "" :"title_fr","description_fr","longdescription_fr","keywords_fr"

### Chunking  
Les textes sont découpés en chunks de **512 caractères** car :  
- Le modèle `all-MiniLM-L6-v2` a une limite de 256 tokens ≈ 512 caractères (assez précis pour cibler l'info recherché, assez grand pour garder du sens)
- La longueur moyenne des textes est de 852 caractères (max : 7036)
- Résultat : **1039 chunks** pour 483 événements

### Embedding  
- **Modèle** : `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensionnalité** : 384
- **Choix** : open-source, local, gratuit, rapide
- **Format** : vecteurs float32


## 4. Choix du modèle NLP

### Modèle sélectionné  
**Mistral** via **Ollama** (exécution locale)

### Pourquoi ce modèle ?  
- Gratuit et open-source
- Compatible avec une exécution locale via Ollama
- En production : remplacement par l'API Mistral cloud

### Prompt utilisé  
```
Tu es un assistant culturel spécialisé dans les événements de Rennes.
Réponds à la question de l'utilisateur en te basant sur les événements ci-dessous.
Si aucun événement ne correspond, dis-le clairement.

Événements disponibles :
{contexte}
Question : {question}
Réponse :
```
### Limites du modèle  
- Ne comprend pas les dates au format technique donc ne trouve pas la réponse adéquate
- Performances limitées par les ressources locales (Ollama)
- En production : utiliser l'API Mistral cloud pour de meilleures performances


## 5. Construction de la base vectorielle

### FAISS utilisé  
**Index** : `IndexFlatL2` — recherche par distance euclidienne, 1039 vecteurs de dimension 384

### Stratégie de persistance  
- **Format** : `index.faiss` (index FAISS) / `metadonnees.pkl` (métadonnées Python)
- **Nommage** : `faiss_index/index.faiss`  / `faiss_index/metadonnees.pkl`

### Métadonnées conservées par chunk

"uid": identifiant unique,  
"titre": titre de l'événement,  
"date_debut": première date,  
"date_fin": dernière date,  
"horaires": résumé des horaires,  
"ville": ville,  
"adresse": adresse complète,  
"url": lien OpenAgenda,  
"texte": contenu du chunk


## 6. API et endpoints exposés

### Framework  
**FastAPI** avec documentation Swagger sur http://localhost:8000/docs

### Endpoints  
`/health` /  GET / Vérifie que l'API est opérationnelle  
`/ask` / POST / Pose une question, retourne une réponse RAG   
`/rebuild`/ POST / Reconstruit l'index FAISS

### Format des requêtes/réponses  
**POST /ask**
```json
// Requête
{"question": "Quels concerts sont prévus à Rennes ?"}

// Réponse
{
  "question": "Quels concerts sont prévus à Rennes ?",
  "reponse": "Il y a plusieurs concerts prévus à Rennes..."
}
```
### Exemple d'appel API avec curl  
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts sont prévus à Rennes ?"}'
```
### Gestion des erreurs  
- Question vide → erreur **400** Bad Request
- Erreur Ollama → message d'erreur explicite dans la réponse


## 7. Évaluation du système

### Jeu de test annoté  
- **5 questions** représentatives
- **Méthode d'annotation** : réponses de référence construites manuellement
- **Fichier** : `docs/jeu_de_test.csv`

### Métriques d'évaluation  
Similarité cosinus (Proximité sémantique référence vs générée) : **0.602**  
Exact Match thématique (Le bon thème est-il abordé ?) : **0.567**   
Classification manuelle (Correcte / Partielle / Incorrecte) : 3/5 correctes 

### Résultats obtenus  
**Analyse quantitative :**
- 3/5 correctes
- 1/5 partiellement correcte
- 1/5 incorrecte  
**Analyse qualitative :**  
Concerts à Rennes / Correcte / Bonne recherche sémantique  
Expositions à Rennes / Partielle / Résultat généraliste retourné  
Événements septembre 2026 / Incorrecte / Dates au format technique non comprises  
Événements sportifs / Correcte / Bonne identification thématique   
Événements gratuits / Correcte / Bonne identification thématique


## 8. Recommandations et perspectives

### Ce qui fonctionne bien  
- Recherche sémantique efficace pour les questions thématiques
- Réponses bien formulées
- Pipeline complet et reproductible
- API fonctionnelle et documentée

### Limites du POC  
- **Dates** : le modèle d'embedding ne comprend pas les dates au format ISO (`2025-12-14T14:00:00`)
- **LLM local** : Ollama plus lent qu'une API cloud
- **RAGAS** : incompatible avec mon Python 3.14 — non intégré

### Améliorations possibles  
- Convertir les dates
- Intégrer RAGAS avec Python 3.11 pour des métriques plus robustes
- Passer à l'API Mistral cloud en production

### Passage en production  
POC local (Ollama) -> API Mistral cloud -> Déploiement cloud -> Monitoring


## 9. Organisation du dépôt GitHub

chatbot_rag/

.env.example              : Modèle de configuration (clés API)  
.gitignore                : Fichiers exclus du versionnage  
requirements.txt          : Dépendances Python (env local)  
requirements_docker.txt   : Dépendances Python (Docker)  
Dockerfile                : Image Docker de l'API  
README.md                 : rapport technique

data/  
raw/                  : Données brutes OpenAgenda (CSV)  
processed/            : Données nettoyées + rapport nettoyage  

scripts/  
recuperation_events.py      : Récupération via API OpenAgenda  
nettoyage_data.py           : Nettoyage et prétraitement  
construction_index_faiss.py : Vectorisation + index FAISS  
evaluation_reponses.py      : Métriques d'évaluation

api/  
main.py               : API FastAPI (endpoints)  
rag_chain.py          : Cœur du système RAG

faiss_index/  
index.faiss            : Index vectoriel FAISS  
metadonnees.pkl        : Métadonnées des chunks

tests/  
api_test.py            : Tests fonctionnels de l'API  
test_unitaires.py      : Tests unitaires (indexation, métriques)

docs/  
jeu_de_test.csv              : Jeu de test annoté (5 questions)  
rapport_evaluation.json      : Résultats des métriques  
rapport_tests_unitaires.txt  : Résultats pytest


## 10. Annexes

### Extraits du jeu de test annoté

question,reponse_reference,reponse_generee,evaluation,commentaire

"Y a-t-il des événements sportifs à Rennes ?","La caravane du sport avec 21 dates à Rennes, Balade à vélo autour de Rennes le 29/05/2026, Atelier auto-réparation de vélo le 05/12/2025","Caravane du sport le 24/08/2026, Tournoi de soccer le 20/07/2026","correcte","Bonne identification des événements sportifs"

### Exemple de réponse JSON  
```json
{
  "question": "Y a-t-il des événements gratuits ?",
  "reponse": "Oui, il y a plusieurs événements gratuits disponibles :

1. Les colonies de vacances : https://openagenda.com/rennes-quartiers-ouest/events/colonies-de-vacances
2. Lectures inspirantes (pour les enfants de 3 à 8 ans) : https://openagenda.com/rennes-education/events/lectures-inspirantes-9884373
3. Festival Ancrages : https://openagenda.com/sortir-rennesmetropole/events/festival-ancrages-8301824
4. Festival Ancrages (pour les jeunes public) : https://openagenda.com/sortir-rennesmetropole/events/festival-ancrages-8301824
5. Mardi Concert - THE FORROSCOPIC'S : https://openagenda.com/ete-rennes/events/mardi-concert-the-forroscopics
6. Falling Walls Lab Rennes 2026 : Libre d'entrée, mais les candidatures sont ouvertes jusqu'au 17 mai : https://actu.univ-rennes.fr/actualites/le-falling-walls-lab-arrive-luniversite-de-rennes"
}
```


# Installation et lancement

```bash
# 1. Cloner le dépôt
git clone https://github.com/deaslm/chatbot_rag.git
cd chatbot_rag

# 2. Créer l'environnement virtuel
python -m venv .venv
source .venv/Scripts/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables
cp .env.example .env

# 5. Construire l'index FAISS
python scripts/recuperation_events.py
python scripts/nettoyage_data.py
python scripts/construction_index_faiss.py

# 6. Lancer l'API
uvicorn api.main:app --reload --port 8000

# 7. Ou via Docker
docker build -t chatbot_rag .
docker run -p 8000:8000 --env-file .env chatbot_rag
```
