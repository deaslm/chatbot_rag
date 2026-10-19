# Cœur du système RAG - orchestré par LangChain (LCEL/LangChain Expression Language) qui connecte FAISS, embeddings et Mistral

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

MODELE_EMBEDDING = "all-MiniLM-L6-v2"
NB_RESULTATS = 5

def charger_vectorstore():
    """
    Charge l'index FAISS.
    """
    embeddings = HuggingFaceEmbeddings(model_name=MODELE_EMBEDDING)
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def construire_prompt_template():
    """
    Crée un PromptTemplate LangChain.
    {context} sera injecté automatiquement par le retriever.
    {question} sera injecté par l'utilisateur.
    """
    template = """Tu es un assistant culturel spécialisé dans les événements de Rennes.
Réponds à la question de l'utilisateur en te basant sur les événements ci-dessous.
Si aucun événement ne correspond, dis-le clairement.

Événements disponibles :
{context}

Question : {question}
Réponse :"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def formater_documents(docs):
    """
    Formate les Documents LangChain retournés par FAISS en texte lisible pour le prompt.
    Chaque document contient page_content + metadata.
    """
    return "\n\n".join([
        f"Événement : {doc.metadata.get('titre', '')}\n"
        f"Date : {doc.metadata.get('date_debut', '')} → {doc.metadata.get('date_fin', '')}\n"
        f"Horaires : {doc.metadata.get('horaires', '')}\n"
        f"Lieu : {doc.metadata.get('adresse', '')}, {doc.metadata.get('ville', '')}\n"
        f"Description : {doc.page_content}\n"
        f"Lien : {doc.metadata.get('url', '')}"
        for doc in docs
    ])

def repondre(question):
    """
    Fonction principale du RAG.

    La chaine LangChain fait automatiquement :
    1. retriever   -> cherche les chunks pertinents dans FAISS
    2. prompt      -> injecte les chunks et la question dans le template
    3. llm         -> envoie à Mistral via Ollama
    4. StrOutputParser -> extrait le texte de la réponse
    """
    if not question.strip():
        return "La question ne peut pas être vide."

    # Chargement du vectorstore et création du retriever LangChain
    vectorstore = charger_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": NB_RESULTATS})

    # LLM Mistral via Ollama
    llm = Ollama(
        model="mistral",
        base_url="http://localhost:11434"
    )

    # Construction de la chaîne RAG avec LCEL (LangChain Expression Language)
    # Le pipe connecte chaque étape : retriever, prompt, llm, parser
    prompt = construire_prompt_template()
    chain = (
        {
            # context : documents récupérés par FAISS et formatés
            "context": retriever | formater_documents,
            # question : transmise directement à l'étape suivante
            "question": RunnablePassthrough()
        }
        | prompt      # injection dans le template
        | llm         # génération par Mistral
        | StrOutputParser()  # extraction du texte final
    )

    try:
        return chain.invoke(question)
    except Exception as e:
        return f"Erreur lors de la génération : {str(e)}"

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