#Chatbot

import random
#import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# Banco de dados de respostas
respostas = {
    "oi": "Olá! Como posso te ajudar?",
    "qual seu nome": "Sou um chatbot de IA!",
    "como você funciona": "Eu uso Processamento de Linguagem Natural para entender suas perguntas.",
    "adeus": "Até logo! Se precisar, estarei aqui."
}

# Função para limpar o texto
def preprocessar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    return texto

# Função para encontrar a melhor resposta com base em similaridade
def encontrar_resposta(pergunta):
    pergunta = preprocessar_texto(pergunta)
    if pergunta in respostas:
        return respostas[pergunta]
    
    textos = list(respostas.keys())
    textos.append(pergunta)
    
    vetorizar = TfidfVectorizer()
    tfidf = vetorizar.fit_transform(textos)
    similaridades = cosine_similarity(tfidf[-1], tfidf[:-1])
    indice_resposta = np.argmax(similaridades)
    
    if similaridades[0, indice_resposta] < 0.2:
        return "Desculpe, não entendi. Pode reformular a pergunta?"
    
    return respostas[textos[indice_resposta]]

# Loop de interação com o usuário
def iniciar_chat():
    print("Chatbot: Olá! Como posso te ajudar? (Digite 'sair' para encerrar)")
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() == "sair":
            print("Chatbot: Até mais!")
            break
        resposta = encontrar_resposta(pergunta)
        print("Chatbot:", resposta)

# Iniciar o chatbot
if __name__ == "__main__":
    iniciar_chat()
