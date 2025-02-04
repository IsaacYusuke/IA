#Teste DeepSeek

from transformers import pipeline
import torch

"""" # Verificar se a GPU está disponível
print(torch.cuda.is_available())  # Deve retornar True
print(torch.cuda.get_device_name(0))  # Deve mostrar "GeForce GTX 1050 Ti"
print(torch.cuda.current_device())  # Deve mostrar 0
"""
# Verificar se a GPU está disponível e definir o dispositivo correto
device = 0 if torch.cuda.is_available() else -1

# Carregar o modelo pré-treinado DeepSeek e usar a GPU se disponível
modelo = pipeline("text-generation", model="deepseek-ai/deepseek-coder-1.3b", device=device)

# Função para gerar texto
def gerar_texto(prompt, max_palavras=100):
    resultado = modelo(prompt, 
                   max_length=100,  # Reduzido para melhorar desempenho
                   temperature=0.6,  # Reduz aleatoriedade
                   top_k=40,  # Usa apenas palavras mais prováveis
                   top_p=0.85,  # Filtra palavras improváveis
                   num_return_sequences=1)
    return resultado[0]['generated_text']

# Testar o gerador de texto
if __name__ == "__main__":
    entrada = input("Digite um começo de texto: ")
    print("\nGerando texto... Aguarde, pode levar alguns segundos.")
    texto_gerado = gerar_texto(entrada)
    print("\nTexto Gerado:\n", texto_gerado)
