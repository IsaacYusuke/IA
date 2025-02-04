#Gerador de texto - transformers

from transformers import pipeline

# Carregar o modelo pré-treinado GPT-2
modelo = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B") #troca o modelo?

# Função para gerar texto
def gerar_texto(prompt, max_palavras=50):
    resultado = modelo(prompt, max_length=max_palavras, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.9)
    return resultado[0]['generated_text']

# Testar o gerador de texto
if __name__ == "__main__":
    entrada = input("Digite um começo de texto: ")
    texto_gerado = gerar_texto(entrada)
    print("\nTexto Gerado:\n", texto_gerado)
