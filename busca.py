import chromadb
from sentence_transformers import SentenceTransformer

print("=== INICIALIZANDO O MECANISMO DE BUSCA ===")

# 1. Carregamos o mesmo modelo médico que usamos para criar os vetores
modelo = SentenceTransformer('NeuML/pubmedbert-base-embeddings')

# 2. Conectamos na pasta onde o banco de dados foi salvo na Sprint 2
cliente_chroma = chromadb.PersistentClient(path="banco_vetorial")

# 3. Acessamos a nossa coleção de artigos
colecao = cliente_chroma.get_collection(name="artigos_dengue")

print(f"Banco conectado! Total de blocos disponíveis para busca: {colecao.count()}\n")

# --- SIMULAÇÃO DE UMA PERGUNTA DO USUÁRIO ---
# Você pode alterar o texto dessa frase para testar outras buscas depois!
pergunta = "How does temperature affect dengue transmission and the mosquito life cycle?"

print(f"-> Pergunta digitada: '{pergunta}'")
print("Calculando o embedding da pergunta e buscando no espaço vetorial...\n")

# 4. Transformamos a pergunta em um vetor no R^768
vetor_pergunta = modelo.encode(pergunta).tolist()

# 5. O ChromaDB calcula a menor distância geométrica entre o vetor_pergunta e os 253 vetores
resultados = colecao.query(
    query_embeddings=[vetor_pergunta],
    n_results=3  # Pedimos para ele trazer os 3 blocos mais próximos (top 3)
)

print("=== RESULTADOS ENCONTRADOS (TOP 3 MAIS RELEVANTES) ===")
print("-" * 60)

# 6. Exibimos os blocos de texto que ganharam a competição de proximidade
for i in range(len(resultados['documents'][0])):
    texto = resultados['documents'][0][i]
    distancia = resultados['distances'][0][i]
    id_bloco = resultados['ids'][0][i]
    
    print(f"🎯 [LUGAR #{i+1}] - Bloco ID: {id_bloco} | Distância Geométrica: {distancia:.4f}")
    print(f"Trecho extraído:\n\"{texto}\"")
    print("-" * 60)