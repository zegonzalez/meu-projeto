import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

# Configura a chave na memória do sistema de forma oculta
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# ... (o restante do seu código do projeto RAG continua daqui para baixo)



import os
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

print("=== INICIALIZANDO ASSISTENTE CIENTÍFICO (RAG) ===")

# --- COLE SUA CHAVE APENAS AQUI NA LINHA 9 ---
os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6Ig0-daHcsYB1xdgtk0i0-nNQ5JqZqOnmBGamiBleddKg"

# Inicializar componentes locais (Busca)
modelo_embeddings = SentenceTransformer('NeuML/pubmedbert-base-embeddings')
cliente_chroma = chromadb.PersistentClient(path="banco_vetorial")
colecao = cliente_chroma.get_collection(name="artigos_dengue")

# Inicializar o cliente do Gemini (Geração)
client = genai.Client()

print("Sistema pronto! Digite sua pergunta sobre os artigos científicos.")
print("-" * 60)

# Entrada da pergunta em português
pergunta_usuario = "Como a variação de temperatura ao longo do dia afeta a capacidade do mosquito de transmitir dengue?"

print(f"\nUser: {pergunta_usuario}")
print("Buscando evidências científicas nos PDFs de dengue...")

# Retrieval: Busca semântica no banco vetorial
vetor_pergunta = modelo_embeddings.encode(pergunta_usuario).tolist()
resultados = colecao.query(query_embeddings=[vetor_pergunta], n_results=3)

# Reunimos os 3 blocos de texto encontrados em uma única string de contexto
contexto_artigos = "\n\n".join(resultados['documents'][0])

# Generation: Engenharia de Prompt e envio para a LLM
prompt_estruturado = f"""
Você é um assistente acadêmico especialista em epidemiologia e modelagem matemática.
Sua tarefa é responder à pergunta do usuário utilizando estritamente os trechos de artigos científicos fornecidos abaixo como contexto.
Responda de forma clara, formal e em português, citando os dados técnicos presentes no texto (como faixas de temperatura, se houver).

Trechos dos artigos científicos (Contexto):
{contexto_artigos}

Pergunta do Usuário:
{pergunta_usuario}

Resposta Acadêmica:
"""

print("Sintetizando resposta final com o Gemini...")

# Chamamos o modelo Gemini 2.5 Flash
resposta_ia = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt_estruturado,
)

print("\n" + "="*20 + " RESPOSTA DA IA VINCULADA AOS ARTIGOS " + "="*20)
print(resposta_ia.text)
print("=" * 78)