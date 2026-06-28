import os
import shutil
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# --- FUNÇÕES DE APOIO ---

def extrair_texto_completo(caminho_pdf):
    """Lê todas as páginas do PDF e junta em uma grande string de texto."""
    leitor = PdfReader(caminho_pdf)
    texto_completo = ""
    for pagina in leitor.pages:
        conteudo = pagina.extract_text()
        if conteudo:
            texto_completo += conteudo + "\n"
    return texto_completo

def fatiar_texto(texto, tamanho_chunk=600, overlap=120):
    """
    Divide o texto em blocos de caracteres com uma janela de sobreposição
    para manter o contexto semântico e estatístico das frases.
    """
    chunks = []
    ponteiro = 0
    while ponteiro < len(texto):
        fim = ponteiro + tamanho_chunk
        chunk = texto[ponteiro:fim]
        chunks.append(chunk.strip())
        ponteiro += (tamanho_chunk - overlap)
    return chunks

# --- FLUXO PRINCIPAL ---

pasta_dados = "dados"
pasta_banco = "banco_vetorial"
todos_os_chunks = []

print("=== PASSO 1: Lendo e Fatiando os PDFs ===")
if os.path.exists(pasta_dados):
    for arquivo in os.listdir(pasta_dados):
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(pasta_dados, arquivo)
            print(f"Processando: {arquivo}")
            
            texto_artigo = extrair_texto_completo(caminho)
            chunks_do_artigo = fatiar_texto(texto_artigo)
            todos_os_chunks.extend(chunks_do_artigo)

    print(f"Total de blocos (chunks) gerados: {len(todos_os_chunks)}")
    
    # === PASSO 2: Carregando o Modelo de Embeddings ===
    print("\n=== PASSO 2: Inicializando Modelo Médico BioBERT ===")
    modelo = SentenceTransformer('NeuML/pubmedbert-base-embeddings')
    
    # === PASSO 3: Criando e Armazenando no ChromaDB ===
    print("\n=== PASSO 3: Criando o Banco de Dados Vetorial ===")
    
    # Se a pasta do banco já existir, vamos limpá-la para não duplicar dados neste teste
    if os.path.exists(pasta_banco):
        shutil.rmtree(pasta_banco)
        
    # Inicializa o cliente do ChromaDB apontando para uma pasta persistente local
    cliente_chroma = chromadb.PersistentClient(path=pasta_banco)
    
    # Criamos uma coleção (equivalente a uma tabela em bancos tradicionais)
    colecao = cliente_chroma.create_collection(name="artigos_dengue")
    
    print(f"Gerando embeddings e populando o banco com os {len(todos_os_chunks)} blocos...")
    
    # Criamos IDs únicos para cada pedaço de texto (ex: "id_0", "id_1", ...)
    ids_chunks = [f"id_{i}" for i in range(len(todos_os_chunks))]
    
    # Calculamos os vetores para os 253 chunks de uma só vez
    vetores_chunks = modelo.encode(todos_os_chunks)
    
    # Convertemos os vetores de numpy array para listas, formato exigido pelo ChromaDB
    vetores_lista = vetores_chunks.tolist()
    
    # Adicionamos tudo ao banco vetorial de forma estruturada
    colecao.add(
        embeddings=vetores_lista,
        documents=todos_os_chunks,
        ids=ids_chunks
    )
    
    print("\n=== SPRINT 2 CONCLUÍDA COM SUCESSO! ===")
    print(f"Banco Vetorial salvo na pasta '{pasta_banco}' com {colecao.count()} registros.")
    print("Pronto para a fase de consultas inteligentes (Busca por Similaridade)!")

else:
    print("Pasta 'dados' não encontrada.")