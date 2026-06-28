# meu-projeto (Projeto RAG Seguro com ChromaDB)

Este é um sistema de **RAG (Retrieval-Augmented Generation)** projetado de forma segura. A aplicação utiliza o **ChromaDB** como banco de dados vetorial para armazenamento e busca semântica, gerenciando chaves e credenciais sensíveis localmente via variáveis de ambiente (`.env`).

## 📁 Estrutura do Projeto

* `banco_vetorial/`: Diretório onde o ChromaDB persiste os dados e índices localmente.
* `dados/`: Pasta para armazenar os documentos (PDFs, TXT, etc.) que compõem a base de conhecimento.
* `app.py`: Interface principal da aplicação ou ponto de entrada do sistema.
* `busca.py`: Módulo responsável pelas consultas (queries) e busca por similaridade no ChromaDB.
* `leitor.py`: Script para processamento, chunking (divisão) e extração de texto dos documentos originais.
* `read.py`: Script utilitário para manipulação ou leitura de arquivos auxiliares.
* `.gitignore`: Proteção que impede o envio de dados locais (`dados/`, `banco_vetorial/`) e do arquivo `.env` para o GitHub.

## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **ChromaDB**: Banco de dados vetorial nativo de IA.
* **python-dotenv**: Gerenciamento seguro de variáveis de ambiente.
* *Mecanismo de LLM/Embeddings (Ex: OpenAI, HuggingFace, Ollama)*

## 🚀 Como Configurar e Executar

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/meu-projeto.git](https://github.com/seu-usuario/meu-projeto.git)
    cd meu-projeto
    ```

2.  **Configure o ambiente seguro (.env):**
    Crie um arquivo `.env` na raiz do projeto. Adicione as chaves necessárias para o seu modelo de *embeddings* ou LLM:
    ```env
    # Exemplo caso use OpenAI (substitua pelo seu provedor se for outro)
    OPENAI_API_KEY=sua_chave_api_aqui
    
    # Caminho opcional para a persistência do Chroma
    CHROMA_DB_DIR=./banco_vetorial
    ```

3.  **Instale as dependências:**
    *(Altere os pacotes de acordo com o seu ambiente de embeddings)*
    ```bash
    pip install chromadb python-dotenv openai
    ```

4.  **Popule o banco vetorial e execute:**
    Primeiro, garanta que os seus documentos de teste estejam na pasta `dados/`, processe-os (se houver um fluxo de ingestão isolado) e execute o app:
    ```bash
    python app.py
    ```
