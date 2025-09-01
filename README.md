# Smart Librarian

## Descriere
Acest proiect implementeaza un chatbot AI pentru recomandari de carti folosind RAG.
Chatbotul poate raspunde conversational la intrebari, poate recomanda carti dintr-o baza locala si poate afisa rezumate detaliate.

Arhitectura include:
- ChromaDB ca vector store pentru cautare
- Ollama ca provider de embeddings si model de chat
- FastAPI pentru backend si API-uri
- CLI pentru interactiune simpla cu utilizatorul

## Functionalitati implementate
### 1. Baza de date de rezumate de carti  
- Fisier `book_summaries.json` cu peste 10 carti si descrieri

### 2. Vector Store cu embeddings  
- Incarcarea rezumatelor in ChromaDB  
- Folosirea de embeddings prin Ollama (`nomic-embed-text:latest`)  
- Posibilitatea de a face cautari semantice dupa tema/context

### 3. Chatbot AI  
- Integrare cu Ollama LLM (`llama3.1:8b`)  
- Chatbotul raspunde la intrebari de tip:  
  - "Vreau o carte despre prietenie si magie"  
  - "Ce recomanzi pentru cineva care prefera povesti de razboi?"
 
### 4. Tool: `get_summary_by_title`  
- Functie Python care returneaza rezumatul complet pentru un titlu exact  
- Dupa ce LLM recomanda o carte, tool-ul ofera rezumatul complet

### 5. Filtru de limbaj nepotrivit  
- Daca utilizatorul foloseste cuvinte ofensatoare, chatbotul nu trimite promptul la LLM si raspunde politicos

### 6. UI simplu: CLI  
- Implementare de Command Line Interface  
- Comenzi disponibile:
  - `health` → verifica status backend  
  - `books` → listeaza toate cartile disponibile  
  - `book "<titlu>"` → arata rezumatul unei carti  
  - `search -q "<termen>"` → cauta carti dupa context
 
## Cum rulezi proiectul

### 1. Instaleaza dependintele
```bash

pip install -r requirements.txt

ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest

python -m app.ingest_chroma

uvicorn app.main:app --host 127.0.0.1 --port 8000

Apoi se pot cere diferite recomandari.
  - `ask -q "<intrebare>"` → intreaba chatbotul direct  
  - `chat` → mod conversational  
  - `shell` → mod tip shell, unde se pot folosi comenzi (`ask`, `search`, `books`)
