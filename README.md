# Medical Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions using the documents you provide. Documents are split into chunks, embedded with Mistral, stored in Pinecone, and retrieved at question time so the LLM answers from your sources instead of guessing.

> **Disclaimer:** This project is for educational purposes. It provides general information only and is not a substitute for professional medical advice, diagnosis, or treatment.

## How it works

```
Documents (data/)
   │  extract + filter + split into chunks
   ▼
Mistral embeddings (mistral-embed, 1024 dimensions)
   │
   ▼
Pinecone vector index  ◄── top 3 similar chunks ──┐
                                                  │
User question ──► Flask /get ──► Retriever ──► Prompt + context ──► Groq LLM ──► Answer (markdown)
```

1. **Indexing (run once):** the indexing script loads files from `data/`, splits them into 1000-character chunks (20 overlap), embeds them, and uploads them to a Pinecone index called `medical-chatbot`.
2. **Chatting:** the Flask app takes the user's question, retrieves the 3 most similar chunks, sends them with the question to the LLM, and returns a structured markdown answer.
3. **Frontend:** a single-page chat UI renders the answer with headings, bullets, and bold text.

## Tech stack

| Part | Tool |
|---|---|
| Backend | Flask |
| Orchestration | LangChain |
| Embeddings | Mistral AI (`mistral-embed`) |
| Vector database | Pinecone (serverless, AWS `us-east-1`, cosine) |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Frontend | HTML, CSS, vanilla JavaScript |

## Project structure

```
.
├── app.py                 # Flask app and RAG chain
├── store_index.py         # Builds the Pinecone index from data/
├── setup.py               # Package setup
├── template.sh            # Script that creates the project folders and files
├── requirements.txt
├── data/                  # Source documents (The Gale Encyclopedia of Medicine, 3rd Edition)
├── research/
│   └── trials.ipynb       # Notebook used for experimenting with the pipeline
├── src/
│   ├── __init__.py
│   ├── helper.py          # extract_text, filter_documents, format_docs
│   └── prompt.py          # system_prompt
├── templates/
│   └── index.html         # Chat page
├── static/
│   └── style.css          # Chat styles
├── .env                   # API keys (not committed)
├── .gitignore
├── LICENSE
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/aayushklyn-maker/Medical-Q-A-Chatbot.git
cd Medical-Q-A-Chatbot
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API keys

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_key
MISTRAL_API_KEY=your_mistral_key
GROQ_API_KEY=your_groq_key
```

You can get keys from [Pinecone](https://www.pinecone.io/), [Mistral AI](https://console.mistral.ai/), and [Groq](https://console.groq.com/).

### 4. Add your documents

Place your medical documents in the `data/` folder. This project uses *The Gale Encyclopedia of Medicine, 3rd Edition* as its knowledge source. The book is copyrighted, so it is not included in the repository. Add your own copy to `data/` before indexing.

### 5. Build the vector index

```bash
python store_index.py
```

This creates the `medical-chatbot` index in Pinecone (if it doesn't exist) and uploads your embedded chunks. You only need to run it again when your documents change.

### 6. Run the app

```bash
python app.py
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

## Customizing

- **Answer format and tone:** edit `system_prompt` in `src/prompt.py`.
- **Number of retrieved chunks:** change `k` in `app.py` (`search_kwargs={"k": 3}`).
- **Chunk size and overlap:** change them in `store_index.py`, then rebuild the index.
- **LLM model:** change the model name in `ChatGroq(...)` in `app.py`.
- **Embedding model:** if you switch it, make sure the Pinecone index dimension matches the new model's output, then rebuild the index.

## API

| Method | Route | Description |
|---|---|---|
| GET | `/` | Serves the chat page |
| POST | `/get` | Accepts form field `msg` and returns the answer as markdown text |

## License

This project is licensed under the [Apache License 2.0](LICENSE).
