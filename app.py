from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from src.prompt import *
from src.helper import format_docs
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

embedding_model = MistralAIEmbeddings(
    model='mistral-embed'
)

index_name = "medical-chatbot"
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(search_type='similarity',search_kwargs = {"k" : 3})
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])
model = ChatGroq(model="openai/gpt-oss-20b")
rag_chain = (
    {'context' : retriever | format_docs, "question" : RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a question.", 400

    try:
        answer = rag_chain.invoke(msg)
    except Exception as e:
        print("RAG error:", e)
        return "Sorry, I couldn't process that question. Please try again.", 500

    return answer

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)