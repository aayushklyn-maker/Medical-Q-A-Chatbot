from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.helper import extract_text,filter_documents
from langchain_mistralai import MistralAIEmbeddings
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

extracted_docs = extract_text(data="data/")
minimal_docs = filter_documents(extracted_docs)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
chunks = splitter.split_documents(minimal_docs)
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=1024,
        metric="cosine",
        spec = ServerlessSpec(cloud="aws",region="us-east-1")
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    index_name=index_name
)