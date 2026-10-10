import os
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

def extract_text(data):
    loader = DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    return docs


def filter_documents(docs : List[Document]) -> List[Document]:
    """ Returns list of documents containing only source inside the metadata section and
    page content of the document"""
    minimal_docs : List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {"source" : src}
            )
        )
    return minimal_docs

def format_docs(docs):
    ''' Combines the page content of the documents returned by the retriever into a single document'''
    return "\n\n".join(doc.page_content for doc in docs)