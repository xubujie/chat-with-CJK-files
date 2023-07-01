from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
import os
import glob
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from config import CHROMA_SETTINGS


LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}
from dotenv import load_dotenv
load_dotenv()

def load_single_document(file_path: str) -> Document:
    """Loads a single document from a file path."""
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]


def get_text_chunks(docs, chunk_size=250, chunk_overlap=50):
    """Splits a list of documents into chunks of text."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Loaded {len(docs)} documents")
    print(f"Split into {len(chunks)} chunks of text (max. {chunk_size} characters each)")
    return chunks

if __name__ == "__main__":
    docs = load_documents("docs/")
    text_chunks = get_text_chunks(docs)
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(text_chunks, embedding, persist_directory="db", client_settings=CHROMA_SETTINGS)
    vectorstore.persist()
    vectorstore = None
