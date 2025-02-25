import os
from chromadb import PersistentClient
from dotenv import load_dotenv
from enum import Enum
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pathlib import Path
from typing import Any, List, Generator


load_dotenv()


class Rag(Enum):
    GPT_MODEL = "gpt-4o-mini"
    EMBED_MODEL = OpenAIEmbeddings()
    DB_NAME = "vector_db"


def add_metadata(doc: Document, doc_type: str) -> Document:
    """
    Add metadata to a Document object.

    :param doc: The Document object to add metadata to.
    :type doc: Document
    :param doc_type: The type of document to be added as metadata.
    :type doc_type: str
    :return: The Document object with added metadata.
    :rtype: Document
    """
    doc.metadata["doc_type"] = doc_type
    return doc


def get_chunks(folders: Generator[Path, None, None], file_ext=".md") -> List[Document]:
    """
    Load documents from specified folders, add metadata, and split them into chunks.

    :param folders: List of folder paths containing documents.
    :type folders: List[str]
    :param file_ext:
        The file extension to get from a local knowledge base (e.g. '.md')
    :type file_ext: str
    :return: List of document chunks.
    :rtype: List[Document]
    """
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob=f"**/*{file_ext}", loader_cls=TextLoader)

        folder_docs = loader.load()
        documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    return chunks


def create_vector_db(db_name: str, chunks: List[Document], embeddings: Any) -> Any:
    """
    Create a vector database from document chunks.

    :param db_name: Name of the database to create.
    :type db_name: str
    :param chunks: List of document chunks.
    :type chunks: List[Document]
    :param embeddings: Embedding function to use.
    :type embeddings: Any
    :return: Created vector store.
    :rtype: Any
    """
    # Delete if already exists
    if os.path.exists(db_name):
        Chroma(
            persist_directory=db_name, embedding_function=embeddings
        ).delete_collection()

    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=db_name
    )

    return vectorstore


def get_local_vector_db(path: str) -> Any:
    """
    Get a local vector database.

    :param path: Path to the local vector database.
    :type path: str
    :return: Persistent client for the vector database.
    :rtype: Any
    """
    return PersistentClient(path=path)


def get_vector_db_info(vector_store: Any) -> None:
    """
    Print information about the vector database.

    :param vector_store: Vector store to get information from.
    :type vector_store: Any
    """
    collection = vector_store._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)

    print(
        f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
    )


def get_conversation_chain(vectorstore: Any) -> ConversationalRetrievalChain:
    """
    Create a conversation chain using the vector store.

    :param vectorstore: Vector store to use in the conversation chain.
    :type vectorstore: Any
    :return: Conversational retrieval chain.
    :rtype: ConversationalRetrievalChain
    """
    llm = ChatOpenAI(temperature=0.7, model_name=Rag.GPT_MODEL.value)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    return conversation_chain
