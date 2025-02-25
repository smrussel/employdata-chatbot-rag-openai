import gradio as gr
from langchain_chroma import Chroma
from pathlib import Path
from utils import (
    create_vector_db,
    Rag,
    get_chunks,
    get_conversation_chain,
    get_local_vector_db,
)


def chat(question, history) -> str:
    """
    Get the chat data need for the gradio app

    :param question:
        The question being asked in the chat app.
    :type question: str
    :param history:
        A list of the conversation questions and answers.
    :type history: list
    :return:
        The answer from the current question.
    """

    result = conversation_chain.invoke({"question": question})
    answer = result["answer"]

    return answer


def main():
    gr.ChatInterface(chat, type="messages").launch(inbrowser=True)


if __name__ == "__main__":

    create_new_db = False if Path("vector_db").exists() else True

    if create_new_db:
        folders = Path("knowledge_base").glob("*")
        chunks = get_chunks(folders=folders)
        vector_store = create_vector_db(
            chunks=chunks, db_name=Rag.DB_NAME.value, embeddings=Rag.EMBED_MODEL.value
        )
    else:
        client = get_local_vector_db(path="./vector_db")
        vector_store = Chroma(client=client, embedding_function=Rag.EMBED_MODEL.value)

    conversation_chain = get_conversation_chain(vectorstore=vector_store)

    main()
