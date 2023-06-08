# https://github.com/jina-ai/langchain-serve

import os
from dotenv import load_dotenv
import re

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from lcserve import serving


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Specify path to vector store
persist_directory = "../chatbot/db"


@serving
def ask(
    question: str, company: str, collection_name: str, persist_directory: str
) -> str:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    print(persist_directory)
    store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    llm = ChatOpenAI(
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=512,
    )

    sales_template = """
    As a customer marketing bot, your goal is to provide accurate and helpful information about [COMPANY].
    You should answer user inquiries based on the context provided and avoid making up answers.
    If you don't know the answer, simply state that you don't know.
    Remember to provide relevant information about how [COMPANY] can assist the user through its services, strengths and benefits.

    {context}
    =========
    Question: {question}
    """

    sales_template = re.sub(r"\[\w+\]", company, sales_template)

    SALES_PROMPT = PromptTemplate(
        template=sales_template, input_variables=["context", "question"]
    )

    print(SALES_PROMPT.template)

    sales_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(),
        chain_type_kwargs={"prompt": SALES_PROMPT},
    )

    return sales_qa.run(question)


if __name__ == "__main__":
    ask(
        "What does the company do?",
        company="Contact Out",
        collection_name="cto",
        persist_directory=persist_directory,
    )
