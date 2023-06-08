import os
import re
import sys
import dotenv
import argparse

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# loads .env file with your OPENAI_API_KEY
dotenv.load_dotenv()


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self


def create_prompt(text: str) -> str:
    """Create prompt from text file."""
    joined = " ".join(text)
    prompt = re.sub(r"\[COMPANY\]", COMPANY, joined)
    if PERSONA is not None:
        prompt = re.sub(r"\[PERSONA\]", f"Speak in the style of {PERSONA}.", prompt)
    else:
        prompt = re.sub(r"\[PERSONA\]", "", prompt)
    return prompt


parser = argparse.ArgumentParser()

parser.add_argument("--company", type=str, required=True)
parser.add_argument("--url", type=str, required=True)
parser.add_argument("--collection", type=str, required=True)
parser.add_argument(
    "--db",
    type=str,
    default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "db"),
)
parser.add_argument("--persona", type=str, default=None)
parser.add_argument("--size", type=int, default=1000)
parser.add_argument("--overlap", type=int, default=200)
parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
parser.add_argument("--temp", type=float, default=0, choices=Range(0, 1))
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument(
    "--file",
    type=argparse.FileType("r"),
    default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "prompt.txt"),
)
parser.add_argument("--dry-run", type=str, default="yes")

args = parser.parse_args()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Regex pattern to match a URL
HTTP_URL_PATTERN = r"^http[s]*://.+"

COMPANY = args.company
PERSONA = args.persona
FULL_URL = args.url
DOMAIN = re.sub(r"(http[s]://)(\w+.\w+)(/\w+)?", r"\2", FULL_URL)
COLLECTION = args.collection
DATABASE = args.db
CHUNK_SIZE = args.size
CHUNK_OVERLAP = args.overlap
MODEL = args.model
TEMPERATURE = args.temp
MAX_TOKENS = args.max_tokens
PROMPT_TEMPLATE = create_prompt(args.file.readlines())
DRY_RUN = args.dry_run == "yes"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


if __name__ == "__main__":
    if not DRY_RUN:
        store = Chroma(
            collection_name=COLLECTION,
            persist_directory=DATABASE,
            embedding_function=embeddings,
        )

        # Check if collection is empty
        if not store.get()["documents"]:
            print(
                f"Collection {store._client.get_collection(COLLECTION).name} is empty."
            )
            print("Please add documents to collection first.")
            sys.exit()

        llm = ChatOpenAI(
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            model_name=MODEL,
            temperature=TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=MAX_TOKENS,
        )

        PROMPT = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=store.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
        )

        # Gather user input for the first question to kick off the bot
        question = input("Hi! How can I help you?\n\n")

        # Keep the bot running in a loop to simulate a conversation
        while question:
            print("\n")
            answer = qa(question)
            print("\n\n")
            question = input()

    else:
        print(COMPANY)
        print(PERSONA)
        print(FULL_URL)
        print(DOMAIN)
        print(COLLECTION)
        print(DATABASE)
        print(CHUNK_SIZE)
        print(CHUNK_OVERLAP)
        print(MODEL)
        print(TEMPERATURE)
        print(MAX_TOKENS)
        print(PROMPT_TEMPLATE)
        print(DRY_RUN)

        print("\n\n")

        try:
            store = Chroma(
                collection_name=COLLECTION,
                persist_directory=DATABASE,
                embedding_function=embeddings,
            )

            # Check if collection exists
            if store._client.get_collection(COLLECTION):
                print(f"Collection exists.")
                print(len(store.get()["documents"]))
        except Exception as e:
            print(e)
