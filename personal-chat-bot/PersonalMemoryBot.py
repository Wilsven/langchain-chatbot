# Import necessary modules
import re
from io import BytesIO
from typing import List

import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader


# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    """
    Takes a PDF file object, extracts its text, and cleans it up by
    merging hyphenated words, fixing newlines in the middle of sentences,
    and removing multiple newlines.

    Parameters
    ----------
    file: BytesIO
        PDF file pbject.

    Returns
    -------
    output: List[str]
        List of text where each element is a page in the PDF.
    """
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences (use negative look behind and look ahead)
        text = re.sub(r"(?<!\n\s)\n(?!\n\s)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)

    return output


# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """
    Converts a string or list of strings (e.g., the output of parse_pdf()) to a list
    of LangChain Document objects. Each Document represents a chunk of text
    of up to 4000 characters (configurable). The Document objects also store
    metadata such as the page number and chunk number from which the text was extracted.

    Parameters
    ----------
    text: str
        String of text to be chunked and converted into LangChain Document objects.

    Returns
    -------
    pages: List[Document]
        List of LangChain Document objects which also store meta data such as the page
        number and chunk number from which the text was extracted.
    """
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    pages = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources to metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            pages.append(doc)

    return pages


# Define a function for the embeddings
@st.cache_data
def test_embed(_pages: List[Document], openai_api_key: str) -> VectorStore:
    """
    Uses the LangChain OpenAI embeddings by indexing the documents using
    the FAISS vector store. The resulting vector store is returned.

    Parameters
    ----------
    pages: List[Document]
        List of Documents to index into the Vector Store.

    openai_api_key: str
        OpenAI API key.

    Returns
    -------
    vector_store: VectorStore
        Resulting Vector Store is returned.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Indexing
    # Save in a Vector Store
    with st.spinner("It's indexing..."):
        vector_store = FAISS.from_documents(_pages, embeddings)
    st.success("Embeddings done!", icon="✅")

    return vector_store


# Set up the Streamlit app
st.title("🤖 Personalized Bot with Memory 🧠")
st.markdown(
    """
        #### 💬 Chat with your PDF files 📄 with `Conversational Buffer Memory`
        > *powered by [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [OpenAI]('https://platform.openai.com/docs/models/gpt-3-5') + [DataButton](https://www.databutton.io/)*
        ----
    """
)

st.markdown(
    """
    `openai`
    `langchain`
    `tiktoken`
    `pypdf`
    `faiss-cpu`
    
    ---------
    """
)

# Set the sidebar
st.sidebar.markdown(
    """
    ### Steps:
    1. Upload PDF File 📄
    2. Enter Your Secret Key for Embeddings 🔐
    3. Perform Q&A 🙋🏻‍♂️
    
    **Note: File content and API key are not stored in any form.**
    """
)

# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])

if uploaded_file:
    name_of_file = uploaded_file.name
    output = parse_pdf(uploaded_file)
    pages = text_to_docs(output)
    if pages:
        # Allow the user to select a page and view its content
        with st.expander("Show Page Content", expanded=False):
            page_sel = st.number_input(
                label="Select Page", min_value=1, max_value=len(pages), step=1
            )
            pages[page_sel - 1]

        # Allow the user to enter an OpenAI API key
        openai_api_key = st.text_input(
            "**Enter OpenAI API Key**",
            type="password",
            placeholder="sk-",
            help="https://platform.openai.com/account/api-keys",
        )

        if openai_api_key:
            # Test the embeddings and save the index in a vector database
            vector_store = test_embed(pages, openai_api_key=openai_api_key)
            # Set up the question-answering system
            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(openai_api_key=openai_api_key),
                chain_type="map_reduce",
                retriever=vector_store.as_retriever(),
            )
            # Set up the conversational agent
            tools = [
                Tool(
                    name="State of Union QA System",
                    func=qa.run,
                    description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
                )
            ]
            prefix = """
            Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
            You have access to a single tool:
            """
            suffix = """
            Begin! 
            
            {chat_history}
            =================
            Question: {input}
            =================
            {agent_scratchpad}
            """

            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )

            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history"
                )

            llm_chain = LLMChain(
                llm=OpenAI(
                    temperature=0,
                    openai_api_key=openai_api_key,
                    model_name="gpt-3.5-turbo",
                ),
                prompt=prompt,
            )
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
            )

            # Allow the user to enter a query and generate a response
            query = st.text_input(
                "**What's on your mind?**",
                placeholder=f"Ask me anything from {name_of_file}",
            )

            if query:
                with st.spinner(f"Generating Answer to your Query : `{query}`"):
                    res = agent_chain.run(query)
                    st.info(res, icon="🤖")

            # Allow the user to view the conversation history and other information stored in the agent's memory
            with st.expander("History/Memory"):
                st.session_state.memory
