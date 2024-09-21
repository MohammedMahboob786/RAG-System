import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: pink;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Constants
CHROMA_PATH = "DataBase"
MODEL_NAME = "thenlper/gte-small"

key = os.getenv("OPENAI_API_KEY")

if key is None:
    raise Exception("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")


def initialize_chroma():
    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return Chroma(
        collection_name="vector_database",
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH
    )

def create_rag_chain(db):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    SYS_MESSAGE = """You are an AI assistant specialized in answering questions using
    retrieved contextual information. Your task is to carefully analyze the provided context
    and deliver a precise, accurate response.
    
    -> Use only the information given in the retrieved context to formulate your answer.
    -> If the context does not provide enough information to answer the question,
    state clearly that you do not know.
    -> Ensure your response is concise, clear."""
    
    system_message = SystemMessage(content=SYS_MESSAGE)
    
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}
    Answer the question based on the above context: {question}.
    Provide a detailed answer.
    Don't justify your answers.
    Don't give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    """
    
    prompt_template = ChatPromptTemplate.from_messages([system_message, PROMPT_TEMPLATE])
    chat_model = ChatOpenAI(openai_api_key=key, model="gpt-4o-mini")
    parser = StrOutputParser()
    
    return {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_template | chat_model | parser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def process_uploaded_file(uploaded_file):
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read())
    
    loader = PyMuPDFLoader(temp_file)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    
    return chunks

def display_previous_conversation():
    """Display the conversation history"""
    if "messages" in st.session_state:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

def main():
    st.title("RAG Conversational App")
    # Include custom CSS
    st.markdown("""
    <style>
    .stChatMessage.st-emotion-cache-1c7y2kd.eeusbqq4 {
        text-align: right;
        display: flex;
        flex-direction: row-reverse;
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    # Initialize Chroma and RAG chain
    db = initialize_chroma()
    rag_chain = create_rag_chain(db)
    
    # Layout: File uploader at the top, conversation below it
    st.sidebar.title("Upload PDF File")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file:
        chunks = process_uploaded_file(uploaded_file)
        db.add_documents(chunks)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Display previous conversation in the main area
    
    display_previous_conversation()

    # Get user query and provide responses
    prompt = st.chat_input("Ask a question...")
    if prompt:
        # Add the user's message to the session state
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get the assistant's response
        try:
            answer = st.chat_message("assistant").write_stream(rag_chain.stream(prompt))
            st.session_state["messages"].append({"role": "assistant", "content": answer})
                
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

                

