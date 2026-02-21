import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Upload PDF files
st.header("My first Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDF file and start asking questions", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
                       

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
                     

    # generating embedding
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)             

    # define the LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=1000,
        openai_api_key=OPENAI_API_KEY
    )

    # define a basic prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that answers questions strictly based on the provided context from the PDF document. "
         "Only answer questions using information from the context below. "
         "If the question cannot be answered using the context, respond with: 'I can only answer questions related to the uploaded PDF document.'\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    #output results
    #chain -> take the question, get relevant document, pass it to the LLM, generate the output
    retriever = vector_store.as_retriever()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # get user questions
    user_question = st.text_input("Type Your question here")

    if user_question:
        # Get response using the modern LCEL chain
        response = chain.invoke(user_question)

        # Display answer
        st.write(response)
