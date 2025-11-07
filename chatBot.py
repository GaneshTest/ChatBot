import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#Upload PDF files
st.header("My first chatbot")
with st.sidebar:
     st.title("Your browser")
     file = st.file_uploader("Upload a PDF file start asking questions", type="pdf")


#Extract the text
if file is not None:
    pdf_reader=PdfReader(file)
    text=""
    for page in pdf_reader.pages:
     text+= page.extract_text()


#Break it into chunks
    text_spliter=RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks=text_spliter.split_text(text)
    #generating embedding
    embedding =OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store=FAISS.from_texts(chunks,embedding)
    retriever = vector_store.as_retriever()
    #get user question 
    user_question = st.text_input("Type your question")

    #similarity search 
    if user_question:
        match=vector_store.similarity_search(user_question)
       # st.write(match)
        llm= ChatOpenAI(
            api_key=OPENAI_API_KEY,temperature=0,model="gpt-3.5-turbo" 
            )
         # Prompt for "stuff" chain (context stuffed directly into prompt)
        prompt = ChatPromptTemplate.from_template(
          "Use the following context to answer the question.\n\n"
          "Context:\n{context}\n\n"
          "Question: {question}"
            )

        qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
         | prompt
        | llm
        | StrOutputParser()
         )
       # 5️⃣ Prepare input dictionary
      

        result = qa_chain.invoke(user_question)
        st.write(result)

        