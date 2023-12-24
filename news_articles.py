# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:40:38 2023

@author: user
"""
import streamlit as st
import os
from secret_key import openapi_key
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
os.environ['OPENAI_API_KEY']=openapi_key
file_path="vectorstore.pkl"
st.title(" Article Insight Engine ")
st.sidebar.title("Articles URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_url_button = st.sidebar.button("Process URLs")
llm = OpenAI(temperature=0.8, max_tokens=500)
if process_url_button:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vector_index = FAISS.from_documents(docs, embeddings)
    with open(file_path, "wb") as f:
        pickle.dump(vector_index, f)
query = st.text_input('**Question**')
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.write('**Answer**')
            st.write(result["answer"])
            sources = result.get("sources", "")
            if sources:
                st.write('**Sources**')
                sources_list = sources.split("\n")  
                for source in sources_list:
                    st.write(source)

