import os
import streamlit as st
import pickle
import time
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA

load_dotenv()
llm=GooglePalm(google_api_key=os.environ['GOOGLE_API_KEY'],temperature=0)
embeddings = HuggingFaceInstructEmbeddings(model_name= "hkunlp/instructor-large",  model_kwargs={"device": "cpu"},query_instruction="Represent the query for retrieval: ")

vector_db_path="faiss_index"

def create_vector_db():
    loader=CSVLoader(file_path='codebasics_faqs.csv',source_column='prompt')
    data=loader.load()
    vectordb=FAISS.from_documents(embedding=embeddings,documents=data)
    vectordb.save_local(vector_db_path)


def get_qa_chain():
    # Load the vector db from the local folder
    vector_db=FAISS.load_local(vector_db_path,embeddings)

    # Create a retriever for querying the vector database
    retreiver=vector_db.as_retriever()
    prompt_template="""Given the following context and a question, generate an answer based on this contest only.If the answer
    is not found in the context, kindly state "I don't know." Don't try to makeup an answer.

    CONTEXT: {context}
    QUESTION: {question}"""

    PROMPT=PromptTemplate(template=prompt_template,input_variables=['context','question'])
    chain=RetrievalQA.from_chain_type(
    chain_type='stuff',
    retriever=retreiver,
    llm=llm,
    input_key='query',
    return_source_documents=True,
    chain_type_kwargs={'prompt':PROMPT})

    return chain


if __name__=='__main__':
    #create_vector_db()
    chain=get_qa_chain()
    print(chain("do you provide python course?"))
