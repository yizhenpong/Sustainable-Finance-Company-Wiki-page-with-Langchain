"""
instructions:
- download ollama `https://ollama.ai/download`
    - ollama pull mistral
- pip install langchain

RAG
- Indexing (load, split, store)
- Retrieving and generation (retrieve, generate)
"""

############################################################################################################################## 
'''general'''
import data_helper as dh

'''to get llm'''
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

'''load, chunking, embed and store'''
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

'''Retrieving and generation'''
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema import StrOutputParser
# from langchain.output_parsers.pydantic import PydanticOutputParser
# from typing import List
# from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import SimpleSequentialChain, SequentialChain

############################################################################################################################## 

''' PART 1 - Indexing (load, split, store)'''
company_symbol = "CCEP"
loader = PyPDFLoader(dh.get_SR_file_path(company_symbol))
pages = loader.load_and_split()
pages = pages[:3]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(pages)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings())

print("completed stage 1 of RAG -- Indexing (load, split, store)")

############################################################################################################################## 
'''PART 2 - Retrieving and generation (retrieve, generate)'''

'''open source llm - mistral 7b'''
llm = Ollama(model='mistral',
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

'''pre generation'''
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

#==================================================

'''retrieve & generate OUTLINE'''

question = """What is the Environmental Social Governance (ESG) approach of this company? 
            You may consider material topics, sustainability or environmental approach of companies.
            This may cover sustainable practices and policies, from products to supply chain, and also covers human rights"""

retrieved_docs = retriever.get_relevant_documents(question)
print(f"retrieved documents for {question}")
    
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Come up with list of pointers for the question and keep the pointers as concise as possible
{context}
Question: {question}
Remember to structure your answer using these format instructions: {format_instructions}
Helpful Answer:"""

rag_prompt_custom = PromptTemplate(
    template= template,
    input_variables=["context", "question"],
    partial_variables={"format_instructions": format_instructions})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt_custom
    | llm
    | CommaSeparatedListOutputParser()
)

outline_ls = rag_chain.invoke("What is the Environmental Social Governance (ESG) approach of this company?")
dh.write_output(company_symbol,outline_ls,"ESG_approach_outline", list_type=True)

print("completed stage 2a of RAG -- retrieve & generate OUTLINE")

#==================================================
'''retrieve & generate SPECIFIC POINTS in the outline'''

template2 = """ You are tasked to write an article about the Environmental Social Governance (ESG) approach of a company.
Specifically focusing on this aspect of the ESG Approach: \n {pointer}
Use only the following pieces of context to generate the text. {context}

Text:"""

rag_prompt_custom2 = PromptTemplate(
    template= template2,
    input_variables=["context", "pointer"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain2 = (
    {"context": retriever | format_docs, "pointer": RunnablePassthrough()}
    | rag_prompt_custom2
    | llm
    | StrOutputParser() 
)

print("len(outline_ls)")
print(len(outline_ls))

for point in outline_ls:
    dh.write_output(company_symbol,point,"ESG_approach",header=True) 
    retrieved_docs = retriever.get_relevant_documents(point)
    print(f"retrieved documents for {point}")
    article = rag_chain2.invoke("What is generated sub-section?")
    dh.write_output(company_symbol,article,"ESG_approach") 

print("completed stage 2b of RAG -- retrieve & generate POINTERS")
