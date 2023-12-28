"""
instructions:
- download ollama `https://ollama.ai/download`
    - ollama pull mistral
- pip install langchain
"""

############################################################################################################################## 
'''general'''
import data_helper as dh
# from drafts.ensemble_retriever import retriever_creation

'''to get llm'''
# import torch
# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

'''load, chunking, embed and store'''
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
# from langchain import hub
from langchain.retrievers.multi_query import MultiQueryRetriever

'''interacting with llm'''
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.chains import SimpleSequentialChain, SequentialChain

     
############################################################################################################################## 

'''open source llm - mistral 7b'''
llm = Ollama(model='mistral',
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
# '''open source llm - mistral 7b - YARN'''
# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Mistral-7b-128k")
# model = AutoModelForCausalLM.from_pretrained("NousResearch/Yarn-Mistral-7b-128k",trust_remote_code=True)

# conda install -c nvidia cuda
'''get data'''
company_symbol = "CCEP"
loader = PyPDFLoader(dh.get_SR_file_path(company_symbol))
pages = loader.load_and_split()
pages = pages[:3] # will die if it runs for too long...
print(len(pages))
print(pages[0])
print("REACHED HEREEEEEE /n")

'''split into chunks'''
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(pages)
# vectorstore = Chroma.from_documents(documents=docs, embedding=OllamaEmbeddings())
vectorstore = FAISS.from_documents(docs, OllamaEmbeddings())
# template = """Answer the question based only on the following context:
# {context}
# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)
print("REACHED HEREEEEEE /n")
# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=vectorstore.as_retriever(), llm=llm,
# )
# question = "What are some of the Environmental, Social, and Governance (ESG) or sustainability approaches of the company?"
# unique_docs = retriever_from_llm.get_relevant_documents(query=question)

# print(unique_docs) 
     
# https://python.langchain.com/docs/use_cases/question_answering/  #FOLLOW STEP BY STEP

query_1 = """Table of contents"""
docs_1 = vectorstore.similarity_search(query_1, k = 2) # max k keep at 8
print(len(docs_1))

table_of_contents_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extract table of contents: {text_input}"
)
table_of_contents_chain = LLMChain(llm=llm, prompt=table_of_contents_prompt)
table_of_contents = table_of_contents_chain.run(docs)
dh.write_output(company_symbol,table_of_contents,"table_of_contents")


wiki_outline_prompt = PromptTemplate(
    input_variables=["table_of_contents"],
    template="""
    You are tasked to conduct thorough Environmental, Social, and Governance (ESG) evaluations for companies. 
    Generate a list of headers (No more than 10 headers, can be less as long as it is comprehensive) 
    that explains the companies ESG approach from a policy level. 
    based on this predicted table_of_contents: {table_of_contents}
    """
    # template="""As a highly skilled management consultant with expertise in structuring and summarizing information, 
    # your role is pivotal in conducting thorough Environmental, Social, and Governance (ESG) evaluations for companies. 
    # Critical thinking is paramount, and you are encouraged to transcend the confines of the table of contents to generate your own structure. 
    # Assess each section's relevance to the ESG evaluation, and restructure information accordingly. 
    # Your task involves meticulous assessment and synthesis of key data to craft comprehensive summary pages, ensuring a thorough representation of essential aspects for clients. 
    # Please refer to the provided table of contents to create a well-organized and informative outline :\n\n {table_of_contents}"""
)
wiki_outline_chain = LLMChain(llm=llm, prompt=wiki_outline_prompt)
wiki_outline = wiki_outline_chain.run(table_of_contents)
dh.write_output(company_symbol,wiki_outline,"wiki_outline")



# full_chain = SimpleSequentialChain(chains=[table_of_contents_chain, wiki_outline_chain], verbose=True)
# response = full_chain.run(docs)


# ############################################################################################################################## 
# '''test case'''
# # print(dh.get_all_companies()) # to see all the companies
# # print(dh.get_SR_file_path("CCEP"))

# """
# -	Making use of Retrieval-augmented generation (RAG) 
# -	Also for dealing with long inputs:
# o	Start off from the “table of contents” 
# o	Binary response of whether the section is relevant to sustainability 
# o	Keep the list
# o	Ask it to rank which section might be the most relevant
# -	Create prompt template to get basic company information
# o	Brainstorm on what information might be basic
# o	EG company registered name, ISIN code, revenue, whether they have a sustainability report, link to sustainability report, ESG ratings etc
# -	Create the output parser using the fields that I am interested in. (allows for structured output)
# -	Create more prompt templates to get corresponding information
# o	EG if they have a sustainability report, then what is the ESG reporting framework they are using, policies, themes/material topics, sustainability products and services etc, since when etc
# o	If relevant, search for more webpages that is related to the keywords generated earlier
# -	Build tools to decide what to source
# """

# _input = rag_prompt_custom.format(subject="ice cream flavors")
# output = llm(_input)
# output_parser.parse(output)

# pointers_prompt = PromptTemplate(
#     input_variables=["pointers", "context"],
#     template="Take the following list of pointers and turn them into triples for a knowledge graph:\n\n {facts}"
# )
