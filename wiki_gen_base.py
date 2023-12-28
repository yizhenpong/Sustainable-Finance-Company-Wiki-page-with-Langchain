"""
instructions:
- download ollama `https://ollama.ai/download`
    - ollama pull mistral
- pip install langchain

RAG
- PART 1) Indexing (load, split, store)
- PART 2) Retrieving and generation (retrieve, generate)

Dealing with long inputs - selecting top k, possibly using a fine tuned YARN Mistral 7B model
Dealing with long outputs - use an outline and tackle from top down
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
from langchain.output_parsers import CommaSeparatedListOutputParser,StructuredOutputParser, ResponseSchema
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain

############################################################################################################################## 
def run_wiki_gen_base(company_symbol):
    ''' PART 1 - Indexing (load, split, store)'''
    # company_symbol = "CCEP"
    loader = PyPDFLoader(dh.get_SR_file_path(company_symbol))
    pages = loader.load_and_split()
    # pages = pages[:3]
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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    #==================================================
    '''retrieve & generate CONTENT for PART A - GENERAL COMPANY INFO'''

    interested_fields = ["Company name", "ISIN Code", "Headquarters", "founded in", "founded by", "Important People",
                        "Website", "Sustainability Report Name"]

    response_schemas = [
        ResponseSchema(name=interested_fields[0], description=f"Official {interested_fields[0]}"),
        ResponseSchema(name=interested_fields[1], description=f"Official {interested_fields[1]}"),
        ResponseSchema(name=interested_fields[2], description=f"Official {interested_fields[2]}"),
        ResponseSchema(name=interested_fields[3], description=f"The place where company was {interested_fields[3]}"),
        ResponseSchema(name=interested_fields[4], description=f"Who founded this company?"),
        ResponseSchema(name=interested_fields[5], description=f"Give a list of {interested_fields[5]} who are vital to the compay. Can be CEO or Chief Sustainability Officer"),
        ResponseSchema(name=interested_fields[6], description=f"Main company {interested_fields[6]}"),
        ResponseSchema(name=interested_fields[7], description=f"{interested_fields[6]} may vary from corporate social responsibility report etc")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template0 = """You are tasked to create a Sustainable Finance Wikipedia page for a company 
            and in particular the section on general company information. 
            Find the information for the list of fields and provide your answer in key,value pairs: {fields}
            Only use information from this context, 
            if you don't know the answer, use the value NA and do not make up an answer:  {context}"""  

    rag_prompt_custom0 = PromptTemplate(
        template= template0,
        input_variables=["context", "fields"])

    '''retriever and rag_chain'''
    # retrieved_docs = retriever.get_relevant_documents(str(interested_fields))
    # print(f"retrieved documents for {interested_fields}")
    rag_chain0 = (
        {"context": retriever | format_docs, "fields": RunnablePassthrough()}
        | rag_prompt_custom0
        | llm
        | StrOutputParser() #CommaSeparatedListOutputParser() # StrOutputParser() -- rerun for this....?
    )
    company_info_key_val_pairs = rag_chain0.invoke(str(interested_fields))
    dh.write_output(company_symbol,"Header: Company_info","Company_info", header=True)
    dh.write_output(company_symbol,company_info_key_val_pairs,"Company_info")
    print("completed stage 2a of RAG -- General company info")

    #==================================================
    '''retrieve & generate CONTENT for PART C - ESG Approach
    (detailed information is required, sustainability policies, material topics etc)
    '''

    '''(i) retrieve & generate OUTLINE first!!'''   

    '''prompt engineering'''
    question1 = """What is the Environmental Social Governance (ESG) approach of this company? 
                You may consider material topics, sustainability or environmental approach of companies.
                This may cover sustainable practices and policies, from products to supply chain, and also covers human rights"""    
    template1 = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Come up with list of pointers for the question and keep the pointers as concise as possible
    {context}
    Question: {question}
    Remember to structure your answer using these format instructions: {format_instructions}
    Helpful Answer:"""

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()

    rag_prompt_custom1 = PromptTemplate(
        template= template1,
        input_variables=["context", "question"],
        partial_variables={"format_instructions": format_instructions})
    '''retriever and rag_chain'''
    # retrieved_docs = retriever.get_relevant_documents(question)
    # print(f"retrieved documents for {question}")
    rag_chain1 = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_custom1
        | llm
        | CommaSeparatedListOutputParser()
    )
    # outline_ls = rag_chain.invoke("What is the Environmental Social Governance (ESG) approach of this company in list format?")
    outline_ls = rag_chain1.invoke(question1)
    dh.write_output(company_symbol,outline_ls,"ESG_approach_outline", list_type=True)
    print("completed stage 2c(i) of RAG -- ESG Approach outline")

    #==================================================
    '''(ii) retrieve & generate SPECIFIC POINTS in the outline'''

    '''prompt engineering, structuring prompts'''
    template2 = """ You are tasked to write a section of an article about the Environmental Social Governance (ESG) approach of a company.
    Specifically focusing on this aspect of the ESG Approach: \n {pointer}
    Do not hallucinate or create your own content to complete the story. 
    There is also no need to start with "Title:", directly produce the contents.
    Use only the following pieces of context to generate the text. {context}

    Text:"""
    rag_prompt_custom2 = PromptTemplate(
        template= template2,
        input_variables=["context", "pointer"])

    '''rag_chain'''
    rag_chain2 = (
        {"context": retriever | format_docs, "pointer": RunnablePassthrough()}
        | rag_prompt_custom2
        | llm
        | StrOutputParser() 
    )

    '''confirm that we have the outline'''
    print(outline_ls)
    print("len(outline_ls)")
    print(len(outline_ls))

    '''generation'''
    for point in outline_ls:
        dh.write_output(company_symbol, f"/n Sub header: {point}","ESG_approach",header=True) 
        # retrieved_docs = retriever.get_relevant_documents(point)
        # print(f"retrieved documents for {point}")
        article = rag_chain2.invoke(point)
        dh.write_output(company_symbol,article,"ESG_approach") 

    print("completed stage 2c(ii) of RAG -- ESG Approach pointers")

    #==================================================
    '''retrieve & generate CONTENT for PART B -- ESG OVERVIEW:
    - ESG Overview
        - commitment 
        - achievements
        - ESG ratings
        - ESG reporting frameworks
    '''

    '''prompt engineering'''

    template3 = """You are tasked to give a Environmental Social Governance (ESG) Overview of this company. 
    You may wish to cover their sustainable commitments like carbon zero or net zero, achievements, what are their current ESG reporting frameworks, their ESG ratings.
    You shall not include this list of ESG approaches as they have been covered already: {ESG_approaches} 
    You must not hallucinate and come up with your own content, a highly accurate answer that can be referenced to a page number is much preferred.
    Use only context from here: {context}"""

    rag_prompt_custom3 = PromptTemplate(
        template= template3,
        input_variables=["ESG_approaches", "context"])
    '''retriever and rag_chain'''
    # retrieved_docs = retriever.get_relevant_documents(question3)
    # print(f"retrieved documents for {question3}")
    rag_chain = (
        {"context": retriever | format_docs, "ESG_approaches": RunnablePassthrough()}
        | rag_prompt_custom3
        | llm
        | StrOutputParser() 
    )
    ESG_overview = rag_chain.invoke(outline_ls)
    dh.write_output(company_symbol,"Header: ESG_overview","ESG_overview", header=True)
    dh.write_output(company_symbol,ESG_overview,"ESG_overview")

    print("completed stage 2b of RAG -- ESG Overview")

    print("setting up for new run of RAG -- ESG Overview")
    vectorstore.delete_collection()
    print(f"deleted information in vectorstore for {company_symbol}")

    print(vectorstore._collection.count())
    if vectorstore._collection.count() == 0:
        print(f"double confirmed nothing left in vectorstore, ready for next run")


    ##############################################################################################################################


# run_wiki_gen_base("CCEP")
# 