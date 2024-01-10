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
import time
from time_helper import save_time

'''to get llm'''
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

'''load, chunking, embed and store'''
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import chromadb
from langchain.embeddings import OllamaEmbeddings

'''Retrieving and generation'''
from langchain.output_parsers import CommaSeparatedListOutputParser,StructuredOutputParser, ResponseSchema
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain, RetrievalQA
from operator import itemgetter

############################################################################################################################## 
def run_wiki_gen_base(company_symbol,pageRange = "-1",ToCStatus = False):
    ''' PART 1 - Indexing (load, split, store)'''
    # company_symbol = "CCEP"
    start_time = time.time()
    loader = PyPDFLoader(dh.get_SR_file_path(company_symbol))
    pages = loader.load_and_split()

    #===== to run RAG + ToC ========= start ==========
    if pageRange != "-1":
        try:
            pages = pages[pageRange]
        except:
            return dh.write_output(company_symbol, 
                                   f"No point running RAG + ToC approach, unable to slice pages","Company_info",ToC=ToCStatus)
    #===== to run RAG + ToC ========= end ========== 
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings())
    print("created vectorstore...")
    index_time = time.time()

    print("completed stage 1 of RAG -- Indexing (load, split, store)")

    ############################################################################################################################## 
    '''PART 2 - Retrieving and generation (retrieve, generate)
    note: order is PART A > C outline > C > B
    '''

    '''open source llm - mistral 7b'''
    llm = Ollama(model='mistral',
                 system="You are an expert at sustainable finance and ESG evaluation",
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    # methods to consider hyde, cohere reranker etc
    # https://python.langchain.com/docs/templates/hyde?ref=blog.langchain.dev
    # https://python.langchain.com/docs/integrations/retrievers/cohere-reranker?ref=blog.langchain.dev 

    '''pre generation'''

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    #==================================================
    '''retrieve & generate CONTENT for PART A - GENERAL COMPANY INFO'''

    interested_fields = ["Company name", "ISIN Code", "Headquarters", "founded in", "founded by", "Important People",
                        "Website", "Sustainability Report Name"]
    response_schemas = [
        ResponseSchema(name=interested_fields[0], description=f"Official {interested_fields[0]}"),
        ResponseSchema(name=interested_fields[1], description=f"International Securities Identification Number code, which is a 12-character alphanumeric code"),
        ResponseSchema(name=interested_fields[2], description=f"Official {interested_fields[2]}"),
        ResponseSchema(name=interested_fields[3], description=f"The place where company was {interested_fields[3]}"),
        ResponseSchema(name=interested_fields[4], description=f"Who founded this company?"),
        ResponseSchema(name=interested_fields[5], description=f"Give a list of {interested_fields[5]} who are vital to the compay. Can be CEO or Chief Sustainability Officer"),
        ResponseSchema(name=interested_fields[6], description=f"Main company {interested_fields[6]}"),
        ResponseSchema(name=interested_fields[7], description=f"{interested_fields[6]} may vary from corporate social responsibility report etc")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template0 = """You are tasked to create the general company information section for the sustainable finance Wikipedia page of a company \n
            The company's stock code is:"""+company_symbol+"""\n
            You should provide the answer as key,value pairs for these fields: {fields} \n
            You may use your knowledge and this context to help formulate your answer:  {context}
            If you do not know the answer, please write NA instead of making up an answer.
            Please format your answer based on these format instructions: {format_instructions}"""  
    
    rag_prompt_custom0 = PromptTemplate(
        template = template0,
        input_variables=["context", "fields"],
        partial_variables={"format_instructions": format_instructions})
    
    rag_chain0 = (
        {"context": retriever | format_docs, "fields": RunnablePassthrough()}
        | rag_prompt_custom0
        | llm
        | output_parser
    )

    try:
        company_info_key_val_pairs = rag_chain0.invoke(str(interested_fields))
        dh.write_output(company_symbol,company_info_key_val_pairs,"Company_info", json_type=True, ToC=ToCStatus)
    except:
        # to deal w JSON decode error, invoke for 5 more attempts
        attempts = 5
        for _ in range(attempts):
            try:
                company_info_key_val_pairs = rag_chain0.invoke(str(interested_fields))
                dh.write_output(company_symbol,company_info_key_val_pairs,"Company_info", json_type=True, ToC=ToCStatus)
                break
            except:
                # If really cannot decode, then directly parse out as string and save to txt file
                rag_chain0 = (
                    {"context": retriever | format_docs, "fields": RunnablePassthrough()}
                    | rag_prompt_custom0
                    | llm
                    | StrOutputParser()
                )
                company_info_key_val_pairs = rag_chain0.invoke(str(interested_fields))
                dh.write_output(company_symbol,company_info_key_val_pairs,"Company_info", ToC=ToCStatus)


    print("completed stage 2a of RAG -- General company info")

    #==================================================
    '''retrieve & generate CONTENT for PART C - ESG Approach
    (detailed information is required, sustainability policies, material topics etc)
    '''

    '''(i) retrieve & generate OUTLINE first!!'''   
    if not ToCStatus:
        keywords = ["material topics", "sustainability", "environmental", "sustainable", "supply chain", "human rights"]
        question1 = f"""What is the Environmental Social Governance (ESG) approach of this company? 
                    You may consider the following keywords: {keywords}"""    
        template1 = """You are tasked to identify several themes that answers question {question} \n
                Each theme must be less than 5 words.
                Only use information from this context, if you don't know the answer, just say that you don't: {context} \n
                Please format your answer based on these format instructions: {format_instructions}"""  
        output_parser = CommaSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions()

        rag_prompt_custom1 = PromptTemplate(
            template= template1,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": format_instructions})

        rag_chain1 = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt_custom1
            | llm
            | output_parser
        )
        outline_ls = rag_chain1.invoke(question1)
        dh.write_output(company_symbol,outline_ls,"ESG_approach_outline", list_type=True,ToC=ToCStatus)
        print("completed stage 2c(i) of RAG -- ESG Approach outline")

    if ToCStatus:
        # do some retrieval of Table of Contents
        # then generate the outline based on ToC
        # just make sure that u use CommaSeparatedListOutputParser()
        pass


    #==================================================
    '''(ii) retrieve & generate SPECIFIC POINTS in the outline'''

    '''prompt engineering, structuring prompts'''
    template2 = """You are tasked to write the ESG approach section for the sustainable finance Wikipedia page of a company,
                    specifically on {pointer}. \n
                    Write about three to five paragraphs, reference the source of the context using the page number as much as possible.
                    Only use information from this context to generate text: {context}"""
    
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
    dh.write_output(company_symbol, f"Header: ESG_approach","ESG_approach",header=True,ToC=ToCStatus) 
    for point in outline_ls:
        dh.write_output(company_symbol, f"Sub header: {point}","ESG_approach",header=True,ToC=ToCStatus) 
        article = rag_chain2.invoke(point)
        dh.write_output(company_symbol,article,"ESG_approach",ToC=ToCStatus) 

    print("completed stage 2c(ii) of RAG -- ESG Approach pointers")
    
    

    #==================================================
    '''retrieve & generate CONTENT for PART B -- ESG OVERVIEW:
    - ESG Overview
        - commitment 
        - achievements
        - ESG ratings # in the future...
        - ESG reporting frameworks
    '''

    keywords = ["sustainable commitments", "2050", "2030", "carbon zero", "net zero", "achievements", "reporting frameworks",
                "IFRS", "GRI", "SASB", "SDG", "CDP"]
    question3 = f"""What are their sustainable commitments, achievements, and reporting standards?
                You may consider the following keywords: {keywords}""" 


    template3 = """You are tasked to create the ESG overview section for the sustainable finance Wikipedia page of a company \n
            You should write one to two paragraphs answering this question: {question} \n
            Include just a line that talks about the themes that was covered earlier: {themes} \n 
            if the information does not exist, just say so, you must only use information from this context:  {context}"""  

    rag_prompt_custom3 = PromptTemplate(
        template= template3,
        input_variables=["question", "context"])

    rag_chain3 = (
        {
            "context": itemgetter("question")| retriever,
            "question": itemgetter("question"),
            "themes": itemgetter("themes"),
        }
        | rag_prompt_custom3
        | llm
        | StrOutputParser() 
    )
    ESG_overview = rag_chain3.invoke({"question":question3, "themes":str(outline_ls)})
    dh.write_output(company_symbol,"Header: ESG_overview","ESG_overview", header=True,ToC=ToCStatus)
    dh.write_output(company_symbol,ESG_overview,"ESG_overview",ToC=ToCStatus)

    print("completed stage 2b of RAG -- ESG Overview")
    

    #############################################################################################################################
    end_time = time.time()
    save_time(company_symbol, start_time, index_time, end_time,ToCStatus=ToCStatus)

    #===========================

    print("setting up for new run of RAG -- ESG Overview")
    vectorstore.delete_collection()
    print(f"deleted information in vectorstore for {company_symbol}")

    ##############################################################################################################################


if __name__ == '__main__':
    # run_wiki_gen_base("CCEP")
    pass