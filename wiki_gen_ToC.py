"""
instructions:
- download ollama `https://ollama.ai/download`
    - ollama pull mistral
- pip install langchain

RAG
- PART 1) Indexing (load, split, store)
- PART 2) Retrieving and generation (retrieve, generate)

Dealing with long inputs - selecting top k, possibly using a fine tuned YARN Mistral 7B model, ToC model(Own thoughts)
Dealing with long outputs - use an outline and tackle from top down

''' PART 3 - Some tests to optimise output'''
Introducing ToC - Using table of contents as a filter to deal with long inputs

Instead of map reduce OR selecting topk using vectorstore retriever,
Can i directly predict the outline based on table of contents? [idea]
To do so:
    - Generate the table of contents (see if this is even accurate)
    - Filter strategy
    - Generate outline based on ToC

Resources:
https://python.langchain.com/docs/modules/chains/document/map_reduce
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
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chains import LLMChain

############################################################################################################################## 
''' PART 3 - Some tests to optimise output'''

'''
Instead of map reduce OR selecting topk using vectorstore retriever,
Can i directly predict the outline based on table of contents? [idea]
To do so:
    - Generate the table of contents (see if this is even accurate)
    - Filter strategy
    - run base after filter

Resources:
https://python.langchain.com/docs/modules/chains/document/map_reduce
'''

############################################################################################################################# 


def gen_ToC(company_symbol):
    ''' PART 1 - Indexing (load, split, store)'''
    loader = PyPDFLoader(dh.get_SR_file_path(company_symbol))
    pages = loader.load_and_split()
    pages = pages[:3]  # intentional - guess that table of contents is at the front
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(pages)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings())
    print("created vectorstore...")
    print("completed stage 1 of ToC -- Indexing (load, split, store)")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    ############################################################################################################################# 
    '''generating table of contents'''

    template4 = """You are tasked to extract the table of contents with key value pairs 
                    where key = section_name, value = page_number.
                    Only use information from this context to generate text: {context}
                    Locate all the major section headers and find the page references. 
                    Ensure that the table of contents is ordered based on ascending value"""
    
    llm = Ollama(model='mistral',
                system="You are an expert at sustainable finance and ESG evaluation",
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    
    retriever = vectorstore.as_retriever()
    rag_prompt_custom4 = PromptTemplate(
        template= template4,
        input_variables=["context"])
    rag_chain4 = (
        {"context": retriever | format_docs}
        | rag_prompt_custom4
        | llm
        | StrOutputParser() 
    )

    ToC_ascending_prompt = PromptTemplate(input_variables=["table_of_contents"],
        template="""Given a table of contents in key,value pairs where key = section_name, value = page_number,
                    order it by the page number in ascending order.
                    This is the table of contents: {table_of_contents}""")

    ToC_ascending_chain = LLMChain(llm=llm, prompt=ToC_ascending_prompt)
    TableOfContents_unclean = rag_chain4.invoke("")
    dh.write_output(company_symbol,TableOfContents_unclean,"TableOfContents_unordered",ToC=True)
    TableOfContents = SimpleSequentialChain(chains=[rag_chain4, ToC_ascending_chain], verbose=True)
    dh.write_output(company_symbol,TableOfContents, "TableOfContents", ToC=True)
    print(f"table of contents of saved!!!")
    print("completed generation of ToC")




# gen_ToC("CCEP")

# def filter_from_ToC(company_symbol):


    # ############################################################################################################################## 
    # '''idea 1b - Filter strategy based on ToC'''

    # '''prompt engineering'''

    # ''' few shot prompting'''

    # examples = [
    # {"TableOfContents": 
    # """
    # 1. Introduction 3
    # 2. CEO Welcome 5
    # 3. Leadership Team 5
    # 4. Corporate Overview 6
    # 5. Our Business Model 12
    # 5.1 Key Offerings 13
    # 5.2 Operational Strategies 14
    # 5.3 Tech and Innovation 15
    # 5.4 Financial Insights 17
    # 5.5 Governance Framework 18
    # 6. Strategic Initiatives 21
    # 7.1 Environmental Sustainability 25
    # 7.2 Product Portfolio 31
    # 7.3 Community Engagement 38
    # 7. Sustainable Practices 40
    # 8. Climate Action 43
    # 9. Report Overview 69
    # 10. Appendices and Data 70""",
    #     "max_pages": 88,
    #     "answer": "[5:11,21:68]",
    # },
    # {"TableOfContents": 
    # """
    # 1. Executive Summary 3
    # 2. CEO's Message 5
    # 3. Board of Directors 5
    # 4. Company Overview 6
    # 4.1 Mission and Vision 7
    # 4.2 Operating Principles 8
    # 4.3 Financial Highlights 9
    # 5. Strategic Focus 12
    # 6. Sustainability Initiatives 15
    # 6.1 Environmental Impact 17
    # 6.2 Social Responsibility 20
    # 6.3 Governance Practices 23
    # 7. Product Portfolio 28
    # 8. Innovation and Technology 31
    # 9.Financial Performance 34
    # 10.Corporate Governance 37
    # 11.Community Engagement 40
    # 12. Environmental Stewardship 45
    # 13. Conclusion 50
    # 14. Appendices 55 """,
    #     "max_pages": 88,
    #     "answer": "[6:7,12:27,40:49]",
    # },
    # {"TableOfContents": 
    # """
    # 1. Executive Summary 3
    # 2. CEO's Message 5
    # 3. Board of Directors 7
    # 4. Company Overview 9
    # 4.1 Mission and Vision 11
    # 4.2 Operating Principles 13
    # 4.3 Financial Highlights 15
    # 5. Sustainability Initiatives 18
    # 5.1 Environmental Impact 20
    # 5.2 Social Responsibility 23
    # 5.3 Governance Practices 26
    # 6. Product Portfolio 29
    # 7. Innovation and Technology 32
    # 8. Environmental Stewardship 35
    # 9. Financial Performance 39
    # 10. Corporate Governance 42
    # 11. Community Engagement 45
    # 12. Conclusion 49
    # 13. Appendices 53""",
    #     "max_pages": 88,
    #     "answer": "[18:48]",
    # },
    # ]


    # example_formatter_template = """
    # TableOfContents: {TableOfContents}
    # max_pages: {max_pages}
    # answer: {answer}\n
    # """

    # query = """

    # You are tasked to decide pages that I should filter for if I want to generate a Sustainable Finance Wiki Page for companies.
    # Some generic information that i would like on that page is mainly:
    # "Environmental Social Governance policies, data, ESG commitment, achievements, ESG reporting frameworks, general company information"

    # Return range of pages I should read in code format, based on this table of contents: {TableOfContents}
    # Structure it in a code format where i can index easily
    # For example, pages[2:7] which refers to page 3 to 7
    # Max pages is {max_pages}

    # Answer about the range of page I should read:"""

    # example_prompt = PromptTemplate(input_variables=["TableOfContents","max_pages", "answer"], template=example_formatter_template)

    # few_shot_prompt = FewShotPromptTemplate(
    #     examples=examples,
    #     example_prompt=example_prompt,
    #     prefix = query,
    #     suffix="TableOfContents: {TableOfContents} \n max_pages: {max_pages}",
    #     input_variables=["TableOfContents", "max_pages"],
    #     example_separator="\n\n",
    # )
    # total_num_pages = len(pages)
    # ToC_filter_prompt = few_shot_prompt.format(TableOfContents = TableOfContents, max_pages = total_num_pages)
    # chain = LLMChain(llm=llm, prompt=few_shot_prompt)
    # ToC_filter_messy = chain.run({"TableOfContents": TableOfContents , "max_pages": total_num_pages})
    # ToC_filter = dh.get_filtered_pages(ToC_filter_messy)


    # dh.write_output(company_symbol," \n Header: ToC_filter","TableOfContents", header=True, ToC=True)
    # dh.write_output(company_symbol,f" \n Num pages: {total_num_pages}","TableOfContents", ToC=True)
    # dh.write_output(company_symbol,ToC_filter,"TableOfContents", ToC=True)
    # print("completed part 3 - idea 1b - Filter strategy based on ToC")

    # ############################################################################################################################## 
    # llm2 = Ollama(model='mistral', 
    #             system="You are an expert at sustainable finance and ESG evaluation",
    #             temperature=2, #default is 0.8
    #                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    
    # template5 = """Explain the Environmental Social Governance (ESG) approach of this company.
    #     You may consider material topics, sustainability or environmental approach of companies.
    #     This may cover sustainable practices and policies, from products to supply chain, and also covers human rights.
        
    #     Only use this table of contents to get an overview and organise your answer based on the themes you found.
    #     Do not assume the company has any other policies, other than whatever is mentioned in the table of contents.
        
    #     Try to organise the themes creatively and do not be confined to the table of contents structure.
    #     Come up with list of pointers  and keep the pointers as concise as possible
    #     Remember to structure your answer using these format instructions: {format_instructions}

    #     Here is the table of contents: {TableOfContents}
    #     Helpful Answer:"""    

    # output_parser = CommaSeparatedListOutputParser()
    # format_instructions = output_parser.get_format_instructions()

    # rag_ToC_prompt = PromptTemplate(
    #     template =template5,
    #     input_variables=["TableOfContents"],
    #     partial_variables={"format_instructions": format_instructions})

    # chain = LLMChain(llm=llm2, prompt=rag_ToC_prompt)
    # rag_ToC_outline = chain.run(TableOfContents=TableOfContents)

    # dh.write_output(company_symbol,f"Header: rag_ToC_outline","ToC_only_ESG_approach_outline", ToC=True)
    # dh.write_output(company_symbol,rag_ToC_outline,"ToC_only_ESG_approach_outline", ToC=True)
    # print("completed part 3 - idea 1c - Filter strategy based on ToC")

    # ############################################################################################################################## 
    # print("setting up for new run of RAG -- ESG Overview")
    # vectorstore.delete_collection()
    # print(f"deleted information in vectorstore for {company_symbol}")

    # return ToC_filter #list of pages to be filtered












# output_parser = CommaSeparatedListOutputParser()
# format_instructions = output_parser.get_format_instructions()

# filter_prompt = PromptTemplate(
#     template= template_ToC_filter,
#     input_variables=["TableOfContents", "max_pages"])


# ToC_filter = filter_prompt.format(TableOfContents=TableOfContents, max_pages=total_num_pages)


'''discarded few shot prompting'''

# examples = [
#   {"TableOfContents": 
# """
# 1. Introduction 3
# 2. CEO Welcome 5
# 3. Leadership Team 5
# 4. Corporate Overview 6
# 5. Our Business Model 12
# 6. Strategic Initiatives 21
# 7 Environmental Sustainability 25
# 8. Sustainable Practices 40
# 9. Climate Action 43
# 10. Report Overview 69
# 11. Appendices and Data 70""",
#     "max_pages": 88,
#     "answer": """
# How many topics are there in the table of contents?: 10
# What is the page range of this topic?: 3-5
# Is the first topic relevant to sustainablity approach?: No
# Is there a next topic? Yes

# What is the page range of next topic?: 5-5
# Is this new topic relevant to sustainablity approach?: No
# Is there a next topic? Yes

# What is the page range of next topic?: 5-5
# Is this new topic relevant to sustainablity approach?: No
# Is there a next topic? Yes

# What is the page range of next topic?: 6-12
# Is this new topic relevant to sustainablity approach?: No
# Is there a next topic? Yes

# What is the page range of next topic?: 12-21
# Is this new topic relevant to sustainablity approach?: No
# Is there a next topic? Yes

# What is the page range of next topic?: 21-25
# Is this new topic relevant to sustainablity approach?: Yes
# What is the page range I need to read in code format? [21:25]
# Is there a next topic? Yes

# What is the page range of next topic?: 25-40
# Is this new topic relevant to sustainablity approach?: Yes
# What is the page range I need to read in code list of integers format? [21:25, 25-40]
# Is there a next topic? Yes

# What is the page range of next topic?: 40-43
# Is this new topic relevant to sustainablity approach?: Yes
# What is the page range I need to read in code list of integers format? [21:25, 25-40, 40-43]
# Is there a next topic? Yes

# What is the page range of next topic?: 43-69
# Is this new topic relevant to sustainablity approach?: Yes
# What is the page range I need to read in code list of integers format? [21:25, 25-40, 40-43, 43-69]
# Is there a next topic? Yes

# What is the page range of next topic?: 69-70
# Is this new topic relevant to sustainablity approach?: No
# Is there a next topic? Yes

# What is the page range of next topic?: 70-88
# Is this new topic relevant to sustainablity approach?: No
# Is there a next topic? No

# So final answer for page range I need to read in code list of integers format: [21:25, 25-40, 40-43, 43-69]""",
#   },
#   {"TableOfContents": 
# """
# Executive Summary 3
# CEO's Message 5
# Sustainability Initiatives 15
# Social Responsibility 20
# Governance Practices 23
# Community Engagement 40
# Environmental Stewardship 45
# Conclusion 50
# Appendices 55 """,
#     "max_pages": 75,
#     "answer": 
#     """
# What is the page range of next topic?: 3-5
# Is this new topic relevant to sustainablity approach?: No
# Is there a next topic? Yes

# What is the page range of next topic?: 5-15
# Is this new topic relevant to sustainablity approach?: No
# Is there a next topic? Yes

# What is the page range of next topic?: 15-20
# Is this new topic relevant to sustainablity approach?: Yes
# What is the page range I need to read in code list of integers format? [15:20]
# Is there a next topic? Yes

# What is the page range of next topic?: 20-23
# Is this new topic relevant to sustainablity approach?: Yes
# What is the page range I need to read in code list of integers format? [15:20, 20-23]
# Is there a next topic? Yes

# What is the page range of next topic?: 23-40
# Is this new topic relevant to sustainablity approach?: Yes
# What is the page range I need to read in code list of integers format? [15:20,20-23, 23-40]
# Is there a next topic? Yes

# What is the page range of next topic?: 40-45
# Is this new topic relevant to sustainablity approach?: Yes
# What is the page range I need to read in code list of integers format? [15:20,20-23, 23-40, 40:45]
# Is there a next topic? Yes

# What is the page range of next topic?: 45-50
# Is this new topic relevant to sustainablity approach?: Yes
# What is the page range I need to read in code list of integers format? [15:20,20-23, 23-40, 40:45, 45:50]
# Is there a next topic? Yes

# What is the page range of next topic?: 50-55
# Is this new topic relevant to sustainablity approach?: No
# Is there a next topic? No

# What is the page range of next topic?: 55-75
# Is this new topic relevant to sustainablity approach?: No
# Is there a next topic? No

# So final answer for page range I need to read in code list of integers format: [15:20,20-23, 23-40, 40:45, 45:50]""",
#   },
# ]

