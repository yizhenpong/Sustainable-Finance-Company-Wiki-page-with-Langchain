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
from langchain.prompts.few_shot import FewShotPromptTemplate

############################################################################################################################## 
''' PART 3 - Some tests to optimise output'''

'''
IDEA 1:
Instead of map reduce OR selecting topk using vectorstore retriever,
Can i directly predict the outline based on table of contents? [idea]
To do so:
    - Generate the table of contents (see if this is even accurate)
    - Filter strategy
    - run base after filter

Resources:
https://python.langchain.com/docs/modules/chains/document/map_reduce
'''

#==================================================
''' PART 1 - Indexing (load, split, store)'''
company_symbol = "CCEP"
loader = PyPDFLoader(dh.get_SR_file_path(company_symbol))
pages = loader.load_and_split()
pages_ToC_guess = pages[:3] # directly guess that Table Of Contents in first 3 pages
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(pages_ToC_guess)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings())
print("completed stage 1 of RAG -- Indexing (load, split, store)")

#==================================================
'''idea 1a - generating table of contents'''

'''prompt engineering'''
question4 = """table of contents"""

template4 = """You are tasked to create the table of contents with key value pairs 
where key = section_name, value = page_number based on this context: {context}
Do not try to make up what's not in the context.
Locate all the section headers and find the page references. 
Ensure that the table of contents is ordered based on ascending value """
llm = Ollama(model='mistral',
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt_custom4 = PromptTemplate(
    template= template4,
    input_variables=["context"])
'''retriever and rag_chain'''
retrieved_docs = retriever.get_relevant_documents(question4)
print(f"retrieved documents for {question4}")
rag_chain = (
    {"context": retriever | format_docs}
    | rag_prompt_custom4
    | llm
    | StrOutputParser() 
)
TableOfContents = rag_chain.invoke("")
dh.write_output(company_symbol,TableOfContents, "TableOfContents", ToC=True)

print("completed part 3 - idea 1a - generate table of contents")
#==================================================
'''idea 1b - Filter strategy based on ToC'''

'''prompt engineering'''

''' few shot prompting'''
examples = [
  {"TableOfContents": 
"""
1. Introduction 3
2. CEO Welcome 5
3. Leadership Team 5
4. Corporate Overview 6
5. Our Business Model 12
5.1 Key Offerings 13
5.2 Operational Strategies 14
5.3 Tech and Innovation 15
5.4 Financial Insights 17
5.5 Governance Framework 18
6. Strategic Initiatives 21
7.1 Environmental Sustainability 25
7.2 Product Portfolio 31
7.3 Community Engagement 38
7. Sustainable Practices 40
8. Climate Action 43
9. Report Overview 69
10. Appendices and Data 70""",
    "max_pages": 88,
    "answer": "[5:11,21:68]",
  },
  {"TableOfContents": 
"""
1. Executive Summary 3
2. CEO's Message 5
3. Board of Directors 5
4. Company Overview 6
4.1 Mission and Vision 7
4.2 Operating Principles 8
4.3 Financial Highlights 9
5. Strategic Focus 12
6. Sustainability Initiatives 15
6.1 Environmental Impact 17
6.2 Social Responsibility 20
6.3 Governance Practices 23
7. Product Portfolio 28
8. Innovation and Technology 31
9.Financial Performance 34
10.Corporate Governance 37
11.Community Engagement 40
12. Environmental Stewardship 45
13. Conclusion 50
14. Appendices 55 """,
    "max_pages": 88,
    "answer": "[6:7,12:27,40:49]",
  },
   {"TableOfContents": 
"""
1. Executive Summary 3
2. CEO's Message 5
3. Board of Directors 7
4. Company Overview 9
4.1 Mission and Vision 11
4.2 Operating Principles 13
4.3 Financial Highlights 15
5. Sustainability Initiatives 18
5.1 Environmental Impact 20
5.2 Social Responsibility 23
5.3 Governance Practices 26
6. Product Portfolio 29
7. Innovation and Technology 32
8. Environmental Stewardship 35
9. Financial Performance 39
10. Corporate Governance 42
11. Community Engagement 45
12. Conclusion 49
13. Appendices 53""",
    "max_pages": 88,
    "answer": "[18:48]",
  },
]

# print(example_prompt.format(**examples[0]))

template_ToC_filter = """

You are tasked to decide pages that I should filter for if I want to generate a Sustainable Finance Wiki Page for companies.
Some generic information that i would like on that page is mainly:
"Environmental Social Governance policies, data, ESG commitment, achievements, ESG reporting frameworks, general company information"

Return range of pages I should read in code format, based on this table of contents: {TableOfContents}
Structure it in a code format where i can index easily
For example, pages[2:7] which refers to page 3 to 7
Max pages is {max_pages}

Answer about range of page I should read:"""

example_prompt = PromptTemplate(input_variables=["TableOfContents","max_pages", "answer"], template=template_ToC_filter)

filter_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="TableOfContents: {TableOfContents}, max_pages: {max_pages}",
    input_variables=["TableOfContents", "max_pages"]
)
total_num_pages = len(pages)
ToC_filter = filter_prompt.format(max_pages = total_num_pages, TableOfContents = TableOfContents)
print(ToC_filter)

# output_parser = CommaSeparatedListOutputParser()
# format_instructions = output_parser.get_format_instructions()

# filter_prompt = PromptTemplate(
#     template= template_ToC_filter,
#     input_variables=["TableOfContents", "max_pages"])


# ToC_filter = filter_prompt.format(TableOfContents=TableOfContents, max_pages=total_num_pages)


dh.write_output(company_symbol,"ToC_filter","TableOfContents", header=True, ToC=True)
dh.write_output(company_symbol,f"Num pages: {total_num_pages}","TableOfContents", header=True, ToC=True)
dh.write_output(company_symbol,ToC_filter,"TableOfContents", list_type=True, ToC=True)
print("completed part 3 - idea 1b - Filter strategy based on ToC")
