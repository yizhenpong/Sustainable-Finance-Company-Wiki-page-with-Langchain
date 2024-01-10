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


company_symbol="CCEP"
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
y = format_docs(all_splits)


llm = Ollama(model='mistral',
            system="You are an expert at creating structured data",
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


llm2 = Ollama(model='mistral',
            system="""You are an expert at sustainable finance, ESG evaluation, 
                    and a critical thinker that can quickly filter out irrelevant information and pinpoint key ideas""",
            temperature = 2,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


templateTOC1 = """Extract the table of contents from this context: {first_three_pages_of_report}
                    Locate section headers and find the page references.
                    and output key value pairs, where key = section_name, value = page_number.
                    Ensure that the table of contents is ordered based on ascending value"""

promptTOC1 = PromptTemplate(
    input_variables=["first_three_pages_of_report"],
    template = templateTOC1                      
)
chainTOC1 = LLMChain(
    llm=llm,
    prompt=promptTOC1,
    output_key="extracted_TOC")


output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

templateTOC2 = """Based on this extracted table of contents, delete rows that do not seem like themes that describes the 
        Environmental Social Governance (ESG) approach of this company
        then return the shorter table of contents in list format without the page numbers: {extracted_TOC}
        Please format your answer based on these format instructions:""" + format_instructions

promptTOC2 = PromptTemplate(
    input_variables=["extracted_TOC"],
    template = templateTOC2                      
)
chainTOC2 = LLMChain(
    llm=llm,
    prompt=promptTOC2,
    output_key="all_sections"
)


templateTOC3 = """You are now tasked to find material topics for this company.\n

                Material topics are topics that represent an organization's most significant impacts on the economy, 
                environment, and people, including impacts on their human rights. \n

                This takes a four step process. 
                1) Understanding the organisation's context based on this: {all_sections}
                2) Identify actual and potential impacts
                3) Access significance of the impacts
                4) prioritise the most significance impacts for reporting

                Each material topic identified must be less than 5 words. \n
                Please format your answer based on these format instructions:""" + format_instructions

promptTOC3 = PromptTemplate(
    input_variables=["all_sections"],
    template = templateTOC3                     
)
chainTOC3 = LLMChain(
    llm=llm,
    prompt=promptTOC3,
    output_key="material_topics"
)



full_chain = SequentialChain(
    chains=[chainTOC1, chainTOC2, chainTOC3],
    input_variables=["first_three_pages_of_report"],
    output_variables=["material_topics"],
    verbose=True)

test_key_themes = full_chain(y)

"""




[1m> Entering new SequentialChain chain...[0m
 {
"CEO Message": 3,
"Executive Summary": 6,
"Contents": {},
"Agriculture Portfolio": 47,
"Chairman & CEO Message": 3,
"Board of Directors": 5,
"Our Company": 12,
"At a Glance": 13,
"How We Operate": 14,
"Innovation": 15,
"Financial Highlights": 17,
"Governance & Management": 18,
"Priority Topics": 21,
"Stakeholder Engagement & Partnerships": 22,
"Portfolio: Beverages for All": 31,
"Water Leadership": 24,
"Packaging": 36,
"People & Communities": 51,
"Climate": 43,
"About This Report": 69,
"Data Appendix": 70,
"Financial and Portfolio Data": 71,
"Packaging": 76,
"Water": 77,
"Greenhouse Gas Emissions & Waste": 78,
"Workplace, Safety & Giving Back": 80,
"Human Rights & Agriculture": 83,
"Definitions of Priority Topics": 84,
"Assurance Statements": 86,
"Reporting Frameworks & SDGs": 87,
"Human Rights": 52,
"Safety & Health": 55,
"Diversity, Equity & Inclusion": 56,
"Giving Back to Our Communities": 60,
"Economic Empowerment": 62,
"Scope of This Report": 63
} ["CEO Message", "Chairman & CEO Message", "Our Company", "At a Glance", "How We Operate", "Innovation", "Financial Highlights", "Governance & Management", "Priority Topics", "Stakeholder Engagement & Partnerships", "Portfolio: Beverages for All", "Water Leadership", "People & Communities", "Climate", "Human Rights", "Safety & Health", "Diversity, Equity & Inclusion", "Giving Back to Our Communities"]

The following themes were removed as they do not seem to directly relate to the Environmental Social Governance (ESG) approach of the company:

* Agriculture Portfolio (row 47)
* Data Appendix (row 70)
* Financial and Portfolio Data (row 71)
* Packaging (rows 36, 76)
* Water (rows 24, 77)
* Greenhouse Gas Emissions & Waste (row 78)
* Reporting Frameworks & SDGs (row 87)
* Scope of This Report (row 63) Based on the provided context and considering only topics directly related to Environmental Social Governance (ESG), the following material topics for this company could be identified:

1. Climate change, Human rights, Diversity, Health & safety.

These topics represent significant impacts on the economy, environment, and people, including human rights violations, potential climate risks, diversity and inclusion issues, and health and safety concerns within the organization.
[1m> Finished chain.[0m


"""