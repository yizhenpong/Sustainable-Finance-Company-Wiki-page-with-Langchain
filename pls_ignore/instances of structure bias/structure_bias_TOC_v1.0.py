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

templateTOC2 = """Based on this extracted table of contents, you are tasked to identify several themes that describes the 
        Environmental Social Governance (ESG) approach of this company. Each theme must be less than 5 words. \n
        Here is the extracted table of contents: {extracted_TOC} \n
        Please format your answer based on these format instructions:""" + format_instructions

promptTOC2 = PromptTemplate(
    input_variables=["extracted_TOC"],
    template = templateTOC1                  
)
chainTOC2 = LLMChain(
    llm=llm2,
    prompt=promptTOC2,
    output_key="key_themes"
)

full_chain = SequentialChain(
    chains=[chainTOC1, chainTOC2],
    input_variables=["first_three_pages_of_report"],
    output_variables=["key_themes"],
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
"Peoples & Communities": 51,
"Climate": 43,
"About This Report": 69,
"Data Appendix": 70,
"Financial and Portfolio Data": 71,
"Packaging": 76,
"Water": 77,
"Greenhouse Gas Emissions & Waste": 78,
"Sustainable Agriculture": 47,
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
"Operations Highlights": 63,
"Asia Pacific": 64,
"Europe, Middle East & Africa": 65,
"Latin America": 66,
"North America": 67,
"Global Ventures/Bottling Investments Group": 68,
"Scope of This Report": 63
} {
"CEOMessage": 3,
"ExecutiveSummary": 6,
"Contents": {},
"AgriculturePortfolio": 47,
"Chairman & CEO Message": 3,
"BoardOfDirectors": 5,
"SustainabilityReportScope": {
"pageNumber": 69,
"sectionName": "About This Report"
},
"Portfolio: BeveragesForAll": 31,
"WaterLeadership": 24,
"Packaging": 36,
"OperationsHighlights": 63,
"AsiaPacific": 64,
"EuropeMiddleEastAfrica": 65,
"LatinAmerica": 66,
"NorthAmerica": 67,
"GlobalVenturesBottlingInvestmentsGroup": 68,
"PriorityTopics": 21,
"StakeholderEngagement & Partnerships": 22,
"SustainableAgriculture": 47,
"GovernanceManagement": 18,
"FinancialHighlights": 17,
"HumanRights": 52,
"Safety & Health": 55,
"DiversityEquityInclusion": 56,
"GivingBackToOurCommunities": 60,
"EconomicEmpowerment": 62,
"Climate": 43
}
[1m> Finished chain.[0m

"""