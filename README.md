# Sustainable-Finance-Company-Wiki-page-with-Langchain
 
# Abstract 

Large language models (LLMs) play a pivotal role in various Natural Language Processing (NLP) tasks. This study focuses on the application of LLMs in the realm of sustainable finance, specifically framing the task as closed-domain question answering. Leveraging the Retrieval-Augmented Generation (RAG) model with top-k retrieval, this paper introduces an innovative approach that combines RAG with insights from Table of Contents (ToC), denoted as RAG + ToC. This method effectively addresses the problem of long inputs, under the assumption of structured data as inputs. In handling extensive outputs, we employ a conventional method of generating an outline, but tailored using RAG + ToC implemented with chain of thought prompting."

# Approach

## Part 1 - Data collection
Collect data relevant for sustainable finance
- `nasdaq_screener.csv` file
<!-- - `web_scrapping.py` to extract the sustainability report links and generate `company_info.csv`, now done manually -->
- `data_helper.py` to create empty output folders, uncomment line 168 to download all sustainability reports in dataframe

## Part 2 - Content generation
Desired structure of output in Wikipedia page
- (Part a) General company information
    - Info from CSV file: Symbol, Name,Last Sale,Net Change,% Change,Market Cap,Country,IPO Year,Volume,Sector,Industry,Company_website,Sustainability_report_link, ESG ratings... 
- (Part b) ESG Overview
    - commitment 
    - achievements
    - ESG reporting frameworks
- (Part c) ESG Approach (detailed information is required, sustainability policies, material topics etc)

Pipeline for content generation: --- see `wiki_gen_base.py` + `wiki_gen_ToC.py`
- Part A
- Part Ci - outline of ESG approaches 
    - Chain of thought prompting: (Zero shot reasoners and allows model to think longer)
        - Assume that ToC is in first three pages, extract table of contents
        - delete rows that are unlikely to be material topics
        - rank the material topics to eliminate less important ones
        - generate as list
- Part Cii - for each point in outline (Ci), generate 3-5 paragraphs
- Part B - generate 1-2 paragraphs 
    - write 1-2 paragraphs about sustainable commitments, achievements
    - transition line that introduces the themes (output from Ci)

^ At different points of the model, different temperature llms are utilised
^ Understanding of the concept "materiality/material topics/material information" may be required for holistic understanding.


## Part 3 - Evaluation










# prev stuff

## Task Background
Encyclopedia documents are integrated information specifically focused on certain topics, and their construction significantly impacts the user experience of search engines. On one hand, by reading encyclopedia web pages, users gain a comprehensive understanding of topics they find interesting. On the other hand, encyclopedia web pages themselves serve as crucial sources of knowledge, forming the basis for the presentation of search results such as knowledge cards, knowledge graphs, and automatic question answering. However, due to the characteristics of encyclopedia documents, including a large number of references, high information compression rates, and high requirements for document quality, the automatic construction of encyclopedia documents faces a series of challenges in text generation.

## Task Content
Select some keywords, gather relevant information, and use a model to generate encyclopedia documents for these keywords. Keywords can include individuals, events, works, and so on, and corresponding encyclopedia documents should be generated.

## Project overview
Embarking on a scholarly initiative, I am developing a project focused on generating Wikipedia pages dedicated to sustainable finance for companies, admist the surging prominence of Environmental, Social, and Governance (ESG). The objective is to offer users comprehensive insights for their ESG evaluation of companies.

While entities like MSCI and Sustainalytics offer quantitative ESG ratings, the inclusion of qualitative data is essential for a more effective and holistic evaluation. This need for qualitative insights is underscored by S&P Global's recent decision to cease publishing alphanumeric ESG credit indicators for publicly rated entities. The initiative also aims to confront greenwashing and contribute to cultivating a more informed and discerning consumer base when assessing corporate sustainability efforts.