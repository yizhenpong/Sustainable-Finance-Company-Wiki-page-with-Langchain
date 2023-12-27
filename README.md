# Sustainable-Finance-Company-Wiki-page-with-Langchain
 
# Introduction 

## Task Background
Encyclopedia documents are integrated information specifically focused on certain topics, and their construction significantly impacts the user experience of search engines. On one hand, by reading encyclopedia web pages, users gain a comprehensive understanding of topics they find interesting. On the other hand, encyclopedia web pages themselves serve as crucial sources of knowledge, forming the basis for the presentation of search results such as knowledge cards, knowledge graphs, and automatic question answering. However, due to the characteristics of encyclopedia documents, including a large number of references, high information compression rates, and high requirements for document quality, the automatic construction of encyclopedia documents faces a series of challenges in text generation.

## Task Content
Select some keywords, gather relevant information, and use a model to generate encyclopedia documents for these keywords. Keywords can include individuals, events, works, and so on, and corresponding encyclopedia documents should be generated.

## Project overview
Embarking on a scholarly initiative, I am developing a project focused on generating Wikipedia pages dedicated to sustainable finance for companies, admist the surging prominence of Environmental, Social, and Governance (ESG). The objective is to offer users comprehensive insights for their ESG evaluation of companies.

While entities like MSCI and Sustainalytics offer quantitative ESG ratings, the inclusion of qualitative data is essential for a more effective and holistic evaluation. This need for qualitative insights is underscored by S&P Global's recent decision to cease publishing alphanumeric ESG credit indicators for publicly rated entities. The initiative also aims to confront greenwashing and contribute to cultivating a more informed and discerning consumer base when assessing corporate sustainability efforts.

# Approach

## Part 1 - Data collection
Collect data relevant for sustainable finance
- nasdaq screener csv file
- `web_scrapping.py` to extract the sustainability report links to generate `company_info.csv`
- `data_helper.py` to download all sustainability reports

## Part 2 - Content generation
Ideally the structure that I want to have:
- General company information
    - Info from CSV file: Symbol, Name,Last Sale,Net Change,% Change,Market Cap,Country,IPO Year,Volume,Sector,Industry,Company_website,Sustainability_report_link
- ESG Overview
    - commitment (work require generation)
    - achievements
    - ESG ratings
    - ESG reporting frameworks
- ESG Approach (detailed information is required, sustainability policies, material topics etc)

## Part 3 - Evaluation




## Sources
https://github.com/Vasanthengineer4949/NLP-Projects-NHV/tree/main/Langchain%20Projects/7_AI_Financial_Advisor/src 
https://github.com/samwit/langchain-tutorials/blob/main/RAG/YT_Chat_your_PDFs_Langchain_Template_for_creating.ipynb 
https://github.com/AIAnytime/Haystack-and-Mistral-7B-RAG-Implementation/blob/main/model_add.py 

