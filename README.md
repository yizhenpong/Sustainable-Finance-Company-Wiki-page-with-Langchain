# Sustainable-Finance-Company-Wiki-page-with-Langchain
 
# Abstract 

Large language models (LLMs) play a pivotal role in various Natural Language Processing (NLP) tasks. This study focuses on the application of LLMs in the realm of sustainable finance, specifically framing the task as closed-domain question answering. Leveraging on Retrieval-Augmented Generation (RAG) with top-k retrieval, this paper introduces an innovative approach that combines RAG with insights from Table of Contents (ToC), denoted as RAG + ToC. This method effectively addresses the problem of long inputs, under the assumption of structured data as inputs where ToC acts as an optimised filter for handling our query. In handling extensive outputs, we employ a conventional method of generating an outline, but tailored using RAG + ToC with chain of thought prompting.

# Approach

LLM model = Mistral 7b, LLM framework = langchain, OllamaEmbeddings, Chroma vector store, LLM hosted on Ollama

## Part 1 - Data collection
Collect data relevant for sustainable finance
- `nasdaq_screener.csv` file
- `company_info.csv` file: contains all selected nasdaq stock codes, fill in the column on sustainability report (SR) link
- `data_helper.py` file downloads all SR in `company_info.csv`, creates output folders and corresponding stock code empty files

## Part 2 - Content generation
Structure of the output (Sustainable Finance Wikipedia pages):
- (Part a) General company information
    - Info from CSV file: Symbol, Name,Last Sale,Net Change,% Change,Market Cap,Country,IPO Year,Volume,Sector,Industry,Company_website,Sustainability_report_link, ESG ratings... 
- (Part b) ESG Overview
    - commitment 
    - achievements
- (Part c) ESG Approach (detailed information is required, sustainability policies, material topics etc)

Pipeline for content generation: --- see `wiki_gen_base.py` + `wiki_gen_ToC.py`
- Part A
- Part Ci - outline of ESG approaches 
    - [ToC pipeline as described in paper]
    - Chain of thought prompting: (Zero shot reasoners and allows model to think longer)
        - Assumes that ToC is in first three pages, extract table of contents
        - delete rows that are unlikely to be material topics
        - rank the material topics to eliminate less important ones
        - generate as list
- Part Cii - for each point in outline (Ci), generate 3-5 paragraphs
- Part B - generate 1-2 paragraphs 
    - write 1-2 paragraphs about sustainable commitments, achievements
    - transition line that introduces the themes (output from Ci)

^ At different points of the model, different temperature llms are utilised, please refer to the system prompts and llm
^ Understanding of the concept "materiality/material topics/material information" may be required for holistic understanding.

## Part 3 - Evaluation
- Part A 
    - See ground truth in `eval/section_A_ground_truth.json` - human generated
    - See factual evaluation in `eval/section_A_eval.json`:
        ``` for each company:
            for each field in section A
                eval: 0 or 1 (binary) 
                    (1 means it is factually correct)
                type: absolute or relative 
                    (absolute means that the information is usually static or any random two people woudl come up with the same answer. relative means that information is debatable, example, important people - who is to determine the extent of importance)
                reason: NA or any text 
                    (NA for eval = 1/type = absolute) ```
- Part B and C
    - readibility score (ARI)
    - language tool (grammar check)










