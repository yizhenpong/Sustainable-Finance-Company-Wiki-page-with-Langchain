import time
import pandas as pd


"""
# Time taken from start to Indexing(RAG) - indexRAG	
# Time taken from start to Indexing (RAG + ToC)	- indexRAGToC
# Time from start to end (RAG) 	- allRAG
# Time from start to end (RAG + ToC) -allRAGToC
# ["indexRAG","indexRAGToC","allRAG","allRAGToC"]"""


def save_time(company_symbol, start, index, end,ToCStatus=False):
    if not ToCStatus:
        indexRAG = index - start
        allRAG = end - start
        executn_time = {
        "indexRAG":indexRAG,
        "allRAG": allRAG,
        "start": start,
        "index": index,
        "end": end,
        }
        file_path = f"output_base/{company_symbol}/executn_time.csv"
    else:
        indexRAGToC = index - start
        allRAGToC = end - start
        executn_time = {
        "indexRAGToC":indexRAGToC,
        "allRAGToC": allRAGToC,
        "start": start,
        "index": index,
        "end": end,
        }
        file_path = f"output_ToC/{company_symbol}/executn_time.csv"
    df = pd.DataFrame(executn_time, index=[company_symbol])
    df.to_csv(file_path, index=False)