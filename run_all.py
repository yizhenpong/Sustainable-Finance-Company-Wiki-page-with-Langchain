'''
go to `data_helper.py` download pdf files and generate required output folders
'''

from wiki_gen_base import run_wiki_gen_base
# from wiki_gen_ToC import gen_ToC
from data_helper import get_all_companies

for symbol in get_all_companies()['Symbol']:

    print(f"""
          #====================================================== 
          # RAG function has started for {symbol}
          #====================================================== """)

    run_wiki_gen_base(symbol)

    print(f"""
          #====================================================== 
          # RAG function has completed for {symbol}
          #====================================================== """)
    
    
    
    
#     print(f"""
#           #====================================================== 
#           # ToC function has started for {symbol}
#           #====================================================== """)

#     pageRange = run_wiki_gen_ToC(symbol) # filtered pages range - from ToCfilter

#     print(f"""
#           #====================================================== 
#           # ToC function has completed for {symbol}
#           #====================================================== """)
    
    
    
#     print(f"""
#           #====================================================== 
#           # RAG + ToC function has started for {symbol}
#           #====================================================== """)

#     pageRange = run_wiki_gen_base(symbol,pageRange=pageRange, ToCStatus=True) # filtered pages range - from ToCfilter

#     print(f"""
#           #====================================================== 
#           # RAG + ToC function has completed for {symbol}
#           #
#           # Whole run completed!
#           #
#           #====================================================== """)