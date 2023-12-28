'''
go to `data_helper.py` download pdf files and generate required output folders
'''

from wiki_gen_base import run_wiki_gen_base
from data_helper import get_all_companies

for symbol in get_all_companies()['Symbol']:

    print(f"""
          #====================================================== 
          # run_wiki_gen_base function has started for {symbol}
          #====================================================== """)

    run_wiki_gen_base(symbol)

    print(f"""
          #====================================================== 
          # run_wiki_gen_base function has completed for {symbol}
          #====================================================== """)