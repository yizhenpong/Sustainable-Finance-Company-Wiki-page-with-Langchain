import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re

'''
Getting data from csv file 'company_info.csv'
Downloading the sustainability reports

Functions: 
    extract_company_info: 
        input: (str) symbol (unique for each company)
        output: (dataframe) of all relevant company fields 
    download_pdf: 
        input: 
            (str) url of the pdf
            (str) symbol (unique for each company)
        output: sustainability reports saved in 'data/sustainabiltiy_reports/' and named as {symbol}.pdf
    batch_download: 
        output: all sustainability reports for companies listed in 'company_info.csv
'''

############################################################################################################################## 
'''import overall data info as data frame'''
df = pd.read_csv('data/company_info.csv')
headers = list(df)
# print(headers)
# df = pd.read_csv ('data/nasdaq_screener_1702878444666.csv')
# print(len(df)) # num of rows
# print(df['Symbol'].nunique())

def extract_company_info(symbol):
    return df.loc[df["Symbol"]==symbol, :]

def get_csv_headers():
    return headers

def get_all_companies():
    return df.iloc[:, :2]

def download_pdf(url, symbol):
    response = requests.get(url,stream=True)
    if response.status_code == 200:
        filepath = "data/sustainability_reports/"+symbol+".pdf"
        with open(filepath, 'wb') as pdf_object:
            pdf_object.write(response.content)
            print(f'{symbol}.pdf was successfully saved!')
    else:
        print(f'Uh oh! Could not download {symbol}.pdf,')
        print(f'HTTP response status code: {response.status_code}')

def get_SR_file_path(symbol):
    return f"data/sustainability_reports/{symbol}.pdf"

def create_folder(symbol):
    folder_name = f"output/{symbol}"
    # path = os.path.join("output", symbol) 
    try:
        # Create a folder with the specified name
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully in output folder.")
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists in output folder.")

def batch_download():
    for index, row in df.iloc[1:, :].iterrows():  # Skip the header row
        symbol = row['Symbol']
        url = row['Sustainability_report_link']
        download_pdf(url, symbol)
        create_folder(symbol)

# def batch_create_output_folder():
#     for index, row in df.iloc[:, :].iterrows():  # Skip the header row
#         symbol = row['Symbol']
#         create_folder(symbol)
        

def write_output(symbol,text_output,
                 portion=["ESG_approach","ESG_approach_outline", "ESG_overview", "Company_info"], 
                 header = False, list_type = False, title = False):
    file_path = f"output/{symbol}/{portion}.txt"
    if list_type:
        with open(file_path, 'a') as file:
            [file.write(f"{item} /n ") for item in text_output]
            print(f"/n {symbol} list in text_output saved into {file_path} /n")
    else:
        with open(file_path, 'a') as file:
            file.write(f"{text_output} /n ")
        if header:
            print(f"/n {symbol} header, {text_output}, saved into {file_path} /n")
        if title:
            print(f"/n {symbol} company name saved into {file_path} /n")
        else:
            print(f"/n {symbol} output saved into {file_path} /n")

def write_test_output(symbol, text_output, what_output):
    file_path = f"output/{symbol}/{what_output}.txt"
    with open(file_path, 'a') as file:
            file.write(f"{text_output} /n ")


##############################################################################################################################   
'''test case'''
# symbol = "AAPL"
# print(extract_company_info(symbol))
# url = extract_company_info(symbol, df)['Sustainability_report_link'].iloc[0]
# print(url)
# print(download_pdf(url,"AAPL"))
# print(get_all_companies())
# print(get_csv_headers())

############################################################################################################################## 
'''main'''
if __name__ == '__main__':
    # batch_download()
    # batch_create_output_folder()
    pass