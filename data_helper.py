import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re
import json

'''
Getting data from csv file 'company_info.csv'
Downloading the sustainability reports

download pdf to file path: "data/sustainability_reports/{symbol}.pdf"
output folders:
    output_base
        - individual companies based on symbol
            - company_info.txt
            - ESG_approach_outline
            - ESG_approach
            - ESG_overview
    output_ToC
        - same as above
'''

############################################################################################################################## 
'''import overall data info as data frame'''
df = pd.read_csv('data/company_info.csv')
headers = list(df)

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

def create_folder(main_f_path=["output_base", "output_ToC", "data/sustainability_reports"],symbol=-1):
    if symbol == -1:
        folder_name = f"{main_f_path}"
    else:
        folder_name = f"{main_f_path}/{symbol}"
    # path = os.path.join("output", symbol) 
    try:
        # Create a folder with the specified name
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully in output folder.")
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists in output folder.")

def batch_download():
    for index, row in df.iloc[:, :].iterrows():
        symbol = row['Symbol']
        url = row['Sustainability_report_link']
        download_pdf(url, symbol)

def batch_create_output_folder(main_f_path):
    for index, row in df.iloc[:, :].iterrows():  # Skip the header row
        symbol = row['Symbol']
        create_folder(main_f_path,symbol)
        

def write_output(symbol,text_output,
                 portion=["ESG_approach","ESG_approach_outline", "ESG_overview", 
                          "Company_info","ToC_ESG_approach_outline", "ToC_ESG_approach_outline_process"], 
                 header = False, list_type = False, json_type = False, ToC = False):
    if ToC:
        if not json_type:
            file_path = f"output_ToC/{symbol}/{portion}.txt"
        else:
            file_path = f"output_ToC/{symbol}/{portion}.json"
    if not ToC:
        if not json_type:
            file_path = f"output_base/{symbol}/{portion}.txt"
        else:
            file_path = f"output_base/{symbol}/{portion}.json"
    if list_type:
        with open(file_path, 'a') as file:
            [file.write(f"{item}\n ") for item in text_output]
            print(f"\n {symbol} list in text_output saved into {file_path} ")
    elif json_type:
        with open(file_path,"w") as file:
            json.dump(text_output,file)
    else:
        with open(file_path, 'a') as file:
            file.write(f"{text_output} \n ")
        if header:
            print(f"\n {symbol} header, {text_output}, saved into {file_path}\n")
        else:
            print(f"\n {symbol} output saved into {file_path} \n")

def get_filtered_pages(input_string):
    # Define a regular expression pattern to match "answer:" followed by a list
    pattern = r"answer:\s*\[([^\]]*)\]"

    # Use re.search to find the pattern in the input string
    match = re.search(pattern, input_string)

    # If a match is found, return the captured group (the content of the square brackets)
    if match:
        text_with_newlines = match.group(1)
        cleaned_text = text_with_newlines.replace("\n", "")
        return cleaned_text
    else:
        # Return None if no match is found
        return f"""Pls find manually... \n \n
        #==========================================
        {input_string}
        #=========================================="""
    
class RangeElement:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        if ':' in self.value:
            start, end = map(int, self.value.split(':'))
            return f"{start}:{end}"
        else:
            return str(self.value)

def convert_to_list(input_string):
    # Split the string into individual elements
    elements = input_string.split(',')

    # Convert each element to the appropriate format
    output_list = []
    for element in elements:
        element = element.strip()
        output_list.append(RangeElement(element))

    return output_list

def get_ToC_f_path(symbol):
    return f"output_ToC/{symbol}/TableOfContents.txt"





##############################################################################################################################   
'''test case'''
# symbol = "AAPL"
# print(extract_company_info(symbol))
# url = extract_company_info(symbol)['Sustainability_report_link'].iloc[0]
# print(url)
# print(download_pdf(url,"AAPL"))
# print(get_all_companies())
# print(get_csv_headers())

############################################################################################################################## 
'''main'''
if __name__ == '__main__':
    # batch_download()
    create_folder(main_f_path="output_base")
    create_folder(main_f_path="output_ToC") # creates base folders
    batch_create_output_folder("output_base")
    batch_create_output_folder("output_ToC") #creates individual company folders
    pass