import requests
from bs4 import BeautifulSoup
import pandas as pd

"""
Generate `company_info.csv`:
    - all columns from nasdaq screener
    - sustainability report links -  - need to do web scraping
    - ISIN code (TBC) - need to do web scraping
    - ESG rating (TBC) - need to do web scraping
Remember to deal with edge cases where there are no sustainabiltity report links, ISIN codes, ESG rating etc

For now the `company_info.csv` is 手动爬取的
Basically this is to ensure the expandability of the project
"""

# want apple, Johnson & Johnson, Procter & Gamble, Bank of America, Coca-Cola

def get_website_urls(company_name, keywords):
    # Step 1: Search for the company's main website URL
    search_url = f"https://www.google.com/search?q={company_name} {keywords}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(search_url, headers=headers)

    # Step 2: Extract the main website URL from the search results
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        with open("output.txt", "w", encoding="utf-8") as file:
            file.write(soup.prettify())
        links_cap = 10
        href_values = []    
        for link in soup.find_all('a', href=True):
            links_cap-=1
            href_values.append(link.get('href'))
            print(link.get('href'))
            if links_cap == 0:
                break
            # if 'url?q=' in link['href'] and 'webcache' not in link['href']:
            #     # Extract the URL
            #     main_website_url = link['href'].split('url?q=')[1].split('&')[0]
            #     return main_website_url

        df = pd.DataFrame({'index': range(1, len(href_values) + 1), 'link.get(\'href\')': href_values})
        print(df)
    
    # return None

get_website_urls("Apple","sustainability report")

# def get_sustainability_report_links(company_website_url):
#     # Step 3: Access the company's main website and find links to sustainability reports
#     if company_website_url:
#         response = requests.get(company_website_url, headers=headers)
        
#         if response.status_code == 200:
#             website_soup = BeautifulSoup(response.text, 'html.parser')
            
#             # Step 4: Find links to Sustainability Reports or annual reports
#             sustainability_report_keywords = ['sustainability report', 'annual report', 'CSR report']
#             sustainability_report_links = []
            
#             for keyword in sustainability_report_keywords:
#                 report_links = website_soup.find_all('a', string=lambda s: keyword.lower() in s.lower(), href=True)
#                 sustainability_report_links.extend([link['href'] for link in report_links])
            
#             return sustainability_report_links
    
#     return None

# # Example usage:
# company_name = "Apple"
# company_website_url = get_company_website_url(company_name)

# if company_website_url:
#     print(f"Main Website URL for {company_name}: {company_website_url}")

#     sustainability_report_links = get_sustainability_report_links(company_website_url)
#     if sustainability_report_links:
#         print(f"Sustainability Report Links: {sustainability_report_links}")
#     else:
#         print("No Sustainability Report Links found.")
# else:
#     print("Main website URL not found.")


#supposed to be able to export out this function, but if cannot then just drop this component...


