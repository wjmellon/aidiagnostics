import pandas as pd
import re
import requests
from bs4 import BeautifulSoup

# Define a function to scrape text from a URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code.
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return ""

# Load the CSV file
df = pd.read_csv('./citations.csv')

# Regular expression pattern for URLs
url_pattern = r'https?://\S+'

# Extract URLs from the "Citation" column
df['URLs'] = df['Citation'].apply(lambda x: re.findall(url_pattern, x))
urls = df['URLs'].explode().unique().tolist()

# Initialize a variable to hold all the scraped text
all_text = ""

# Scrape text from each URL and append it to the all_text variable
for url in urls:
    text = scrape_text_from_url(url)
    all_text += text + "\n\n"  # Add two newlines as separators between texts

# Save all the scraped text into one big text file
with open('collected_texts.txt', 'w', encoding='utf-8') as file:
    file.write(all_text)

print("Scraping completed and text saved to collected_texts.txt.")

