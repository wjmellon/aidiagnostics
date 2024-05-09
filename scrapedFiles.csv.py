import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import time
import html
import os
import csv

def scrape_text_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text and decode HTML entities
        text = html.unescape(soup.get_text())
        return text
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return ""

# Load the CSV file
df = pd.read_csv('document_management/citations.csv', usecols=["Citation"])

# Verify if df is empty
if df.empty:
    print("Warning: No data found in CSV file.")
else:
    print(f"Loaded {len(df)} records from CSV.")

# Directory to save files
save_dir = './data'
os.makedirs(save_dir, exist_ok=True)

# File to save scraped texts with metadata
output_file_path = os.path.join(save_dir, 'scraped_texts.csv')
with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['URL', 'Text'])  # Writing header

    # Regular expression pattern for URLs
    url_pattern = r'https?://\S+'

    # Extract URLs and scrape text
    for index, row in df.iterrows():
        match = re.search(url_pattern, row['Citation'])
        if match:
            url = match.group()
            print(f"Scraping {url}")
            text = scrape_text_from_url(url)
            if text:
                writer.writerow([url, text])  # Write URL and scraped text to CSV
            else:
                print(f"No text scraped for {url}")
            time.sleep(1)  # Sleep to avoid hitting server-side rate limits
        else:
            print("No URL found in the citation:", row['Citation'])

print(f"Scraping completed and texts saved to {output_file_path}.")
