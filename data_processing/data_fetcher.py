import requests
from bs4 import BeautifulSoup
import os
from config.config import URLS, PATH_TO_SAVE

def download_documents(urls, path):
    headers = {'User-Agent': 'Mozilla/5.0'}

    # Ensure the directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, "w") as aggregated_file:
        for url in urls:
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    # Different handling based on content-type
                    if 'text/plain' in response.headers.get('Content-Type', ''):
                        text_content = response.text
                    else:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        text_content = '\n'.join([p.text for p in soup.find_all('p')])
                    # Check if text_content is not empty
                    if text_content:
                        aggregated_file.write(text_content + "\n\n")
                        print(f"Successfully processed and saved content from {url}")
                    else:
                        print(f"Content extracted from {url} is empty.")
                else:
                    print(f"Failed to download content from {url}, status code: {response.status_code}")
            except Exception as e:
                print(f"An error occurred while processing {url}: {e}")

# Example usage with your URLs
download_documents(URLS, PATH_TO_SAVE)
