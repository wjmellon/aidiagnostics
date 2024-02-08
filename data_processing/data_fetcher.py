
import requests

def download_document(url, path):
    res = requests.get(url)
    if res.status_code == 200:
        with open("data/aggregated.txt", "w") as f:
            f.write(res.text)
    else:
        raise Exception(f"Failed to download document from {url}")
