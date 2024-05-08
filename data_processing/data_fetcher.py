
import requests
# possibly access all the data in one URL itself / might make it easier to manage documents / easier to host evt all at once
def download_document(url, path):
    res = requests.get(url)
    if res.status_code == 200:
        with open("data/aggregated.txt", "w") as f:
            f.write(res.text)
    else:
        raise Exception(f"Failed to download document from {url}")
