import json
from bs4 import BeautifulSoup

# def load_nq_dataset(file_path):
#     texts = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             record = json.loads(line)
#             html = record.get("document_text", "")
#             soup = BeautifulSoup(html, "html.parser")
#             text = soup.get_text(" ", strip=True)
#             texts.append(text)
#     return texts

def load_nq_dataset(file_path,limit=500):
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            if i >= limit:
                break
            record = json.loads(line)
            html = record.get("document_text", "")
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)
            texts.append(text)
            if i % 50 == 0:
                print(f"loaded {i} documents")
    return texts