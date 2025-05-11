import json
from sklearn.feature_extraction.text import TfidfVectorizer
import os


def load_kb(path=os.getenv("PWD") + "/knowledge_base.json"):
    with open(path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    return kb


def prepare_vectorizer(kb_entries):
    texts = [entry["text"] for entry in kb_entries]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors
