# 🔍 RAG логика
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-base")
model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base")
def rag_answer(q, cat, kb):
    texts = [x["text"] for x in kb if x["category_title"] == cat]
    if not texts: return "Нет информации"
    vec = TfidfVectorizer().fit(texts)
    idx = cosine_similarity(vec.transform([q]), vec.transform(texts)).argmax()
    return texts[idx]
