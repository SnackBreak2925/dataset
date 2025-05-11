import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import os

MODEL_PATH = os.path.join(os.getenv("PWD"), "rut5", "rut5-small-finetuned")

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def rag_answer(question, category_title, kb, top_k=1):
    # 1. Фильтруем статьи по категории
    filtered_kb = [
        entry for entry in kb if entry["metadata"]["category"] == category_title
    ]
    if not filtered_kb:
        return "⚠️ Нет инструкций по этой категории."

    # 2. Векторизуем
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    kb_texts = [entry["text"] for entry in filtered_kb]
    kb_vectors = vectorizer.fit_transform(kb_texts)
    question_vector = vectorizer.transform([question])

    # 3. Сходство
    similarities = cosine_similarity(question_vector, kb_vectors).flatten()
    top_idxs = similarities.argsort()[-top_k:][::-1]
    context = "\n".join([kb_texts[i] for i in top_idxs])

    # 4. Генерация
    prompt = (
        f"Категория: {category_title}\nВопрос: {question}\nКонтекст:\n{context}\nОтвет:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
