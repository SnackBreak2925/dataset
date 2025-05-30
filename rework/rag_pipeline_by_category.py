# rag_pipeline_by_category.py
# 🧠 RAG‑пайплайн с улучшенной категоризацией, удалением дубликатов и генерацией разнообразных ответов

import json
import os
import logging
import csv
import datetime
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# ============ Настройка логирования и путей ==========
logging.basicConfig(
    filename="rag_debug.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("\n=== RAG‑скрипт запущен ===")

MODEL_PATH = "./rut5base-finetuned"
KB_DIR = "kb_by_category"
CSV_LOG = "rag_answers_log.csv"
TOP_K = 5
SOFT_THRESHOLD = 0.1
ANSWER_MIN_LEN = 15
PATTERNS_TO_SKIP = [r"^\[SIGNATURE\]$", r"\[URL\]"]

# ============ Загрузка модели и токенизатора ==========
logger.info("Загрузка модели...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).eval()
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

retriever = SentenceTransformer("all-MiniLM-L6-v2")

# ============ Загрузка KB с удалением дубликатов ==========
logger.info("Чтение KB из %s", KB_DIR)
category_kbs, category_embeddings = {}, {}
category_names = []
for fn in os.listdir(KB_DIR):
    if fn.endswith(".json"):
        cat = fn[:-5]
        with open(os.path.join(KB_DIR, fn), encoding="utf-8") as f:
            entries = json.load(f)
        unique_entries = sorted(set(e.strip() for e in entries if e.strip()))
        category_kbs[cat] = unique_entries
        category_embeddings[cat] = retriever.encode(
            unique_entries, convert_to_tensor=True
        )
        category_names.append(cat)

category_name_embeddings = retriever.encode(category_names, convert_to_tensor=True)
print(f"📚 Загружено категорий: {len(category_kbs)}\n")


# ============ Хелперы ==========
def safe_encode(text):
    try:
        return retriever.encode(text, convert_to_tensor=True, show_progress_bar=False)
    except Exception as e:
        logger.exception("Ошибка encode: %s", e)
        return None


def auto_detect_category(question, top_n=3):
    q_emb = safe_encode(question)
    if q_emb is None:
        return None
    scores = util.cos_sim(q_emb, category_name_embeddings)[0]
    top_indices = torch.topk(scores, k=top_n).indices.tolist()
    best_score = scores[top_indices[0]].item()
    if best_score < 0.2:
        return None
    return category_names[top_indices[0]]


def is_uninformative(answer):
    if len(answer) < ANSWER_MIN_LEN:
        return True
    for pattern in PATTERNS_TO_SKIP:
        if re.search(pattern, answer, flags=re.IGNORECASE):
            return True
    return False


def generate_with_contexts(category, question, top_k=5):
    kb = category_kbs[category]
    emb = category_embeddings[category]
    q_emb = safe_encode(question)
    if q_emb is None:
        return ["[Ошибка кодирования]"], []

    try:
        hits = util.semantic_search(q_emb, emb, top_k=top_k)[0]
    except Exception as e:
        logger.exception("semantic_search error: %s", e)
        return ["[Ошибка поиска контекста]"], []

    if not hits:
        return ["[Нет релевантных контекстов]"], []

    max_score = hits[0]["score"]
    adaptive_threshold = max(SOFT_THRESHOLD, max_score * 0.15)

    responses, results = [], []
    seen_answers = set()

    for hit in hits:
        if hit["score"] < adaptive_threshold:
            continue
        idx = hit["corpus_id"]
        ctx = kb[idx]
        prompt = f"Категория: {category}\nКонтекст: {ctx}\nВопрос: {question}"
        logger.info("Prompt used:\n%s\n", prompt)
        try:
            ids = tokenizer.encode(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_length=64,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                )
            ans = tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logger.exception("Генерация ответа: %s", e)
            ans = "[Ошибка генерации]"

        if ans in seen_answers or is_uninformative(ans):
            continue

        responses.append(ans)
        results.append((ctx, ans, hit["score"]))
        seen_answers.add(ans)

    return responses, results


# ============ CSV логгер ==========
def log_to_csv(category, question, results):
    with open(CSV_LOG, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for ctx, ans, score in results:
            writer.writerow(
                [
                    datetime.datetime.now().isoformat(),
                    category,
                    question,
                    ctx,
                    ans,
                    f"{score:.4f}",
                ]
            )


if __name__ == "__main__":
    # ============ Интерактив ==========
    print("Введите запрос (или 'exit'):")
    while True:
        question = input("Запрос: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("До свидания!")
            break
        if not question:
            print("❗ Вопрос не должен быть пустым.")
            continue

        category = auto_detect_category(question)
        if not category:
            print("❌ Не удалось определить категорию запроса.")
            continue

        print(f"📂 Категория определена: {category}")
        answers, results = generate_with_contexts(category, question, top_k=TOP_K)
        if results:
            log_to_csv(category, question, results)
        if not results:
            print("ℹ️ Ответы были сгенерированы, но все отфильтрованы.")

        if not answers:
            print("⚠️ Не найдено релевантных ответов. Попробуйте переформулировать запрос.")
            continue

        print("\n📌 Лучшие ответы:")
        for i, (ctx, ans, score) in enumerate(results, 1):
            print(f"{i}) {ans}\n   🔹 Релевантность: {score:.4f}\n")
