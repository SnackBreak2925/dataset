import json
import os
import torch
import datetime
import re
import logging
import csv
import uuid
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

MODEL_PATH = "./rut5base-finetuned"
KB_DIR = "kb_by_category"
CSV_LOG = "dialogue_log.csv"
DIALOGUE_DIR = "dialogue_sessions"
TOP_K = 5
SOFT_THRESHOLD = 0.1
ANSWER_MIN_LEN = 15
PATTERNS_TO_SKIP = [r"^\[SIGNATURE\]$", r"\[URL\]"]

os.makedirs(DIALOGUE_DIR, exist_ok=True)

logging.basicConfig(
    filename="rag_debug.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# === Безопасная токенизация ===
def safe_encode(text):
    if not isinstance(text, str):
        logging.warning(f"Попытка закодировать не-строку: {repr(text)}")
        return None
    text = text.strip()
    if not text:
        logging.warning("Попытка закодировать пустую строку")
        return None
    try:
        return retriever.encode(text, convert_to_tensor=True)
    except Exception as e:
        logging.exception(f"Ошибка при кодировании текста: {repr(text)} — {e}")
        return None


# === Инициализация моделей ===
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = (
    T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    .eval()
    .to("cuda" if torch.cuda.is_available() else "cpu")
)
retriever = SentenceTransformer("all-MiniLM-L6-v2")

category_kbs, category_embeddings = {}, {}
category_names = []
for fn in os.listdir(KB_DIR):
    if fn.endswith(".json"):
        cat = fn[:-5]
        with open(os.path.join(KB_DIR, fn), encoding="utf-8") as f:
            entries = json.load(f)
        entries = sorted(set(e.strip() for e in entries if e.strip()))
        category_kbs[cat] = entries
        category_embeddings[cat] = retriever.encode(entries, convert_to_tensor=True)
        category_names.append(cat)

category_name_embeddings = retriever.encode(category_names, convert_to_tensor=True)


def is_uninformative(answer):
    if len(answer) < ANSWER_MIN_LEN:
        return True
    for p in PATTERNS_TO_SKIP:
        if re.search(p, answer, flags=re.IGNORECASE):
            return True
    return False


def auto_detect_category(question):
    q_emb = safe_encode(question)
    if q_emb is None:
        return None
    sims = util.cos_sim(q_emb, category_name_embeddings)[0]
    top = torch.topk(sims, 1).indices[0].item()
    return category_names[top] if sims[top] >= 0.2 else None


def generate_response(category, dialogue, top_k=TOP_K):
    kb = category_kbs[category]
    kb_emb = category_embeddings[category]
    last_question = dialogue[-1]["user"]
    q_emb = safe_encode(last_question)
    if q_emb is None:
        return "[Ошибка обработки запроса, попробуйте переформулировать]", 0.0
    hits = util.semantic_search(q_emb, kb_emb, top_k=top_k)[0]
    if not hits:
        return "[нет контекста]", 0.0
    max_score = hits[0]["score"]
    threshold = max(SOFT_THRESHOLD, max_score * 0.15)
    ctx = next(
        (kb[h["corpus_id"]] for h in hits if h["score"] >= threshold),
        kb[hits[0]["corpus_id"]],
    )
    dialogue_text = "\n".join(
        [f"Пользователь: {t['user']}\nМодель: {t['bot']}" for t in dialogue[:-1]]
    )
    prompt = f"Категория: {category}\nКонтекст: {ctx}\nДиалог:\n{dialogue_text}\nПользователь: {last_question}\nМодель:"
    logging.info("Prompt:\n%s\n", prompt)
    try:
        ids = tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_length=64, do_sample=True, top_k=50, top_p=0.95
            )
        answer = tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        logging.exception("Ошибка генерации ответа: %s", e)
        return "[Ошибка генерации ответа]", 0.0
    return answer, hits[0]["score"]


def log_turn(category, turn_id, user, bot, relevance):
    with open(CSV_LOG, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.datetime.now().isoformat(),
                category,
                turn_id,
                user,
                bot,
                f"{relevance:.4f}",
            ]
        )


def save_dialogue_json(session_id, category, dialogue):
    path = os.path.join(DIALOGUE_DIR, f"{session_id}.json")
    data = {
        "session_id": session_id,
        "category": category,
        "timestamp": datetime.datetime.now().isoformat(),
        "dialogue": dialogue,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_saved_sessions():
    files = sorted(f for f in os.listdir(DIALOGUE_DIR) if f.endswith(".json"))
    if not files:
        print("(Нет сохранённых сессий)")
    else:
        print("Доступные ID сессий:")
        for f in files:
            print(" -", f.replace(".json", ""))


def load_dialogue_json(session_id):
    path = os.path.join(DIALOGUE_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        print(f"❌ Сессия {session_id} не найдена.")
        return None, None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["category"], data["dialogue"]


def print_dialogue(dialogue):
    for i, turn in enumerate(dialogue, 1):
        print(f"\n🗣 [{i}] Пользователь: {turn['user']}")
        print(f"🤖 [{i}] Модель: {turn['bot']}")


def continue_dialogue(session_id):
    category, dialogue = load_dialogue_json(session_id)
    if not category or not dialogue:
        return
    print(f"🔄 Продолжение диалога {session_id} (категория: {category})")
    print_dialogue(dialogue)
    while True:
        follow_up = input("\nПользователь: ").strip()
        if follow_up.lower() in {"exit", "quit"}:
            break
        dialogue.append({"user": follow_up, "bot": ""})
        answer, score = generate_response(category, dialogue)
        dialogue[-1]["bot"] = answer
        turn_id = len(dialogue)
        log_turn(category, turn_id, follow_up, answer, score)
        print(f"🤖 {answer} (релевантность: {score:.3f})")
    save_dialogue_json(session_id, category, dialogue)
    print(f"💾 Диалог обновлён: {session_id}.json")


def run_dialogue():
    session_id = uuid.uuid4().hex[:8]
    dialogue = []
    print("Введите запрос (или 'exit'):")
    question = input("Запрос: ").strip()
    if not question or question.lower() in {"exit", "quit"}:
        return
    category = auto_detect_category(question)
    if not category:
        print("Категория не определена.")
        return
    print(f"📂 Категория: {category}\n")
    dialogue.append({"user": question, "bot": ""})
    while True:
        answer, score = generate_response(category, dialogue)
        dialogue[-1]["bot"] = answer
        turn_id = len(dialogue)
        log_turn(category, turn_id, dialogue[-1]["user"], answer, score)
        print(f"🤖 {answer} (релевантность: {score:.3f})")
        follow_up = input("\nПользователь: ").strip()
        if follow_up.lower() in {"exit", "quit"}:
            break
        dialogue.append({"user": follow_up, "bot": ""})
    save_dialogue_json(session_id, category, dialogue)
    print(f"💾 Диалог сохранён: {session_id}.json")


if __name__ == "__main__":
    print("1. Новый диалог")
    print("2. Продолжить диалог по ID")
    mode = input("Выбор (1/2): ").strip()
    if mode == "1":
        run_dialogue()
    elif mode == "2":
        list_saved_sessions()
        session_id = input("Введите ID сессии: ").strip()
        continue_dialogue(session_id)
    else:
        print("Неверный выбор.")
