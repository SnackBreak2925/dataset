# generate_kb_and_rag_docs.py
# 🗂 Создание базы знаний и RAG-документов по категориям с метками статуса

import os
import json
from collections import defaultdict


def generate_kb_and_rag_docs(
    dataset_path="dialogue_dataset.json",
    kb_dir="kb_by_category",
    rag_dir="rag_inputs_by_request",
):
    """Создаёт папку с KB по категориям и RAG-документы по одному на запрос."""

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Файл не найден: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(kb_dir, exist_ok=True)
    os.makedirs(rag_dir, exist_ok=True)

    kb_by_category = defaultdict(set)

    for i, entry in enumerate(data):
        category = entry.get("category_title", "неизвестно").strip()
        context = entry["label"].strip()
        question = entry["text"].strip()

        kb_by_category[category].add(context)

        rag_doc = {
            "id": f"req_{i:05}",
            "category_title": category,
            "question_text": question,
            "expected_answer": context,
            "rag_prompt": f"Категория: {category}\nКонтекст: {context}\nВопрос: {question}",
            "status": "pending",  # можно менять на 'processed', 'validated', 'skipped' и т.д.
            "notes": "",
        }

        with open(
            os.path.join(rag_dir, rag_doc["id"] + ".json"), "w", encoding="utf-8"
        ) as fout:
            json.dump(rag_doc, fout, ensure_ascii=False, indent=2)

    for category, entries in kb_by_category.items():
        cat_file = os.path.join(kb_dir, category + ".json")
        with open(cat_file, "w", encoding="utf-8") as fcat:
            json.dump(sorted(entries), fcat, ensure_ascii=False, indent=2)

    print(f"✅ Создано KB по категориям: {len(kb_by_category)}")
    print(f"✅ Документов RAG: {len(data)} в папке '{rag_dir}'")


if __name__ == "__main__":
    generate_kb_and_rag_docs()
