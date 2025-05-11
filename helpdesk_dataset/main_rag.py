from query import ask_rag

examples = [
    {"question": "Как восстановить пароль?", "category": "Электронный документооборот"},
    {
        "question": "Где взять инструкцию по заявке на пропуск?",
        "category": "Пропускной режим",
    },
    {
        "question": "Как продлить доступ к почте после увольнения?",
        "category": "Размещение публикаций на информационных ресурсах ТУСУР",
    },
]

print("\n=== Проверка модели с фильтрацией по категории ===\n")
for i, ex in enumerate(examples, 1):
    answer = ask_rag(ex["question"], ex["category"])
    print(
        f"{i}. Вопрос: {ex['question']}\n   Категория: {ex['category']}\n   Ответ: {answer}\n"
    )
