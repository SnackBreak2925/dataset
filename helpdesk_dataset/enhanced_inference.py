from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os

# 1. Загружаем твою дообученную модель
model_path = os.path.join(os.getenv("PWD"), "rut5", "rut5-small-finetuned")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()

# 2. Фиксированные тестовые вопросы
test_questions = [
    "Как узнать статус заявки?",
    "Что делать, если потерял пароль?",
    "Как оформить заявку на пропуск?",
    "Куда обращаться за поддержкой по софту?",
    "Какие документы нужны для оформления справки?",
]


# 3. Функция для генерации ответов с "разогревающим" промптом
def generate_answer(question, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Новый промпт — подсказываем модели быть подробной
    prompt = (
        f"Тикет: {question}\n"
        f"Пожалуйста, дайте развёрнутый, информативный ответ пользователю."
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,  # Чуть больше макс длина ответа
            do_sample=True,  # Sampling вместо beam search
            top_p=0.9,  # Nucleus sampling
            temperature=0.8,  # Небольшая креативность
            num_return_sequences=1,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# 4. Генерация и вывод ответов
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n=== Проверка модели с улучшенным промптом ===\n")
for idx, question in enumerate(test_questions, 1):
    answer = generate_answer(question, device)
    print(f"{idx}. Вопрос: {question}")
    print(f"   Ответ: {answer}\n")
