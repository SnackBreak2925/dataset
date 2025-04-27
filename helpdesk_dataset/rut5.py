


from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
# 1. Загружаем обученную модель
model_path = os.path.join(
    os.getenv("PWD"), "rut5", "rut5-small-finetuned"
)  # туда, куда ты сохранил после обучения

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()


# 2. Функция для подготовки правильного промпта
def prepare_prompt(question):
    return f"Тикет: {question}\nОтвет:"


# 3. Функция для генерации ответа на один вопрос
def generate_answer(prompt, max_input_length=512, max_output_length=128):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_output_length,
            num_beams=5,
            early_stopping=True,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# 4. Список вопросов для теста
questions = [
    "Как узнать статус заявки?",
    "Что делать, если потерял пароль от кабинета?",
    "Как оформить заявку на выдачу пропуска?",
    "Куда обращаться за поддержкой по программному обеспечению?",
    "Какие документы нужны для получения справки?",
    "Как изменить личные данные в профиле?",
    "Можно ли продлить доступ к корпоративной почте после увольнения?",
    "Как скачать сертификат о прохождении обучения?",
    "Как восстановить доступ в систему?",
    "Какой срок обработки заявки на ПО Microsoft?",
]

# 5. Генерация и вывод ответов
print("\n--- Результаты тестирования ---\n")
for idx, question in enumerate(questions, 1):
    prompt = prepare_prompt(question)
    answer = generate_answer(prompt)
    print(f"{idx}. Вопрос: {question}")
    print(f"   Ответ: {answer}\n")
