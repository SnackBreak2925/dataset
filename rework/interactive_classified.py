# interactive_classified.py
# 🧠 Классификация + генерация ответа от модели RuT5-base

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Загрузка модели
model_path = "./rut5base-finetuned"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Простая эвристика классификации
def classify_topic(text):
    t = text.lower()
    if any(x in t for x in ["vpn", "удаленный доступ", "документооборот"]):
        return "доступ"
    elif any(x in t for x in ["ошибка", "не подключается", "не работает", "1004"]):
        return "ошибка"
    elif any(
        x in t for x in ["mathcad", "office", "windows", "visual studio", "лицензи"]
    ):
        return "ПО"
    elif any(x in t for x in ["курс", "фдо", "программировани", "подписан"]):
        return "курсы"
    elif any(x in t for x in ["почта", "mail", "email"]):
        return "почта"
    else:
        return "другое"


print("\U0001f4ac Введите сообщение пользователя (или 'exit'):")

while True:
    user_input = input("\nПользователь: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    topic = classify_topic(user_input)
    prompt = f"ТЕМА: {topic}\nЗапрос: {user_input.strip()}"

    input_ids = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=64)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🤖 Модель: {response}")
