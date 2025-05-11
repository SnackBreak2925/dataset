# rework/interactive.py

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Загрузка дообученной модели
model_path = "./rut5base-finetuned"  # Укажи свой последний чекпоинт
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("💬 Введите сообщение пользователя (или 'exit'):")

while True:
    user_input = input("Пользователь: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    prompt = f"Первичное сообщение: {user_input.strip()}"
    input_ids = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=64)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🤖 Модель: {response}\n")
