# 🧠 Дообучение модели
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base")
tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-base")
print("Модель загружена. Здесь будет обучение.")
