from datasets import load_from_disk, concatenate_datasets
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
import torch
import os

# 1. Загружаем датасет
dataset = load_from_disk(os.path.join(os.getenv("PWD"), "ticket_replies_dataset"))

# Удаляем ненужное поле metadata, чтобы объединение прошло без ошибок
dataset["tickets"] = dataset["tickets"].remove_columns(["metadata"])
dataset["knowledge_base"] = dataset["knowledge_base"].remove_columns(["metadata"])

# Объединяем сплиты
full_dataset = concatenate_datasets([dataset["tickets"], dataset["knowledge_base"]])

# 2. Загружаем токенизатор и модель
tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-small")
model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-small")


# 3. Функция препроцессинга
def preprocess_function(examples):
    inputs = examples["text"]
    targets = []

    for i in range(len(inputs)):
        if "reply_message" in examples and examples["reply_message"][i] is not None:
            targets.append(examples["reply_message"][i])
        elif "instruction" in examples and examples["instruction"][i] is not None:
            targets.append(examples["instruction"][i])
        else:
            targets.append("")

    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length"
    )

    # Используем современный способ передачи target-текстов
    labels = tokenizer(
        text_target=targets, max_length=128, truncation=True, padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 4. Применяем токенизацию ко всему датасету
tokenized_dataset = full_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=full_dataset.column_names,
)

# 5. Аргументы обучения
training_args = TrainingArguments(
    output_dir=os.path.join(os.getenv("PWD"), "rut5", "results"),
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir=os.path.join(os.getenv("PWD"), "rut5", "logs"),
    logging_steps=50,
    learning_rate=2e-5,
    warmup_steps=200,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    # fp16=torch.cuda.is_available(),  # Использовать fp16 если доступно
    fp16=False,
    report_to="none",  # Отключаем логгирование в wandb
)

# 6. Создаем тренер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(
        range(100)
    ),  # небольшой валидационный кусочек
)

print(full_dataset[0])
print(tokenized_dataset[0])
# 7. Запускаем обучение
trainer.train()

# 8. Сохраняем модель
save_path = os.path.join(os.getenv("PWD"), "rut5", "rut5-small-finetuned")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
