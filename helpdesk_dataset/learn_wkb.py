from datasets import load_from_disk, concatenate_datasets
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import torch
import os

# --- Фиксированные тестовые вопросы
test_questions = [
    "Как узнать статус заявки?",
    "Как восстановить пароль?",
    "Куда обращаться за поддержкой программного обеспечения?",
    "Здравствуйте, хотелось бы получить ПО Microsoft студенческий",
]


# --- Функция для генерации ответа на один вопрос
def generate_answer(model, tokenizer, question):
    prompt = f"Тикет: {question}\nОтвет:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    device = next(model.parameters()).device  # <<< Получаем устройство модели
    inputs = {
        k: v.to(device) for k, v in inputs.items()
    }  # <<< Переносим всё на устройство

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=5,
            early_stopping=True,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# --- Наш кастомный Callback
class InferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, test_questions, eval_steps=1000):
        self.tokenizer = tokenizer
        self.test_questions = test_questions
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Проверяем: настал ли нужный шаг для проверки
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            print(
                f"\n\n=== Проверка модели на тестовых вопросах (шаг {state.global_step}) ==="
            )
            model = kwargs["model"].eval()
            for idx, question in enumerate(self.test_questions, 1):
                answer = generate_answer(model, self.tokenizer, question)
                print(f"{idx}. Вопрос: {question}")
                print(f"   Ответ: {answer}\n")
            model.train()

# 1. Загружаем датасет
dataset = load_from_disk(os.path.join(os.getenv("PWD"), "ticket_replies_dataset"))

full_dataset = dataset

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
    num_train_epochs=6,
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
    eval_dataset=tokenized_dataset.select(range(100)),
    callbacks=[
        InferenceCallback(tokenizer, test_questions, eval_steps=200)
    ],  # небольшой валидационный кусочек
)

# print(full_dataset[0])
# print(full_dataset[-1])
# print(tokenized_dataset[0])
# print(tokenized_dataset[-1])

# 7. Запускаем обучение
trainer.train()

# 8. Сохраняем модель
save_path = os.path.join(os.getenv("PWD"), "rut5", "rut5-small-finetuned")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
