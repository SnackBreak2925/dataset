# 🧠 Дообучение модели с категорией запроса (category_title)
import json
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    DataCollatorForSeq2Seq,
    EncoderDecoderCache,
)
from datasets import Dataset
import matplotlib.pyplot as plt


# 🔄 Callback для печати примеров
class LogCallback(TrainerCallback):
    def __init__(self, tokenizer, raw_data):
        self.tokenizer = tokenizer
        self.examples = raw_data[:5]

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 500 == 0:
            print(f"\n=== Проверка модели на шаге {state.global_step} ===")
            kwargs["model"].eval()
            for ex in self.examples:
                input_ids = self.tokenizer.encode(
                    ex["text"], return_tensors="pt", max_length=256, truncation=True
                ).to(kwargs["model"].device)
                outputs = kwargs["model"].generate(
                    input_ids,
                    max_length=64,
                    past_key_values=EncoderDecoderCache.from_legacy_cache(None),
                )
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"\n[Категория]: {ex.get('category_title', 'нет')}")
                print(f"[Входной текст]:\n{ex['text']}\n")
                print(f"[Эталонный ответ]: {ex['label']}\n[Ответ модели]: {answer}\n")
            kwargs["model"].train()


# ❄️ Заморозка нижних слоёв
def freeze_layers(model, num_layers=6):
    for i in range(num_layers):
        for param in model.encoder.block[i].parameters():
            param.requires_grad = False
        for param in model.decoder.block[i].parameters():
            param.requires_grad = False


def preprocess(example):
    # Вставка категории в промпт
    category = example.get("category_title", "неизвестно")
    input_text = f"Категория: {category}\n{example['text']}"
    tokenized = tokenizer(
        input_text,
        max_length=max_input,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenized_labels = tokenizer(
        example["label"],
        max_length=max_output,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
    tokenized["labels"] = tokenized_labels["input_ids"].squeeze(0)
    return tokenized


if __name__ == "__main__":
    with open("dialogue_dataset.json", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Добавляем категорию и сохраняем для callback
    dataset_for_callback = [
        {
            "text": f"Категория: {d.get('category_title', 'неизвестно')}\n{d['text']}",
            "label": d["label"],
            "category_title": d.get("category_title", "неизвестно"),
        }
        for d in raw_data
    ]

    raw_dataset = Dataset.from_list(dataset_for_callback).train_test_split(
        test_size=0.1
    )

    model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base")
    tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-base", legacy=False)

    freeze_layers(model)

    max_input = 256
    max_output = 64

    tokenized_dataset = raw_dataset.map(
        preprocess, remove_columns=raw_dataset["train"].column_names
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./rut5base-finetuned",
        num_train_epochs=7,
        learning_rate=3e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        logging_steps=100,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        do_eval=True,
        eval_steps=500,
        logging_dir="./runs",
        fp16=False,
        bf16=False,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        callbacks=[LogCallback(tokenizer, dataset_for_callback)],
    )

    trainer.train()

    trainer.save_model("./rut5base-finetuned")
    tokenizer.save_pretrained("./rut5base-finetuned")
    print("✅ Обучение завершено и модель сохранена")

    # 📈 Построение графика потерь
    logs = trainer.state.log_history
    steps, loss, eval_loss = [], [], []
    for entry in logs:
        if "loss" in entry:
            steps.append(entry["step"])
            loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])

    plt.figure(figsize=(10, 5))
    plt.plot(steps, loss, label="Train Loss")
    if eval_loss:
        plt.plot(steps[: len(eval_loss)], eval_loss, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Evaluation Loss")
    plt.grid()
    plt.savefig("loss_plot.png")
    print("📊 График потерь сохранён как loss_plot.png")
