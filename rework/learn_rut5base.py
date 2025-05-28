import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import os
import re

from callbacks import AccuracyCallback, LogCallback


def freeze_encoder(model, unfreeze_last_n=2):
    total_layers = len(model.encoder.block)
    for i, block in enumerate(model.encoder.block):
        requires_grad = i >= total_layers - unfreeze_last_n
        for param in block.parameters():
            param.requires_grad = requires_grad


def freeze_decoder(model, unfreeze_last_n=2):
    total_layers = len(model.decoder.block)
    for i, block in enumerate(model.decoder.block):
        requires_grad = i >= total_layers - unfreeze_last_n
        for param in block.parameters():
            param.requires_grad = requires_grad


# === Очистка и нормализация ответа ===
def clean_label(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\n\r]+", " ", text)
    text = text.replace("\xa0", " ")
    return text.strip()


# === Препроцессинг ===
def preprocess(example):
    input_text = example["text"]
    cleaned_label = clean_label(example["label"])
    tokenized = tokenizer(
        input_text,
        max_length=max_input,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenized_labels = tokenizer(
        cleaned_label,
        max_length=max_output,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
    tokenized["labels"] = tokenized_labels["input_ids"].squeeze(0)
    return tokenized


# === Основной блок ===
if __name__ == "__main__":
    with open("dialogue_dataset.json", encoding="utf-8") as f:
        full_data = json.load(f)

    train_data, test_data = train_test_split(
        full_data, test_size=0.1, random_state=24, shuffle=True
    )

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base")
    tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-base", legacy=False)

    freeze_encoder(model, unfreeze_last_n=2)
    freeze_decoder(model, unfreeze_last_n=2)

    max_input = 256
    max_output = 64

    tokenized_train = train_dataset.map(
        preprocess, remove_columns=train_dataset.column_names
    )
    tokenized_test = test_dataset.map(
        preprocess, remove_columns=test_dataset.column_names
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./rut5base-finetuned",
        num_train_epochs=10,
        learning_rate=3e-4,
        weight_decay=0.02,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        do_eval=True,
        logging_dir="./runs",
        fp16=False,
        bf16=False,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        callbacks=[
            LogCallback(tokenizer, test_data),
            AccuracyCallback(tokenizer),
        ],
    )

    trainer.train()

    trainer.save_model("./rut5base-finetuned")
    tokenizer.save_pretrained("./rut5base-finetuned")
    print("\n✅ Обучение завершено и модель сохранена")

    # === Графики ===
    logs = trainer.state.log_history
    steps, loss, eval_loss, acc, rouge1, rougeL, bleu = [], [], [], [], [], [], []
    lr = []
    for entry in logs:
        if "loss" in entry:
            steps.append(entry["step"])
            loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
        if "learning_rate" in entry:
            lr.append(entry["learning_rate"])
        if "accuracy" in entry:
            acc.append((entry["step"], entry["accuracy"]))
        if "rouge1" in entry:
            rouge1.append((entry["step"], entry["rouge1"]))
        if "rougeL" in entry:
            rougeL.append((entry["step"], entry["rougeL"]))
        if "bleu" in entry:
            bleu.append((entry["step"], entry["bleu"]))

    # Получаем параметры обучения
    max_steps = trainer.state.max_steps or (max(steps) if steps else 1)
    num_epochs = getattr(trainer.state, "num_train_epochs", 1) or 1

    def step_to_epoch(step):
        steps_per_epoch = max_steps / num_epochs
        return step / steps_per_epoch

    def prepend_zero(xs, ys):
        return [0] + list(xs), [0] + list(ys)

    plt.figure(figsize=(18, 6))

    # === Loss Plot ===
    plt.subplot(1, 3, 1)
    epoch_steps = [step_to_epoch(s) for s in steps]
    x_loss, y_loss = prepend_zero(epoch_steps, loss)
    plt.plot(x_loss, y_loss, label="Train Loss")
    if eval_loss:
        eval_epoch_steps = epoch_steps[: len(eval_loss)]
        x_eval, y_eval = prepend_zero(eval_epoch_steps, eval_loss)
        plt.plot(x_eval, y_eval, label="Eval Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    # === Metrics ===
    plt.subplot(1, 3, 2)
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    if acc:
        acc_steps, acc_vals = zip(*acc)
        acc_steps, acc_vals = prepend_zero(
            [step_to_epoch(s) for s in acc_steps], acc_vals
        )
        plt.plot(acc_steps, acc_vals, label="Accuracy")
    if bleu:
        bleu_steps, bleu_vals = zip(*bleu)
        bleu_steps, bleu_vals = prepend_zero(
            [step_to_epoch(s) for s in bleu_steps], bleu_vals
        )
        plt.plot(bleu_steps, bleu_vals, label="BLEU")
    if rouge1:
        r1_steps, r1_vals = zip(*rouge1)
        r1_steps, r1_vals = prepend_zero([step_to_epoch(s) for s in r1_steps], r1_vals)
        plt.plot(r1_steps, r1_vals, label="ROUGE-1")
    if rougeL:
        rL_steps, rL_vals = zip(*rougeL)
        rL_steps, rL_vals = prepend_zero([step_to_epoch(s) for s in rL_steps], rL_vals)
        plt.plot(rL_steps, rL_vals, label="ROUGE-L")
    plt.title("Text Metrics")
    plt.legend()
    plt.grid(True)

    # === Learning Rate Plot ===
    plt.subplot(1, 3, 3)
    if lr:
        x_lr = epoch_steps[: len(lr)]
        x_lr, y_lr = prepend_zero(x_lr, lr)
        plt.plot(x_lr, y_lr, label="Learning Rate")
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs("train_results", exist_ok=True)
    plt.savefig("train_results/metrics-rut5-base.png")

    print("\n📊 Графики сохранены в train_results/metrics-rut5-base.png")
