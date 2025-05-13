import json
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import os
import re


# === Кастомные Callbacks ===
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


class AccuracyCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        eval_dataloader = kwargs["eval_dataloader"]
        model.eval()
        preds, labels = [], []
        rouge1_list, rougeL_list, bleu_list = [], [], []

        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            label_ids = batch["labels"].to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids, attention_mask=attention_mask, max_length=64
                )

            decoded_preds = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            decoded_labels = self.tokenizer.batch_decode(
                label_ids, skip_special_tokens=True
            )

            preds.extend(decoded_preds)
            labels.extend(decoded_labels)

            for pred, ref in zip(decoded_preds, decoded_labels):
                rs = self.rouge.score(ref, pred)
                rouge1_list.append(rs["rouge1"].fmeasure)
                rougeL_list.append(rs["rougeL"].fmeasure)
                bleu_list.append(
                    sentence_bleu(
                        [ref.split()], pred.split(), smoothing_function=self.smooth
                    )
                )

        acc = accuracy_score(labels, preds)
        rouge1 = sum(rouge1_list) / len(rouge1_list)
        rougeL = sum(rougeL_list) / len(rougeL_list)
        bleu = sum(bleu_list) / len(bleu_list)

        print(f"\n✅ Accuracy @ step {state.global_step}: {acc:.4f}")
        print(f"📊 ROUGE-1: {rouge1:.4f}, ROUGE-L: {rougeL:.4f}, BLEU: {bleu:.4f}")

        state.log_history.append(
            {
                "step": state.global_step,
                "accuracy": acc,
                "rouge1": rouge1,
                "rougeL": rougeL,
                "bleu": bleu,
            }
        )


def freeze_encoder(model, unfreeze_last_n=2):
    total_layers = len(model.encoder.block)
    for i, block in enumerate(model.encoder.block):
        requires_grad = i >= total_layers - unfreeze_last_n
        for param in block.parameters():
            param.requires_grad = requires_grad


def freeze_decoder_layers(model, num_layers=6):
    for i in range(num_layers):
        for param in model.decoder.block[i].parameters():
            param.requires_grad = False


# === Очистка и нормализация ответа ===
def clean_label(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\n\r]+", " ", text)
    text = text.replace("\xa0", " ")
    return text.strip()


# === Препроцессинг ===
def preprocess(example):
    category = example.get("category_title", "неизвестно")
    input_text = f"Категория: {category}\n{example['text']}"
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
        raw_data = json.load(f)

    dataset_for_callback = [
        {
            "text": f"Категория: {d.get('category_title', 'неизвестно')}\n{d['text']}",
            "label": clean_label(d["label"]),
            "category_title": d.get("category_title", "неизвестно"),
        }
        for d in raw_data
    ]

    raw_dataset = Dataset.from_list(dataset_for_callback).train_test_split(
        test_size=0.1
    )

    model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base")
    tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-base", legacy=False)

    freeze_encoder(model, unfreeze_last_n=2)

    max_input = 256
    max_output = 64

    tokenized_dataset = raw_dataset.map(
        preprocess, remove_columns=raw_dataset["train"].column_names
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./rut5base-finetuned",
        num_train_epochs=20,
        learning_rate=1e-4,
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
        label_smoothing_factor=0.1,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        callbacks=[
        LogCallback(tokenizer, dataset_for_callback),
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

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(steps, loss, label="Train Loss")
    if eval_loss:
        plt.plot(steps[: len(eval_loss)], eval_loss, label="Eval Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    if acc:
        acc_steps, acc_vals = zip(*acc)
        plt.plot(acc_steps, acc_vals, label="Accuracy")
    if bleu:
        bleu_steps, bleu_vals = zip(*bleu)
        plt.plot(bleu_steps, bleu_vals, label="BLEU")
    if rouge1:
        r1_steps, r1_vals = zip(*rouge1)
        plt.plot(r1_steps, r1_vals, label="ROUGE-1")
    if rougeL:
        rL_steps, rL_vals = zip(*rougeL)
        plt.plot(rL_steps, rL_vals, label="ROUGE-L")
    plt.title("Text Metrics")
    plt.legend()

    plt.subplot(1, 3, 3)
    if lr:
        plt.plot(steps[: len(lr)], lr, label="Learning Rate")
    plt.title("Learning Rate")
    plt.legend()

    plt.tight_layout()
    os.makedirs("train_results", exist_ok=True)
    plt.savefig("train_results/metrics.png")
    plt.show()

    print("\n📊 Графики сохранены в train_results/metrics.png")
