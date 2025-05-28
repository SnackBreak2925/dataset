import torch
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm

import nltk

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


class LogCallback(TrainerCallback):
    def __init__(self, tokenizer, raw_data):
        self.tokenizer = tokenizer
        self.examples = raw_data[:5]

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        for ex in self.examples:
            prompt = (
                f"Категория: {ex.get('category_title', 'нет')}\n"
                f"Тема: {ex.get('subject', 'нет')}\n"
                f"Пользователь: {ex.get('ticket_message', 'нет')}"
            )
            input_ids = self.tokenizer.encode(
                prompt, return_tensors="pt", max_length=256, truncation=True
            ).to(model.device)
            outputs = model.generate(
                input_ids,
                max_length=64,
                num_beams=5,
                num_return_sequences=5,
                early_stopping=True,
            )
            tqdm.write("=" * 100)
            tqdm.write(f"[Категория]: {ex.get('category_title', 'нет')}")
            tqdm.write(f"[Тема]: {ex.get('subject', 'нет')}")
            tqdm.write(f"[Первичное сообщение]: {ex.get('ticket_message', 'нет')}")
            tqdm.write(f"[Промт]: {prompt}")
            tqdm.write(f"[Входной текст]: {ex['text']}")
            tqdm.write(f"[Эталонный ответ]: {ex.get('label', 'нет')}")
            for i, output in enumerate(outputs):
                answer = self.tokenizer.decode(output, skip_special_tokens=True)
                tqdm.write(f"\n[Beam {i + 1}]: {answer}")
            tqdm.write("=" * 100)
            tqdm.write("")
        model.train()


class AccuracyCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

        # Добавляем импортируемые метрики:
        from nltk.translate.meteor_score import meteor_score
        from sacrebleu.metrics import CHRF
        import Levenshtein

        self.meteor_score = meteor_score
        self.chrf_metric = CHRF()
        self.Levenshtein = Levenshtein

    @staticmethod
    def masked_accuracy(logits, labels):
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = ((preds == labels) & mask).sum().item()
        total = mask.sum().item()
        return correct / total if total > 0 else 0.0

    def on_evaluate(self, args, state, control, **kwargs):
        import csv
        import os

        self.metrics_logfile = "metrics_log.csv"

        model = kwargs["model"]
        eval_dataloader = kwargs["eval_dataloader"]
        model.eval()
        preds, labels = [], []
        (
            rouge1_list,
            rougeL_list,
            bleu_list,
            meteor_list,
            chrf_list,
            fuzzy_list,
            lenratio_list,
        ) = [], [], [], [], [], [], []
        masked_accs = []

        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            label_ids = batch["labels"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=label_ids
                )
                logits = outputs.logits
            masked_accs.append(self.masked_accuracy(logits, label_ids))

            num_beams = 5
            gen_outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True,
            )
            batch_size = input_ids.shape[0]
            decoded_labels = self.tokenizer.batch_decode(
                label_ids, skip_special_tokens=True
            )
            decoded_preds = self.tokenizer.batch_decode(
                gen_outputs, skip_special_tokens=True
            )

            for i in range(batch_size):
                ref = decoded_labels[i]
                beams = decoded_preds[i * num_beams : (i + 1) * num_beams]

                best_rouge1 = best_rougeL = best_bleu = best_meteor = best_chrf = (
                    best_fuzzy
                ) = best_lenratio = 0
                for pred in beams:
                    rs = self.rouge.score(ref, pred)
                    curr_rouge1 = rs["rouge1"].fmeasure
                    curr_rougeL = rs["rougeL"].fmeasure
                    curr_bleu = sentence_bleu(
                        [ref.split()], pred.split(), smoothing_function=self.smooth
                    )
                    curr_meteor = self.meteor_score([ref.split()], pred.split())
                    curr_chrf = (
                        self.chrf_metric.sentence_score(pred, [ref]).score / 100.0
                    )
                    curr_fuzzy = self.Levenshtein.ratio(pred, ref)
                    curr_lenratio = len(pred) / (len(ref) + 1e-8)

                    best_rouge1 = max(best_rouge1, curr_rouge1)
                    best_rougeL = max(best_rougeL, curr_rougeL)
                    best_bleu = max(best_bleu, curr_bleu)
                    best_meteor = max(best_meteor, curr_meteor)
                    best_chrf = max(best_chrf, curr_chrf)
                    best_fuzzy = max(best_fuzzy, curr_fuzzy)
                    best_lenratio = max(best_lenratio, curr_lenratio)

                rouge1_list.append(best_rouge1)
                rougeL_list.append(best_rougeL)
                bleu_list.append(best_bleu)
                meteor_list.append(best_meteor)
                chrf_list.append(best_chrf)
                fuzzy_list.append(best_fuzzy)
                lenratio_list.append(best_lenratio)
                preds.append(beams[0])
                labels.append(ref)

        acc = accuracy_score(labels, preds)
        rouge1 = sum(rouge1_list) / len(rouge1_list)
        rougeL = sum(rougeL_list) / len(rougeL_list)
        bleu = sum(bleu_list) / len(bleu_list)
        masked_acc = sum(masked_accs) / len(masked_accs)
        meteor = sum(meteor_list) / len(meteor_list)
        chrf = sum(chrf_list) / len(chrf_list)
        fuzzy = sum(fuzzy_list) / len(fuzzy_list)
        len_ratio = sum(lenratio_list) / len(lenratio_list)

        tqdm.write(f"✅ Accuracy @ step {state.global_step}: {acc:.4f}")
        tqdm.write(f"📊 ROUGE-1: {rouge1:.4f}, ROUGE-L: {rougeL:.4f}, BLEU: {bleu:.4f}")
        tqdm.write(
            f"🌟 METEOR: {meteor:.4f}, chrF: {chrf:.4f}, Fuzzy: {fuzzy:.4f}, Length ratio: {len_ratio:.4f}"
        )
        tqdm.write(f"🟢 Masked accuracy: {masked_acc:.4f}")
        tqdm.write(f"Эпоха {state.epoch} завершена!")

        # Сохраняем все метрики в csv
        metrics_row = {
            "step": state.global_step,
            "epoch": state.epoch,
            "accuracy": acc,
            "rouge1": rouge1,
            "rougeL": rougeL,
            "bleu": bleu,
            "masked_accuracy": masked_acc,
            "meteor": meteor,
            "chrf": chrf,
            "fuzzy": fuzzy,
            "len_ratio": len_ratio,
        }
        log_exists = os.path.exists(self.metrics_logfile)
        with open(self.metrics_logfile, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_row.keys())
            if not log_exists:
                writer.writeheader()
            writer.writerow(metrics_row)

        state.log_history.append(metrics_row)
        model.train()
