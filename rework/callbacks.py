import torch
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
from bert_score import score as bertscore_score
from sentence_transformers import SentenceTransformer, util

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
                f"Пользователь: {ex.get('ticket_message', 'нет')}\n"
                "Оператор: "
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
            tqdm.write(f"[Входной текст]: {ex['text']}")
            tqdm.write(f"[Промт]: {prompt}")
            tqdm.write(f"[Эталонный ответ]: {ex.get('label', 'нет')}")
            for i, output in enumerate(outputs):
                answer = self.tokenizer.decode(output, skip_special_tokens=True)
                tqdm.write(f"[Beam {i + 1}]: {answer}")
            tqdm.write("=" * 100)
        model.train()


class AccuracyCallback(TrainerCallback):
    def __init__(self, tokenizer, init_timestamp, model_name):
        self.tokenizer = tokenizer
        self.init_timestamp = init_timestamp
        self.model_name = model_name
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

        from nltk.translate.meteor_score import meteor_score
        from sacrebleu.metrics import CHRF
        import Levenshtein

        self.meteor_score = meteor_score
        self.chrf_metric = CHRF()
        self.Levenshtein = Levenshtein
        self.sim_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

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

        self.metrics_logfile = (
            f"logs/metrics-log-{self.model_name}-{self.init_timestamp}.csv"
        )

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
        semantic_similarities = []

        all_preds_for_bertscore = []
        all_refs_for_bertscore = []

        for batch in tqdm(eval_dataloader, desc="Валидация (батчи)"):
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

                for pred in beams:
                    all_preds_for_bertscore.append(pred)
                    all_refs_for_bertscore.append(ref)

                    rs = self.rouge.score(ref, pred)
                    rouge1_list.append(rs["rouge1"].fmeasure)
                    rougeL_list.append(rs["rougeL"].fmeasure)
                    bleu_list.append(
                        sentence_bleu(
                            [ref.split()], pred.split(), smoothing_function=self.smooth
                        )
                    )
                    meteor_list.append(self.meteor_score([ref.split()], pred.split()))
                    chrf_list.append(
                        self.chrf_metric.sentence_score(pred, [ref]).score / 100.0
                    )
                    fuzzy_list.append(self.Levenshtein.ratio(pred, ref))
                    lenratio_list.append(len(pred) / (len(ref) + 1e-8))
                    preds.append(pred)
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

        try:
            P, R, F1 = bertscore_score(
                all_preds_for_bertscore,
                all_refs_for_bertscore,
                lang="ru",
                model_type="xlm-roberta-base",
            )
            bertscore_f1 = float(F1.mean())
        except Exception as e:
            tqdm.write(f"BERTScore failed: {e}")
            bertscore_f1 = 0.0

        for pred, ref in zip(all_preds_for_bertscore, all_refs_for_bertscore):
            emb_pred = self.sim_model.encode(pred, convert_to_tensor=True)
            emb_ref = self.sim_model.encode(ref, convert_to_tensor=True)
            sim = float(util.pytorch_cos_sim(emb_pred, emb_ref))
            semantic_similarities.append(sim)

        avg_semantic_similarity = (
            sum(semantic_similarities) / len(semantic_similarities)
            if semantic_similarities
            else 0.0
        )

        tqdm.write(f"✅ Accuracy @ step {state.global_step}: {acc:.4f}")
        tqdm.write(f"📊 ROUGE-1: {rouge1:.4f}, ROUGE-L: {rougeL:.4f}, BLEU: {bleu:.4f}")
        tqdm.write(
            f"🌟 METEOR: {meteor:.4f}, chrF: {chrf:.4f}, Fuzzy: {fuzzy:.4f}, Length ratio: {len_ratio:.4f}"
        )
        tqdm.write(f"🟢 Masked accuracy: {masked_acc:.4f}")
        tqdm.write(f"🧠 BERTScore(F1) (all beams): {bertscore_f1:.4f}")
        tqdm.write(f"🟡 Semantic similarity (MiniLM): {avg_semantic_similarity:.4f}")
        epoch = state.epoch or 0.0
        tqdm.write(f"Эпоха {epoch} завершена!")

        metrics_row = {
            "step": state.global_step,
            "epoch": epoch,
            "accuracy": acc,
            "rouge1": rouge1,
            "rougeL": rougeL,
            "bleu": bleu,
            "masked_accuracy": masked_acc,
            "meteor": meteor,
            "chrf": chrf,
            "fuzzy": fuzzy,
            "len_ratio": len_ratio,
            "bertscore_f1": bertscore_f1,
            "semantic_similarity": avg_semantic_similarity,
        }
        last_train_log = next(
            (log for log in reversed(state.log_history) if "loss" in log), {}
        )
        last_eval_log = next(
            (log for log in reversed(state.log_history) if "eval_loss" in log), {}
        )

        metrics_row.update(
            {
                "train_loss": last_train_log.get("loss"),
                "grad_norm": last_train_log.get("grad_norm"),
                "learning_rate": last_train_log.get("learning_rate"),
                "eval_loss": last_eval_log.get("eval_loss"),
                "eval_runtime": last_eval_log.get("eval_runtime"),
                "eval_samples_per_second": last_eval_log.get("eval_samples_per_second"),
                "eval_steps_per_second": last_eval_log.get("eval_steps_per_second"),
            }
        )

        log_exists = os.path.exists(self.metrics_logfile)
        with open(self.metrics_logfile, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_row.keys())
            if not log_exists:
                writer.writeheader()
            writer.writerow(metrics_row)

        state.log_history.append(metrics_row)
        model.train()
