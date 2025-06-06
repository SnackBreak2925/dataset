import torch
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
from bert_score import score as bertscore_score
from sentence_transformers import SentenceTransformer, util
from helpers.rag_pipeline import RagPipeline
import nltk
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import CHRF
import Levenshtein
import csv
import os
import numpy as np
from helpers.cleaner import postprocess_answer


try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

TQDM_PREFIX = "[METRICS] "


def tlog(msg):
    tqdm.write(f"{TQDM_PREFIX}{msg}")


def print_metrics(prefix, metrics, keys):
    for key in keys:
        tlog(f"{prefix}{key}: {metrics[key]:.4f}")


def compute_seq2seq_metrics(
    preds, refs, *, rouge, meteor_score, chrf_metric, fuzzy_metric, smooth
):
    (
        rouge1_list,
        rougeL_list,
        bleu_list,
        meteor_list,
        chrf_list,
        fuzzy_list,
        lenratio_list,
    ) = [], [], [], [], [], [], []
    for pred, ref in zip(preds, refs):
        rs = rouge.score(ref, pred)
        rouge1_list.append(rs["rouge1"].fmeasure)
        rougeL_list.append(rs["rougeL"].fmeasure)
        bleu_list.append(
            sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
        )
        meteor_list.append(meteor_score([ref.split()], pred.split()))
        chrf_list.append(chrf_metric.sentence_score(pred, [ref]).score / 100.0)
        fuzzy_list.append(fuzzy_metric(pred, ref))
        lenratio_list.append(len(pred) / (len(ref) + 1e-8))
    return dict(
        rouge1=rouge1_list,
        rougeL=rougeL_list,
        bleu=bleu_list,
        meteor=meteor_list,
        chrf=chrf_list,
        fuzzy=fuzzy_list,
        len_ratio=lenratio_list,
    )


def compute_semantic_similarity(preds, refs, sim_model):
    return [
        float(
            util.pytorch_cos_sim(
                sim_model.encode(pred, convert_to_tensor=True, show_progress_bar=False),
                sim_model.encode(ref, convert_to_tensor=True, show_progress_bar=False),
            )
        )
        for pred, ref in zip(preds, refs)
    ]


def aggregate_metric(lst):
    return sum(lst) / len(lst) if lst else 0.0


def compute_bertscore(preds, refs, lang="ru", model_type="xlm-roberta-base"):
    try:
        _, _, F1 = bertscore_score(
            preds, refs, lang=lang, model_type=model_type, verbose=False
        )
        return float(F1.mean())
    except Exception as e:
        tlog(f"BERTScore failed: {e}")
        return 0.0


def compute_masked_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    mask = labels != -100
    correct = ((preds == labels) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def clean_for_decode(arr, vocab_size):
    return [int(t) for t in arr if 0 <= int(t) < vocab_size]


class AccuracyCallback(TrainerCallback):
    def __init__(self, tokenizer, init_timestamp, model_name, raw_test_data=None):
        self.tokenizer = tokenizer
        self.init_timestamp = init_timestamp
        self.model_name = model_name
        self.raw_test_data = raw_test_data
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        self.smooth = SmoothingFunction().method1
        self.meteor_score = meteor_score
        self.chrf_metric = CHRF()
        self.Levenshtein = Levenshtein
        self.sim_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2", device="cpu"
        )
        self.metrics_logfile = (
            f"logs/metrics-log-{self.model_name}-{self.init_timestamp}.csv"
        )

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        eval_dataloader = kwargs["eval_dataloader"]
        model.eval()
        self.rag_pipe = RagPipeline(
            model=kwargs["model"], tokenizer=self.tokenizer, sim_model=self.sim_model
        )

        all_preds_for_bertscore, all_refs_for_bertscore = [], []
        preds, labels = [], []
        masked_accs = []
        decoded_labels_total = []
        decoded_preds_total = []
        num_beams = 5

        total = len(self.raw_test_data)
        running_idx = 0

        with tqdm(total=total, desc="Валидация (примеры)") as val_bar:
            for batch_num, batch in enumerate(eval_dataloader):
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                label_ids = batch["labels"].to(model.device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=label_ids,
                    )
                    logits = outputs.logits
                masked_acc = compute_masked_accuracy(logits, label_ids)
                masked_accs.append(masked_acc)

                try:
                    gen_outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=64,
                        num_beams=num_beams,
                        num_return_sequences=num_beams,
                    )
                except Exception:
                    gen_outputs = model.generate(
                        input_ids,
                        max_new_tokens=128,
                        attention_mask=attention_mask,
                        num_beams=num_beams,
                        num_return_sequences=num_beams,
                    )

                batch_size = input_ids.shape[0]
                vocab_size = len(self.tokenizer)

                if isinstance(label_ids, torch.Tensor):
                    labels_np = label_ids.cpu().numpy()
                else:
                    labels_np = np.array(label_ids)

                safe_labels = [clean_for_decode(seq, vocab_size) for seq in labels_np]

                if isinstance(gen_outputs, torch.Tensor):
                    gen_np = gen_outputs.cpu().numpy()
                else:
                    gen_np = np.array(gen_outputs)

                safe_preds = [clean_for_decode(seq, vocab_size) for seq in gen_np]

                decoded_labels = [
                    postprocess_answer(lbl)
                    for lbl in self.tokenizer.batch_decode(safe_labels, skip_special_tokens=True)
                ]
                decoded_preds = [
                    postprocess_answer(pred)
                    for pred in self.tokenizer.batch_decode(safe_preds, skip_special_tokens=True)
                ]

                for i in range(batch_size):
                    ref = decoded_labels[i]
                    beams = decoded_preds[i * num_beams : (i + 1) * num_beams]
                    for pred in beams:
                        all_preds_for_bertscore.append(pred)
                        all_refs_for_bertscore.append(ref)
                        preds.append(pred)
                        labels.append(ref)
                    running_idx += 1
                    val_bar.update(1)
                decoded_labels_total.extend(decoded_labels)
                decoded_preds_total.extend(decoded_preds)

        tqdm.write("compute_seq2seq_metrics...")
        seq2seq_metrics = compute_seq2seq_metrics(
            preds,
            labels,
            rouge=self.rouge,
            meteor_score=self.meteor_score,
            chrf_metric=self.chrf_metric,
            fuzzy_metric=self.Levenshtein.ratio,
            smooth=self.smooth,
        )
        tqdm.write("compute_other_metrics...")
        masked_acc = aggregate_metric(masked_accs)
        acc = accuracy_score(labels, preds)
        bertscore_f1 = compute_bertscore(
            all_preds_for_bertscore, all_refs_for_bertscore
        )
        semantic_similarities = compute_semantic_similarity(
            all_preds_for_bertscore, all_refs_for_bertscore, self.sim_model
        )
        avg_semantic_similarity = aggregate_metric(semantic_similarities)

        tqdm.write("compute_rag...")

        BATCH_SIZE = 8
        TOP_K = 5
        rag_beams = []
        rag_refs = []

        with tqdm(total=total, desc="Валидация (RAG)") as val_bar:
            for batch_start in range(0, len(decoded_labels_total), BATCH_SIZE):
                batch_examples = self.raw_test_data[
                    batch_start : batch_start + BATCH_SIZE
                ]
                batch_labels = decoded_labels_total[
                    batch_start : batch_start + BATCH_SIZE
                ]

                questions = [ex["ticket_message"] for ex in batch_examples]
                category = self.rag_pipe.auto_detect_category(questions[0])
                if not category:
                    for ref in batch_labels:
                        rag_beams.append(["[Категория не определена]"] * TOP_K)
                        rag_refs.append([ref] * TOP_K)
                    val_bar.update(len(batch_examples))
                    continue

                all_hits = self.rag_pipe.batch_retrieve_contexts(
                    category, questions, top_k=TOP_K
                )

                prompts = []
                prompt_map = []
                adaptive_threshold = 0.1

                for q_idx, (hits, ref) in enumerate(zip(all_hits, batch_labels)):
                    cnt = 0
                    for hit in hits:
                        if hit["score"] < adaptive_threshold:
                            continue
                        idx = hit["corpus_id"]
                        ctx = self.rag_pipe.category_kbs[category][idx]
                        prompt = (
                            f"Категория: {category}\n"
                            f"Контекст: {ctx}\n"
                            f"Вопрос: {questions[q_idx]}\n"
                            "Оператор: "
                        )
                        prompts.append(prompt)
                        prompt_map.append((q_idx, ref, ctx, hit["score"]))
                        cnt += 1
                    if cnt == 0:
                        rag_beams.append(["[Нет релевантных контекстов]"] * TOP_K)
                        rag_refs.append([ref] * TOP_K)
                if not prompts:
                    val_bar.update(len(batch_examples))
                    continue

                answers = self.rag_pipe.batched_generate_answers(
                    prompts, batch_size=BATCH_SIZE
                )

                from collections import defaultdict

                responses_by_question = defaultdict(list)
                for (q_idx, ref, ctx, score), ans in zip(prompt_map, answers):
                    responses_by_question[q_idx].append(ans)
                for q_idx, ref in enumerate(batch_labels):
                    beams = responses_by_question.get(q_idx, [])
                    if len(beams) < TOP_K:
                        beams += ["[Нет ответа]"] * (TOP_K - len(beams))
                    rag_beams.append(beams)
                    rag_refs.append([ref] * TOP_K)
                val_bar.update(len(batch_examples))

        flat_rag_beams = [ans for answers in rag_beams for ans in answers]
        flat_rag_refs = [ref for refs in rag_refs for ref in refs]

        rag_metrics = compute_seq2seq_metrics(
            flat_rag_beams,
            flat_rag_refs,
            rouge=self.rouge,
            meteor_score=self.meteor_score,
            chrf_metric=self.chrf_metric,
            fuzzy_metric=self.Levenshtein.ratio,
            smooth=self.smooth,
        )
        rag_semantic_list = compute_semantic_similarity(
            flat_rag_beams, flat_rag_refs, self.sim_model
        )
        rag_bertscore_f1 = compute_bertscore(flat_rag_beams, flat_rag_refs)

        metrics_row = {
            "step": state.global_step,
            "epoch": state.epoch or 0.0,
            "accuracy": acc,
            "rouge1": aggregate_metric(seq2seq_metrics["rouge1"]),
            "rougeL": aggregate_metric(seq2seq_metrics["rougeL"]),
            "bleu": aggregate_metric(seq2seq_metrics["bleu"]),
            "masked_accuracy": masked_acc,
            "meteor": aggregate_metric(seq2seq_metrics["meteor"]),
            "chrf": aggregate_metric(seq2seq_metrics["chrf"]),
            "fuzzy": aggregate_metric(seq2seq_metrics["fuzzy"]),
            "len_ratio": aggregate_metric(seq2seq_metrics["len_ratio"]),
            "bertscore_f1": bertscore_f1,
            "semantic_similarity": avg_semantic_similarity,
            "rag_rouge1": aggregate_metric(rag_metrics["rouge1"]),
            "rag_rougeL": aggregate_metric(rag_metrics["rougeL"]),
            "rag_bleu": aggregate_metric(rag_metrics["bleu"]),
            "rag_meteor": aggregate_metric(rag_metrics["meteor"]),
            "rag_chrf": aggregate_metric(rag_metrics["chrf"]),
            "rag_fuzzy": aggregate_metric(rag_metrics["fuzzy"]),
            "rag_len_ratio": aggregate_metric(rag_metrics["len_ratio"]),
            "rag_bertscore_f1": rag_bertscore_f1,
            "rag_semantic_similarity": aggregate_metric(rag_semantic_list),
        }
        self.save_and_log_metrics(metrics_row, state)

    def save_and_log_metrics(self, metrics_row, state):
        last_train_log = next(
            (log for log in reversed(state.log_history) if "loss" in log), {}
        )
        last_eval_log = next(
            (log for log in reversed(state.log_history) if "eval_loss" in log), {}
        )
        metrics_row.update(
            {
                "train_loss": last_train_log.get("loss", 0.0),
                "grad_norm": last_train_log.get("grad_norm", 0.0),
                "learning_rate": last_train_log.get("learning_rate", 0.0),
                "eval_loss": last_eval_log.get("eval_loss", 0.0),
                "eval_runtime": last_eval_log.get("eval_runtime", 0.0),
                "eval_samples_per_second": last_eval_log.get(
                    "eval_samples_per_second", 0.0
                ),
                "eval_steps_per_second": last_eval_log.get(
                    "eval_steps_per_second", 0.0
                ),
            }
        )
        log_exists = os.path.exists(self.metrics_logfile)
        with open(self.metrics_logfile, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_row.keys())
            if not log_exists:
                writer.writeheader()
            writer.writerow(metrics_row)
        state.log_history.append(metrics_row)
        print_metrics(
            "RAG ",
            metrics_row,
            [
                "rag_rouge1",
                "rag_rougeL",
                "rag_bleu",
                "rag_meteor",
                "rag_chrf",
                "rag_fuzzy",
                "rag_len_ratio",
                "rag_bertscore_f1",
                "rag_semantic_similarity",
            ],
        )
        print_metrics(
            "S2S ",
            metrics_row,
            [
                "accuracy",
                "rouge1",
                "rougeL",
                "bleu",
                "masked_accuracy",
                "meteor",
                "chrf",
                "fuzzy",
                "len_ratio",
                "bertscore_f1",
                "semantic_similarity",
            ],
        )
        tlog(f"Epoch {metrics_row['epoch']} finished!")
