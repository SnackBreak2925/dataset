import os
import json
import logging
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

from helpers.cleaner import postprocess_answer


class RagPipeline:
    def __init__(
        self,
        output_dir=None,
        kb_dir="kb_by_category",
        retriever_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        device=None,
        model=None,
        tokenizer=None,
        sim_model=None,
    ):
        self.output_dir = output_dir
        self.kb_dir = kb_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever = sim_model or SentenceTransformer(
            retriever_model_name, device="cpu"
        )
        self.category_kbs, self.category_names = self._load_kb_texts()
        self.category_embeddings = self._encode_kbs(self.category_kbs)
        self.category_name_embeddings = self.retriever.encode(
            self.category_names, convert_to_tensor=True, show_progress_bar=False
        )
        self._external_model = model
        self._external_tokenizer = tokenizer
        self._internal_model_path = None
        self.model = None
        self.tokenizer = None
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif output_dir is not None:
            self._internal_model_path = self.get_latest_checkpoint(output_dir)
            if self._internal_model_path:
                self._load_model_and_tokenizer()

    @staticmethod
    def get_latest_checkpoint(output_dir):
        if output_dir is None or not os.path.isdir(output_dir):
            return None
        checkpoints = [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(output_dir, d))
        ]
        if not checkpoints:
            return None
        checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        latest = checkpoints_sorted[-1]
        return os.path.join(output_dir, latest)

    def _load_kb_texts(self):
        """Только читает тексты KB, без encode!"""
        category_kbs, category_names = {}, []
        for fn in os.listdir(self.kb_dir):
            if fn.endswith(".json"):
                cat = fn[:-5]
                with open(os.path.join(self.kb_dir, fn), encoding="utf-8") as f:
                    entries = json.load(f)
                unique_entries = sorted(set(e.strip() for e in entries if e.strip()))
                category_kbs[cat] = unique_entries
                category_names.append(cat)
        return category_kbs, category_names

    def _encode_kbs(self, category_kbs):
        """Считает эмбеддинги для всех KB по категориям, с кэшем."""
        category_embeddings = {}
        for cat, texts in category_kbs.items():
            cache_file = os.path.join(self.kb_dir, f"{cat}_emb.pt")
            if os.path.exists(cache_file):
                emb = torch.load(cache_file)
            else:
                emb = self.retriever.encode(
                    texts, convert_to_tensor=True, show_progress_bar=False
                )
                torch.save(emb, cache_file)
            category_embeddings[cat] = emb
        return category_embeddings

    def _load_model_and_tokenizer(self):
        if self._internal_model_path:
            self.model = (
                T5ForConditionalGeneration.from_pretrained(self._internal_model_path)
                .eval()
                .to(self.device)
            )
            self.tokenizer = T5Tokenizer.from_pretrained(self._internal_model_path)

    def refresh_checkpoint(self):
        if self._external_model is not None and self._external_tokenizer is not None:
            return
        if self.output_dir:
            latest = self.get_latest_checkpoint(self.output_dir)
            if latest and latest != self._internal_model_path:
                self._internal_model_path = latest
                self._load_model_and_tokenizer()

    def safe_encode(self, text):
        try:
            return self.retriever.encode(
                text, convert_to_tensor=True, show_progress_bar=False
            )
        except Exception as e:
            logging.exception("Ошибка encode: %s", e)
            return None

    def auto_detect_category(self, question, top_n=3):
        q_emb = self.safe_encode(question)
        if q_emb is None:
            return None
        scores = util.cos_sim(q_emb, self.category_name_embeddings)[0]
        top_indices = torch.topk(scores, k=top_n).indices.tolist()
        best_score = scores[top_indices[0]].item()
        if best_score < 0.2:
            return None
        return self.category_names[top_indices[0]]

    def is_uninformative(self, answer, answer_min_len=15, patterns_to_skip=None):
        patterns = patterns_to_skip or [r"^\[SIGNATURE\]$", r"\[URL\]"]
        if len(answer) < answer_min_len:
            return True
        for pattern in patterns:
            if re.search(pattern, answer, flags=re.IGNORECASE):
                return True
        return False

    def batched_generate_answers(self, prompts, batch_size=16, max_length=128):
        all_answers = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            input_ids = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).input_ids.to(self.device)
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        input_ids,
                        max_length=max_length,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1,
                    )
                except Exception:
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=128,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1,
                    )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # processed_answers = [postprocess_answer(ans) for ans in decoded]
            # all_answers.extend(processed_answers)
            all_answers.extend(decoded)
        return all_answers

    def _generate_answer(self, prompt):
        return self.batched_generate_answers([prompt], batch_size=1)[0]

    def generate_with_contexts(
        self,
        category,
        question,
        top_k=5,
        answer_min_len=15,
        soft_threshold=0.1,
        batch_gen_size=16,
    ):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("RagPipeline: model/tokenizer не заданы!")
        kb = self.category_kbs[category]
        emb = self.category_embeddings[category]
        q_emb = self.safe_encode(question)
        if q_emb is None:
            return ["[Ошибка кодирования]"], []
        try:
            hits = util.semantic_search(q_emb, emb, top_k=top_k)[0]
        except Exception as e:
            logging.exception("semantic_search error: %s", e)
            return ["[Ошибка поиска контекста]"], []
        if not hits:
            return ["[Нет релевантных контекстов]"], []
        max_score = hits[0]["score"]
        adaptive_threshold = max(soft_threshold, max_score * 0.15)
        prompt_tuples = []
        for hit in hits:
            if hit["score"] < adaptive_threshold:
                continue
            idx = hit["corpus_id"]
            ctx = kb[idx]
            prompt = f"Категория: {category}\nКонтекст: {ctx}\nВопрос: {question}\nОператор: "
            prompt_tuples.append((prompt, ctx, hit["score"]))
        if not prompt_tuples:
            return ["[Нет релевантных контекстов]"], []
        prompts = [pt[0] for pt in prompt_tuples]
        answers = self.batched_generate_answers(prompts, batch_size=batch_gen_size)
        seen_answers = set()
        responses, results = [], []
        for ans, (prompt, ctx, score) in zip(answers, prompt_tuples):
            if ans in seen_answers or self.is_uninformative(ans, answer_min_len):
                continue
            responses.append(ans)
            results.append((ctx, ans, score))
            seen_answers.add(ans)
        return responses, results

    def batch_retrieve_contexts(self, category, questions, top_k=5):
        """Батчевый поиск топ-контекстов для группы вопросов."""
        emb = self.category_embeddings[category]
        q_embs = self.retriever.encode(
            questions, batch_size=32, convert_to_tensor=True, show_progress_bar=False
        )
        all_hits = []
        for i in range(q_embs.shape[0]):
            hits = util.semantic_search(q_embs[i], emb, top_k=top_k)[0]
            all_hits.append(hits)
        return all_hits
