import os
import json
import logging
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(
    filename="rag_debug.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_latest_checkpoint(output_dir):
    checkpoints = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    latest = checkpoints_sorted[-1]
    return os.path.join(output_dir, latest)


class RagPipeline:
    def __init__(
        self,
        output_dir,
        kb_dir="rag_inputs_by_request_xlmroberta",
        retriever_model_name="xlm-roberta-base",
        device=None,
        model_subdir=None,
    ):
        self.output_dir = output_dir
        self.kb_dir = kb_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever = SentenceTransformer(retriever_model_name)
        self.category_kbs, self.category_embeddings, self.category_names = (
            self._load_kbs()
        )
        self.category_name_embeddings = self.retriever.encode(
            self.category_names, convert_to_tensor=True, show_progress_bar=False
        )
        self.model_subdir = model_subdir or get_latest_checkpoint(self.output_dir)
        self._load_model_and_tokenizer()

    def _load_kbs(self):
        category_kbs, category_embeddings = {}, {}
        category_names = []
        for fn in os.listdir(self.kb_dir):
            if fn.endswith(".json"):
                cat = fn[:-5]
                with open(os.path.join(self.kb_dir, fn), encoding="utf-8") as f:
                    entries = json.load(f)
                unique_entries = sorted(set(e.strip() for e in entries if e.strip()))
                category_kbs[cat] = unique_entries
                category_embeddings[cat] = self.retriever.encode(
                    unique_entries, convert_to_tensor=True, show_progress_bar=False
                )
                category_names.append(cat)
        return category_kbs, category_embeddings, category_names

    def _load_model_and_tokenizer(self):
        if not self.model_subdir:
            raise RuntimeError("Не найден ни один чекпоинт для загрузки модели!")
        logger.info(f"Загружаем модель и токенизатор из {self.model_subdir}")
        self.model = (
            T5ForConditionalGeneration.from_pretrained(self.model_subdir)
            .eval()
            .to(self.device)
        )
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_subdir)

    def refresh_checkpoint(self):
        latest = get_latest_checkpoint(self.output_dir)
        if latest and latest != self.model_subdir:
            self.model_subdir = latest
            self._load_model_and_tokenizer()

    def safe_encode(self, text):
        try:
            return self.retriever.encode(
                text, convert_to_tensor=True, show_progress_bar=False
            )
        except Exception as e:
            logger.exception("Ошибка encode: %s", e)
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
        print(top_indices)
        print(scores)
        print(best_score)
        for i in top_indices:
            print(self.category_names[i])
        return self.category_names[top_indices[0]]

    def is_uninformative(self, answer, answer_min_len=15, patterns_to_skip=None):
        patterns = patterns_to_skip or [r"^\[SIGNATURE\]$", r"\[URL\]"]
        if len(answer) < answer_min_len:
            return True
        for pattern in patterns:
            if re.search(pattern, answer, flags=re.IGNORECASE):
                return True
        return False

    def generate_with_contexts(
        self, category, question, top_k=5, answer_min_len=15, soft_threshold=0.1
    ):
        kb = self.category_kbs[category]
        emb = self.category_embeddings[category]
        q_emb = self.safe_encode(question)
        if q_emb is None:
            return ["[Ошибка кодирования]"], []

        try:
            hits = util.semantic_search(q_emb, emb, top_k=top_k)[0]
        except Exception as e:
            logger.exception("semantic_search error: %s", e)
            return ["[Ошибка поиска контекста]"], []

        if not hits:
            return ["[Нет релевантных контекстов]"], []

        max_score = hits[0]["score"]
        adaptive_threshold = max(soft_threshold, max_score * 0.15)

        responses, results = [], []
        seen_answers = set()

        for hit in hits:
            if hit["score"] < adaptive_threshold:
                continue
            idx = hit["corpus_id"]
            ctx = kb[idx]
            prompt = f"Категория: {category}\nКонтекст: {ctx}\nВопрос: {question}\nОператор: "
            logger.info("Prompt used:\n%s\n", prompt)
            try:
                ids = self.tokenizer.encode(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                ).to(self.device)
                with torch.no_grad():
                    out = self.model.generate(
                        ids,
                        max_length=64,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1,
                    )
                ans = self.tokenizer.decode(out[0], skip_special_tokens=True)
            except Exception as e:
                logger.exception("Генерация ответа: %s", e)
                ans = "[Ошибка генерации]"

            if ans in seen_answers or self.is_uninformative(ans, answer_min_len):
                continue

            responses.append(ans)
            results.append((ctx, ans, hit["score"]))
            seen_answers.add(ans)

        return responses, results
