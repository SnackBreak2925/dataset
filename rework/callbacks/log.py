from transformers import TrainerCallback
from tqdm import tqdm
from helpers.rag_pipeline import RagPipeline
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
        self.rag_pipe = RagPipeline(model=kwargs["model"], tokenizer=self.tokenizer)
        self.rag_pipe.refresh_checkpoint()

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
            for idx, output in enumerate(outputs, 1):
                answer = self.tokenizer.decode(output, skip_special_tokens=True)
                tqdm.write(f"[Beam {idx}]: {answer}")

            tqdm.write("*" * 100)

            question = ex["ticket_message"]
            category = self.rag_pipe.auto_detect_category(question)
            if not category:
                rag_beams = ["[Категория не определена]"] * 5
            else:
                rag_beams, _ = self.rag_pipe.generate_with_contexts(
                    category, question, top_k=5
                )
            tqdm.write("[Первоначальное сообщение]:")
            tqdm.write(question)
            for idx, rag in enumerate(rag_beams, 1):
                tqdm.write(f"[RAG Beam {idx}]: {rag}")

            tqdm.write("=" * 100)
        model.train()
