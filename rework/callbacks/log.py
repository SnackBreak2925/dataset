from transformers import TrainerCallback, AutoConfig
from tqdm import tqdm
from helpers.rag_pipeline import RagPipeline
import nltk
from helpers.cleaner import postprocess_answer

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
            config = AutoConfig.from_pretrained(model.config.name_or_path)
            is_t5 = config.model_type == "t5"

            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            if is_t5:
                outputs = model.generate(
                    input_ids,
                    max_length=64,
                    num_beams=5,
                    num_return_sequences=5,
                )
            else:
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    num_beams=5,
                    num_return_sequences=5,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            tqdm.write("=" * 100)

            tqdm.write(f"[Входной текст]: {ex['text']}")
            tqdm.write(f"[Промт]: {prompt}")
            tqdm.write(f"[Эталонный ответ]: {ex.get('label', 'нет')}")
            for idx, output in enumerate(outputs, 1):
                answer = self.tokenizer.decode(output, skip_special_tokens=True)
                tqdm.write(f"[Beam {idx}]: {postprocess_answer(answer)}")

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
