"""
evaluate_compare.py
Сравнение метрик модели RuT5-base на исходных vs. инструкционных промптах
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_metrics(samples, tokenizer, model, max_in=256, max_out=64):
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1
    r1, rL, bleu = [], [], []
    for s in tqdm(samples, desc="Eval"):
        ids = tokenizer.encode(
            s["text"], return_tensors="pt", truncation=True, max_length=max_in
        )
        gen = model.generate(ids, max_length=max_out)
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        ref = s["label"]
        rs = rouge.score(ref, pred)
        r1.append(rs["rouge1"].fmeasure)
        rL.append(rs["rougeL"].fmeasure)
        bleu.append(
            sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
        )
    return np.mean(r1), np.mean(rL), np.mean(bleu)


if __name__ == "__main__":
    # 1️⃣ модель
    model_path = Path("./rut5base-finetuned")
    tok = T5Tokenizer.from_pretrained(model_path)
    mdl = T5ForConditionalGeneration.from_pretrained(model_path)
    mdl.eval()

    # 2️⃣ данные
    with open("dialogue_dataset.json", encoding="utf-8") as f:
        data = json.load(f)
    test = data[-max(1, int(len(data) * 0.1)) :]

    instr_prefix = "ЗАДАЧА: Ответить кратко и по делу.\nЗапрос: "
    test_instr = [{"text": instr_prefix + s["text"], "label": s["label"]} for s in test]

    # 3️⃣ метрики
    base_r1, base_rL, base_b = compute_metrics(test, tok, mdl)
    instr_r1, instr_rL, instr_b = compute_metrics(test_instr, tok, mdl)

    # 4️⃣ таблица + график
    df = pd.DataFrame(
        {
            "Prompt style": ["Original", "Instruction"],
            "ROUGE‑1": [base_r1, instr_r1],
            "ROUGE‑L": [base_rL, instr_rL],
            "BLEU": [base_b, instr_b],
        }
    )
    print("\n=== Сводка ===")
    print(df.to_markdown(index=False, floatfmt=".4f"))

    # график
    x = np.arange(len(df))
    w = 0.25
    plt.figure(figsize=(8, 4))
    plt.bar(x - w, df["ROUGE‑1"], w, label="ROUGE‑1")
    plt.bar(x, df["ROUGE‑L"], w, label="ROUGE‑L")
    plt.bar(x + w, df["BLEU"], w, label="BLEU")
    plt.xticks(x, df["Prompt style"])
    plt.ylabel("Score")
    plt.title("Средние метрики: обычный vs. инструкционный промпт")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    print("📊  График сохранён как metrics_comparison.png")

"""
Добрый день. В 302 ауд. главного корпуса были получены 12  новых компьютеров. Нужны 12 лицензий:1) Windows 10 Pro2) Visual Studio 3) MathCad4) MS Office

Прошу разрешить мне доступ к системе электронного документооборота не из сети ТУСУР, поскольку я сейчас нахожусь в командировке А необходимо отслеживать заявку на закупку оборудования

Добрый день, уважаемые коллеги, прошу подключить мне удаленный доступ к корпоративной сети. Заранее благодарю

Добрый день! Возникли проблемы с отправкой результатов работы по предмету &quot;Программирование&quot;: пишет что я не подписан на такой курс. Также на почту 2 раза пришло письмо &quot;Добро пожаловать на ФДО ТУСУР&quot;, хотя я обучаюсь с ноября 2021 г.

Здравствуйте! Я новый преподаватель в ТУСУР. Пахомова Елизавета Владимировна. Сделайте, мне, пожалуйста, электронную почту от ТУСУРа.
"""
