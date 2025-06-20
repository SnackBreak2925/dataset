import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import math

SERVICE_COLUMNS = {
    "step",
    "epoch",
    "global_step",
    "runtime",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
}

PERCENT_METRICS = {
    "accuracy",
    "rouge1",
    "rougeL",
    "bleu",
    "masked_accuracy",
    "meteor",
    "chrf",
    "fuzzy",
    "bertscore_f1",
    "semantic_similarity",
    "rag_rouge1",
    "rag_rougeL",
    "rag_bleu",
    "rag_masked_accuracy",
    "rag_meteor",
    "rag_chrf",
    "rag_fuzzy",
    "rag_len_ratio",
    "rag_bertscore_f1",
    "rag_semantic_similarity",
}


TRANSLATE = {
    "accuracy": "Классическая точность",
    "masked_accuracy": "Маскированная точность",
    "rouge1": "ROUGE-1",
    "rougeL": "ROUGE-L",
    "bleu": "BLEU",
    "meteor": "METEOR",
    "chrf": "Character F-score (chrF)",
    "fuzzy": "Levenshtein ratio (Fuzzy)",
    "len_ratio": "Cоотношение длин",
    "bertscore_f1": "BERTScore F1",
    "semantic_similarity": "Косинусное сходство предложений",
    "rag_rouge1": "RAG ROUGE-1",
    "rag_rougeL": "RAG ROUGE-L",
    "rag_bleu": "RAG BLEU",
    "rag_meteor": "RAG METEOR",
    "rag_chrf": "RAG Character F-score (chrF)",
    "rag_fuzzy": "RAG Levenshtein ratio (Fuzzy)",
    "rag_len_ratio": "RAG Cоотношение длин",
    "rag_bertscore_f1": "RAG BERTScore F1",
    "rag_semantic_similarity": "RAG Косинусное сходство предложений",
}


class MetricsPlotter:
    def __init__(self, logs_dir="logs", base_out_dir="train_results"):
        self.logs_dir = logs_dir
        self.base_out_dir = base_out_dir

    def find_csv(self, model_name, init_timestamp):
        pattern = os.path.join(
            self.logs_dir, f"metrics-log-{model_name}-{init_timestamp}.csv"
        )
        files = glob.glob(pattern)
        if files:
            return files[0]
        pattern = os.path.join(self.logs_dir, f"*{model_name}*{init_timestamp}*.csv")
        files = glob.glob(pattern)
        if files:
            return files[0]
        raise FileNotFoundError(
            f"CSV log not found for model {model_name}, timestamp {init_timestamp}"
        )

    @staticmethod
    def nice_ceil(x):
        if x == 0:
            return 1
        exp = math.floor(math.log10(x))
        f = x / 10**exp
        if f <= 1:
            nice = 1
        elif f <= 2:
            nice = 2
        elif f <= 5:
            nice = 5
        else:
            nice = 10
        return nice * 10**exp

    def plot_metrics(
        self,
        csv_path=None,
        model_name=None,
        init_timestamp=None,
        metrics=None,
        x_axis="Эпоха",
    ):
        if not csv_path and model_name and init_timestamp:
            csv_path = self.find_csv(model_name, init_timestamp)
        if not csv_path:
            raise ValueError(
                "Either csv_path or (model_name and init_timestamp) must be provided"
            )
        df = pd.read_csv(csv_path)
        all_metrics = set(df.columns) - SERVICE_COLUMNS
        if metrics is None:
            metrics = list(all_metrics)
        model_name = model_name or "model"
        init_timestamp = str(init_timestamp or "ts")
        out_dir = os.path.join(self.base_out_dir, f"{model_name}-{init_timestamp}")
        os.makedirs(out_dir, exist_ok=True)
        if x_axis not in df.columns:
            raise ValueError(
                f"x_axis '{x_axis}' not found in CSV columns. Available: {list(df.columns)}"
            )
        for metric in metrics:
            if metric not in df.columns:
                print(f"⚠️ Нет метрики '{metric}' в логах")
                continue
            plt.figure(figsize=(8, 5))
            x = list(df[x_axis])
            x = list(map(lambda val: int(val), x))
            y = list(df[metric])
            if metric in PERCENT_METRICS:
                y = list(map(lambda x: x * 100, y))
            metric_title = TRANSLATE.get(metric)
            plt.plot(x, y, label=metric_title)
            plt.title(metric_title)
            plt.xlabel("Эпоха")
            plt.ylabel(metric_title)
            plt.grid(True)
            plt.xlim(left=0, right=max(x))
            plt.xticks(range(0, max(x) + 1, max(x) // 20))
            if metric in PERCENT_METRICS:
                plt.ylim(bottom=0, top=100)
                plt.yticks(range(0, 101, 10))
            else:
                y_max = max([v for v in y if pd.notnull(v)])
                plt.ylim(bottom=0, top=self.nice_ceil(y_max))
            out_file = os.path.join(
                out_dir, f"{metric}-{model_name}-{init_timestamp}.png"
            )
            plt.savefig(out_file)
            plt.close()
            print(f"✅ Сохранён график: {out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Построить графики метрик из CSV-логов обучения"
    )
    parser.add_argument("--csv", help="Путь к CSV-файлу логов")
    parser.add_argument("--model_name", help="Имя модели (если нет --csv)")
    parser.add_argument("--init_timestamp", help="timestamp запуска (если нет --csv)")
    parser.add_argument(
        "--out", default="train_results", help="Базовая папка для картинок"
    )
    parser.add_argument("--logs_dir", default="logs", help="Где искать csv")
    parser.add_argument("--x", default="epoch", help="ось X (epoch/step)")
    parser.add_argument("--metrics", nargs="*", default=None, help="Cписок метрик")
    args = parser.parse_args()
    plotter = MetricsPlotter(logs_dir=args.logs_dir, base_out_dir=args.out)
    plotter.plot_metrics(
        csv_path=args.csv,
        model_name=args.model_name,
        init_timestamp=args.init_timestamp,
        metrics=args.metrics,
        x_axis=args.x,
    )
