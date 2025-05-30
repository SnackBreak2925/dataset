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
    "len_ratio",
    "bertscore_f1",
    "semantic_similarity",
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
        x_axis="epoch",
    ):
        # Определение пути к csv, если не задан
        if not csv_path and model_name and init_timestamp:
            csv_path = self.find_csv(model_name, init_timestamp)
        if not csv_path:
            raise ValueError(
                "Either csv_path or (model_name and init_timestamp) must be provided"
            )
        df = pd.read_csv(csv_path)
        # Какие есть метрики
        all_metrics = set(df.columns) - SERVICE_COLUMNS
        if metrics is None:
            metrics = list(all_metrics)
        # Для единообразия названия и папка
        model_name = model_name or "model"
        init_timestamp = str(init_timestamp or "ts")
        out_dir = os.path.join(self.base_out_dir, f"{model_name}-{init_timestamp}")
        os.makedirs(out_dir, exist_ok=True)
        # Масштабы по X и Y (от 0 до max по каждому)
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
            y = list(df[metric])
            plt.plot(x, y, marker="o", label=metric)
            plt.title(metric)
            plt.xlabel(x_axis.capitalize())
            plt.ylabel(metric)
            plt.grid(True)
            plt.legend()
            plt.xlim(left=0, right=max(x))
            if metric in PERCENT_METRICS:
                plt.ylim(bottom=0, top=1.0)
            else:
                y_max = max([v for v in y if pd.notnull(v)])
                plt.ylim(bottom=0, top=self.nice_ceil(y_max))
            out_file = os.path.join(
                out_dir, f"{metric}-{model_name}-{init_timestamp}.png"
            )
            plt.savefig(out_file)
            plt.close()
            print(f"✅ Сохранён график: {out_file}")


# CLI
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
