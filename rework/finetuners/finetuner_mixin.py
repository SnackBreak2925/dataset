import time
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
from callbacks.log import LogCallback
from rework.callbacks.accuracy import AccuracyCallback

from helpers.metrics_plotter import MetricsPlotter


class FinetunerMixin:
    model_cls = None
    tokenizer_cls = None

    def __init__(self, *args, **kwargs):
        self.init_timestamp = int(time.time())
        super().__init__(*args, **kwargs)

    def model_short_name(self):
        return self.model_name.split("/")[-1]

    def load_and_split_dataset(self):
        with open(self.dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        return train_test_split(data, test_size=0.1, random_state=24, shuffle=True)

    def prepare_train_dataset(self, train_data):
        ds = Dataset.from_list(train_data)
        return ds.map(self.preprocess, remove_columns=ds.column_names)

    def prepare_eval_dataset(self, test_data):
        ds = Dataset.from_list(test_data)
        return ds.map(self.preprocess, remove_columns=ds.column_names)

    def load_model_and_tokenizer(self):
        self.model = self.model_cls.from_pretrained(self.model_name)
        self.tokenizer = self.tokenizer_cls.from_pretrained(
            self.model_name, **self.tokenizer_kwargs
        )

    def freeze_layers(self):
        self.freeze_encoder(self.unfreeze_encoder_layers)
        self.freeze_decoder(self.unfreeze_decoder_layers)

    def build_trainer_args(self):
        raise NotImplementedError(
            "build_trainer_args должен быть реализован в наследнике!"
        )

    def get_default_training_args(self):
        return dict(
            output_dir=self.output_dir,
            num_train_epochs=10,
            learning_rate=5e-5,
            weight_decay=0.01,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            logging_strategy="epoch",
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            do_eval=True,
            logging_dir="./runs",
            fp16=False,
            bf16=False,
            report_to="none",
        )

    def get_callbacks(self, test_data):
        return [
            LogCallback(self.tokenizer, test_data),
            AccuracyCallback(
                self.tokenizer,
                self.init_timestamp,
                self.model_short_name(),
                raw_test_data=test_data,
            ),
        ]

    def evaluate_baseline(self, trainer):
        print("==> Evaluate BEFORE training (zero epoch baseline)...")
        trainer.evaluate()

    def train(self, trainer):
        trainer.train()

    def save(self, trainer):
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("\n✅ Обучение завершено и модель сохранена")

    def plot_metrics(self):
        plotter = MetricsPlotter(logs_dir="logs", base_out_dir="train_results")
        plotter.plot_metrics(
            model_name=self.model_short_name(), init_timestamp=self.init_timestamp
        )
