import json
import re
import time
from sklearn.model_selection import train_test_split
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

from metrics_plotter import MetricsPlotter
from callbacks import AccuracyCallback, LogCallback


class T5FineTuner:
    def __init__(
        self,
        model_name,
        dataset_path="./dialogue_dataset.json",
        output_dir=None,
        tokenizer_kwargs=None,
        max_input=256,
        max_output=64,
        freeze_encoder_layers=2,
        freeze_decoder_layers=2,
        trainer_args=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir or f"{model_name}-finetuned"
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.max_input = max_input
        self.max_output = max_output
        self.freeze_encoder_layers = freeze_encoder_layers
        self.freeze_decoder_layers = freeze_decoder_layers
        self.trainer_args = trainer_args or {}
        self.kwargs = kwargs

        self.init_timestamp = int(time.time())

    def model_short_name(self):
        return self.model_name.split("/")[-1]

    def freeze_encoder(self, model, unfreeze_last_n=2):
        total_layers = len(model.encoder.block)
        for i, block in enumerate(model.encoder.block):
            requires_grad = i >= total_layers - unfreeze_last_n
            for param in block.parameters():
                param.requires_grad = requires_grad

    def freeze_decoder(self, model, unfreeze_last_n=2):
        total_layers = len(model.decoder.block)
        for i, block in enumerate(model.decoder.block):
            requires_grad = i >= total_layers - unfreeze_last_n
            for param in block.parameters():
                param.requires_grad = requires_grad

    def clean_label(self, text):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\n\r]+", " ", text)
        text = text.replace("\xa0", " ")
        return text.strip()

    def preprocess(self, example):
        input_text = example["text"]
        cleaned_label = self.clean_label(example["label"])
        tokenized = self.tokenizer(
            input_text,
            max_length=self.max_input,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_labels = self.tokenizer(
            cleaned_label,
            max_length=self.max_output,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        tokenized["labels"] = tokenized_labels["input_ids"].squeeze(0)
        return tokenized

    def run(self):
        with open(self.dataset_path, encoding="utf-8") as f:
            full_data = json.load(f)

        train_data, test_data = train_test_split(
            full_data, test_size=0.1, random_state=24, shuffle=True
        )

        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)

        # Загрузка модели и токенизатора
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_name, **self.tokenizer_kwargs
        )

        self.freeze_encoder(self.model, self.freeze_encoder_layers)
        self.freeze_decoder(self.model, self.freeze_decoder_layers)

        tokenized_train = train_dataset.map(
            self.preprocess, remove_columns=train_dataset.column_names
        )
        tokenized_test = test_dataset.map(
            self.preprocess, remove_columns=test_dataset.column_names
        )

        default_training_args = dict(
            output_dir=self.output_dir,
            num_train_epochs=40,
            learning_rate=3e-5,
            weight_decay=0.02,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            lr_scheduler_type="cosine",
            warmup_steps=200,
            logging_strategy="epoch",
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=3,
            do_eval=True,
            logging_dir="./runs",
            fp16=False,
            bf16=False,
            report_to="none",
        )

        default_training_args.update(self.trainer_args)

        training_args = Seq2SeqTrainingArguments(**default_training_args)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model
        )

        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            data_collator=data_collator,
            callbacks=[
                LogCallback(self.tokenizer, test_data),
                AccuracyCallback(
                    self.tokenizer, self.init_timestamp, self.model_short_name()
                ),
            ],
            **self.kwargs,
        )

        # baseline evaluation before training
        print("==> Evaluate BEFORE training (zero epoch baseline)...")
        trainer.evaluate()

        trainer.train()

        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("\n✅ Обучение завершено и модель сохранена")

        # Построение графиков
        plotter = MetricsPlotter(logs_dir="logs", base_out_dir="train_results")
        plotter.plot_metrics(
            model_name=self.model_short_name(), init_timestamp=self.init_timestamp
        )
