from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import re
from finetuners.finetuner_mixin import FinetunerMixin

class T5FineTuner(FinetunerMixin):
    model_cls = T5ForConditionalGeneration
    tokenizer_cls = T5Tokenizer

    def __init__(
        self,
        model_name,
        dataset_path="./dialogue_dataset.json",
        output_dir=None,
        tokenizer_kwargs=None,
        max_input=256,
        max_output=64,
        unfreeze_encoder_layers=2,
        unfreeze_decoder_layers=2,
        trainer_args=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir or f"{self.model_short_name()}-finetuned"
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.max_input = max_input
        self.max_output = max_output
        self.unfreeze_encoder_layers = unfreeze_encoder_layers
        self.unfreeze_decoder_layers = unfreeze_decoder_layers
        self.trainer_args = trainer_args or {}
        self.kwargs = kwargs

        super().__init__()

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

    def freeze_encoder(self, unfreeze_last_n=2):
        total_layers = len(self.model.encoder.block)
        for i, block in enumerate(self.model.encoder.block):
            requires_grad = i >= total_layers - unfreeze_last_n
            for param in block.parameters():
                param.requires_grad = requires_grad

    def freeze_decoder(self, unfreeze_last_n=2):
        total_layers = len(self.model.decoder.block)
        for i, block in enumerate(self.model.decoder.block):
            requires_grad = i >= total_layers - unfreeze_last_n
            for param in block.parameters():
                param.requires_grad = requires_grad

    def build_trainer_args(self):
        default_args = self.get_default_training_args()
        default_args.update(self.trainer_args)
        return Seq2SeqTrainingArguments(**default_args)

    def clean_label(self, text):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\n\r]+", " ", text)
        text = text.replace("\xa0", " ")
        return text.strip()

    def run(self):
        train_data, test_data = self.load_and_split_dataset()
        self.load_model_and_tokenizer()
        self.freeze_layers()
        tokenized_train = self.prepare_train_dataset(train_data)
        tokenized_test = self.prepare_eval_dataset(test_data)
        training_args = self.build_trainer_args()
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model
        )
        callbacks = self.get_callbacks(test_data)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            data_collator=data_collator,
            callbacks=callbacks,
            **self.kwargs,
        )
        self.evaluate_baseline(trainer)
        self.train(trainer)
        self.save(trainer)
        self.plot_metrics()
