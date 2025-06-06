from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from callbacks.log import LogCallback

from finetuners.finetuner_mixin import FinetunerMixin


class CausalLMFineTuner(FinetunerMixin):
    model_cls = AutoModelForCausalLM
    tokenizer_cls = AutoTokenizer

    def __init__(
        self,
        model_name,
        dataset_path="./dialogue_dataset.json",
        output_dir=None,
        tokenizer_kwargs=None,
        max_length=256,
        unfreeze_encoder_layers=2,
        unfreeze_decoder_layers=2,
        trainer_args=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir or f"{self.model_short_name()}-finetuned"
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.max_length = max_length
        self.unfreeze_encoder_layers = unfreeze_encoder_layers
        self.unfreeze_decoder_layers = unfreeze_decoder_layers
        self.trainer_args = trainer_args or {}
        self.kwargs = kwargs

        super().__init__()

    def preprocess(self, example):
        prompt = example.get("text", "")
        answer = example.get("label", "")
        full_text = prompt + answer
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        label_ids = tokenized["input_ids"].clone()
        prompt_len = (prompt_tokens != self.tokenizer.pad_token_id).sum().item()
        label_ids[:prompt_len] = -100
        label_ids[tokenized["attention_mask"] == 0] = -100
        tokenized["labels"] = label_ids
        return tokenized

    def freeze_encoder(self, unfreeze_last_n=0):
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            total_layers = len(self.model.transformer.h)
            for i, block in enumerate(self.model.transformer.h):
                requires_grad = i >= total_layers - unfreeze_last_n
                for param in block.parameters():
                    param.requires_grad = requires_grad

    def freeze_decoder(self, unfreeze_last_n=0):
        self.freeze_encoder(unfreeze_last_n)

    def build_trainer_args(self):
        default_args = self.get_default_training_args()
        default_args.update(self.trainer_args)
        return TrainingArguments(**default_args)

    # def get_callbacks(self, test_data):
    #     return [
    #         LogCallback(self.tokenizer, test_data),
    #     ]

    def run(self):
        train_data, test_data = self.load_and_split_dataset()
        self.load_model_and_tokenizer()
        self.freeze_layers()
        tokenized_train = self.prepare_train_dataset(train_data)
        tokenized_test = self.prepare_eval_dataset(test_data)
        training_args = self.build_trainer_args()
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        callbacks = self.get_callbacks(test_data)

        trainer = Trainer(
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
