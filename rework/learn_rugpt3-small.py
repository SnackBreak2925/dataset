from finetuners.casual_lm import CausalLMFineTuner

if __name__ == "__main__":
    finetuner = CausalLMFineTuner(
        model_name="sberbank-ai/rugpt3small_based_on_gpt2",
        dataset_path="dialogue_dataset.json",
        output_dir="./rugpt3small-finetuned",
        unfreeze_encoder_layers=4,
        unfreeze_decoder_layers=4,
        trainer_args={
            "num_train_epochs": 30,
            "learning_rate": 4e-5,
        },
    )
    finetuner.run()
