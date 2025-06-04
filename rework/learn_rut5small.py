from finetuners.t5 import T5FineTuner

if __name__ == "__main__":
    finetuner = T5FineTuner(
        model_name="cointegrated/rut5-small",
        dataset_path="dialogue_dataset.json",
        output_dir="./rut5small-finetuned",
        freeze_encoder_layers=2,
        freeze_decoder_layers=6,
        trainer_args={
            "num_train_epochs": 20,
            "learning_rate": 5e-5,
        },
    )
    finetuner.run()
