from finetuners.t5 import T5FineTuner

if __name__ == "__main__":
    finetuner = T5FineTuner(
        model_name="cointegrated/rut5-base",
        dataset_path="dialogue_dataset.json",
        output_dir="./rut5base-finetuned",
        unfreeze_encoder_layers=2,
        unfreeze_decoder_layers=4,
        trainer_args={
            "num_train_epochs": 100,
            "learning_rate": 4e-5,
        },
    )
    finetuner.run()
