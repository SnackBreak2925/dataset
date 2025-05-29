from t5_fine_tuner import T5FineTuner

if __name__ == "__main__":
    finetuner = T5FineTuner(
        model_name="cointegrated/rut5-base",
        dataset_path="dialogue_dataset.json",
        output_dir="./rut5base-finetuned",
        trainer_args={
            "num_train_epochs": 20,
            "learning_rate": 1e-4,
        },
    )
    finetuner.run()
