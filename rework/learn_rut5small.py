from t5_fine_tuner import T5FineTuner

if __name__ == "__main__":
    finetuner = T5FineTuner(
        model_name="cointegrated/rut5-small",
        dataset_path="dialogue_dataset.json",
        output_dir="./rut5small-finetuned",
        trainer_args={
            "num_train_epochs": 30,
            "learning_rate": 5e-5,
        },
    )
    finetuner.run()
