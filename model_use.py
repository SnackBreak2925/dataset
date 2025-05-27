import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Загрузка обученной модели и токенизатора
model_path = 'new_ruBERT-tiny2/best_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()


# Функция для исправления сленга в предложении
def correct_slang(sentence):
    # Токенизация предложения
    inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_ids = logits.argmax(dim=-1)

    corrected_sentence = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

    return corrected_sentence


# Тестирование
def main():
    while True:
        sentence = input("\nВведите предложение (или 'exit' для выхода): ")
        if sentence.lower() == 'exit':
            break
        corrected = correct_slang(sentence)
        print(f"Оригинал: {sentence}")
        print(f"Исправлено: {corrected}")


if __name__ == "__main__":
    main()
