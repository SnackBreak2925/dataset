from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from amazing_printer import ap


def postprocess_answer(answer):
    if "Оператор:" in answer:
        answer = answer.split("Оператор:")[-1]
    if "Пользователь:" in answer:
        answer = answer.split("Пользователь:")[0]
    answer = answer.strip().split("\n\n")[0].strip()
    return answer.strip()


model_path = "./rugpt3small-finetuned"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("💬 Введите сообщение пользователя (или 'exit'):")

while True:
    prompt = "Категория: Электронный документооборот\nТема: тест\nПользователь: kldghlajgnhflksbsm\nОператор: "
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,  # Вот это главное!
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    ap(f"promt {prompt}")
    ap("=" * 100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ap(f"🤖 Модель: {postprocess_answer(response)}")
    break
