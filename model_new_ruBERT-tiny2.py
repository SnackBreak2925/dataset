import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AdamW,
    get_cosine_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from collections import defaultdict

# Конфигурация
os.makedirs('new_ruBERT-tiny2/train_results', exist_ok=True)
os.makedirs('new_ruBERT-tiny2/best_model', exist_ok=True)

# Усиленные параметры регуляризации
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 40  # Увеличено для полного обучения
LEARNING_RATE = 3e-5
WARMUP_STEPS = 400  # Увеличенный warmup
DROPOUT_RATE = 0.2  # Усиленный dropout
WEIGHT_DECAY = 0.02  # Усиленная L2 регуляризация
PATIENCE = 4  # Увеличенный patience для ранней остановки

# Загрузка и аугментация данных
data = pd.read_csv("Data/data_slang_final_shuffled.csv", encoding='utf-8')

# 1. Аугментация данных через синонимы
slang_synonyms = {
    "ТГ": ["Телеграм", "Телеграмм", "TG"],
    "гостить": ["пропадать", "забивать", "исчезать"]
}

def augment_data(df, synonyms, num_augments=2):
    augmented_rows = []
    for _, row in df.iterrows():
        slang = row['slang_word']
        if slang in synonyms:
            for synonym in np.random.choice(synonyms[slang], size=min(num_augments, len(synonyms[slang])), replace=False):
                new_sentence = row['sentence'].replace(slang, synonym)
                augmented_rows.append({
                    'sentence': new_sentence,
                    'slang_word': synonym,
                    'correct_word': row['correct_word'],
                    'corrected_sentence': new_sentence.replace(synonym, row['correct_word'])
                })
    return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

augmented_data = augment_data(data, slang_synonyms)

# 2. Балансировка классов
slang_counts = augmented_data['slang_word'].value_counts()
max_count = slang_counts.max()
balanced_data = pd.DataFrame()

for slang in slang_counts.index:
    slang_data = augmented_data[augmented_data['slang_word'] == slang]
    if len(slang_data) < max_count:
        oversampled = slang_data.sample(max_count - len(slang_data), replace=True, random_state=42)
        slang_data = pd.concat([slang_data, oversampled])
    balanced_data = pd.concat([balanced_data, slang_data])

train_data, val_data = train_test_split(balanced_data, test_size=0.1, random_state=42)

# Датасет с улучшенной обработкой
class AugmentedSlangDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sentence = item['sentence']
        corrected = item['corrected_sentence']

        # Токенизация
        inputs = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            corrected,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten(),
            'target_mask': (targets['input_ids'] != self.tokenizer.pad_token_id).float().flatten()
        }

# Инициализация модели с усиленной регуляризацией
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModelForMaskedLM.from_pretrained("cointegrated/rubert-tiny2")

# Усиленный dropout
model.config.hidden_dropout_prob = DROPOUT_RATE
model.config.attention_probs_dropout_prob = DROPOUT_RATE

# Частичная заморозка слоев (улучшение 2)
for param in model.bert.encoder.layer[:6].parameters():  # Замораживаем первые 6 из 12 слоев
    param.requires_grad = False

# Подготовка данных
train_dataset = AugmentedSlangDataset(train_data, tokenizer, MAX_LEN)
val_dataset = AugmentedSlangDataset(val_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Оптимизатор с усиленной регуляризацией
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),  # Только незамороженные параметры
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    correct_bias=False
)
total_steps = len(train_loader) * EPOCHS

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
    num_cycles=0.5
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Метрики с улучшенной обработкой PAD-токенов
def masked_accuracy(preds, labels, mask):
    preds = torch.argmax(preds, dim=2)
    correct = (preds == labels) * mask
    return correct.sum().item() / mask.sum().item()

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss, total_acc = 0, 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        inputs = {k: v.to(device) for k, v in batch.items() if k != 'target_mask'}
        outputs = model(**inputs)

        loss = outputs.loss
        acc = masked_accuracy(outputs.logits, inputs['labels'], batch['target_mask'].to(device))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)

def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'target_mask'}
            outputs = model(**inputs)

            loss = outputs.loss
            acc = masked_accuracy(outputs.logits, inputs['labels'], batch['target_mask'].to(device))

            total_loss += loss.item()
            total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)

# Обучение с улучшенным логированием
best_val_loss = float('inf')
patience_counter = 0
history = defaultdict(list)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    start_time = time.time()

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)

    epoch_time = time.time() - start_time
    current_lr = scheduler.get_last_lr()[0]

    # Сохранение истории
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    history['epoch_time'].append(epoch_time)

    # Сохранение лучшей модели
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained('new_ruBERT-tiny2/best_model')
        tokenizer.save_pretrained('new_ruBERT-tiny2/best_model')
        patience_counter = 0
    else:
        patience_counter += 1

    # Логирование
    log_msg = (
        f"Epoch {epoch + 1}/{EPOCHS}\n"
        f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}\n"
        f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}\n"
        f"LR: {current_lr:.2e} | Time: {epoch_time:.2f}s\n"
    )
    print(log_msg)

    with open('new_ruBERT-tiny2/train_results/training_log.txt', 'a') as f:
        f.write(log_msg + "\n")

    # Ранняя остановка
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping after {PATIENCE} epochs without improvement")
        break

# Сохранение финальной модели и визуализация
model.save_pretrained('new_ruBERT-tiny2/final_model')
tokenizer.save_pretrained('new_ruBERT-tiny2/final_model')

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history['lr'], label='Learning Rate')
plt.title('LR Schedule')
plt.legend()

plt.tight_layout()
plt.savefig('new_ruBERT-tiny2/train_results/metrics.png')
plt.show()

print("\nTraining complete!")
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Best Val Accuracy: {max(history['val_acc']):.4f}")
