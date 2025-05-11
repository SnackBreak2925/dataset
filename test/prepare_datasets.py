# 🧹 Объединение датасетов
from extract_data import fetch_data
import json

def build_dataset():
    data = fetch_data()
    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    build_dataset()
