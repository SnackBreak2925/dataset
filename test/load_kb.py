# 📚 Загрузка базы знаний
import json
def load_kb(path="dataset.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
