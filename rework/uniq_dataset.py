import json

if __name__ == "__main__":
    with open("dialogue_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    seen = set()
    unique_test_set = []

    for entry in dataset:
        label = entry["label"]
        if label not in seen:
            unique_test_set.append(entry)
            seen.add(label)

    with open("dialogue_dataset_unique_labels.json", "w", encoding="utf-8") as f:
        json.dump(unique_test_set, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Сохранено {len(unique_test_set):,} уникальных ответов в dialogue_dataset_unique_labels.json"
    )
