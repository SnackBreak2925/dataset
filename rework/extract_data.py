import os
import json
import dotenv
import re
import pandas as pd
from sqlalchemy import create_engine, text
from collections import defaultdict, Counter
from helpers.cleaner import TextCleaner


def load_db_config():
    dotenv.load_dotenv()
    return {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "dialect": os.getenv("DB_DIALECT"),
    }


def get_engine():
    db_config = load_db_config()
    connection_string = (
        f"{db_config['dialect']}://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    return create_engine(
        connection_string,
        connect_args={"charset": "utf8mb3", "collation": "utf8mb3_unicode_ci"},
        pool_pre_ping=True,
    )


def repiles_query():
    return r"""
SELECT
    messages.id,
    tickets.id AS ticket_id,
    tickets.subject AS subject,
    tickets.message AS ticket_message,
    messages.message AS reply_message,
    category.name AS category_title,
    messages.dt AS created_at,
    messages.staffid
FROM hesk_replies AS messages
INNER JOIN hesk_tickets AS tickets ON tickets.id = messages.replyto
INNER JOIN hesk_categories  AS category ON category.id = tickets.category
LEFT JOIN hesk_users AS staff ON messages.staffid = staff.id
WHERE
    tickets.status = 3
    AND tickets.staffreplies > 0
ORDER BY ticket_id, created_at
"""


def normalize_label(text):
    text = text.strip()
    text = text.replace(". [URL]", " [URL]")
    text = re.sub(r"\s+", " ", text)
    return text


def sentence_normalize(text):
    sentence_end = re.compile(r"([.!?…]+)(\s+|$)")
    parts = []
    last = 0
    for m in sentence_end.finditer(text):
        start, end = m.span()
        sentence = text[last:end].strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            parts.append(sentence)
        last = end
    if last < len(text):
        sentence = text[last:].strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            if not re.search(r"[.!?…]$", sentence):
                sentence += "."
            parts.append(sentence)
    return " ".join(parts)


def remove_signature_from_message(message, staffid, staff_signatures):
    if staffid != 0:
        for sig in staff_signatures.get(staffid, []):
            if sig and sig.strip() and sig in message:
                message = message.replace(sig, "")
    return message


def build_dialogue(messages, upto_idx):
    if upto_idx == 0:
        return ""
    return "\n".join(f"{m['role']}: {m['message']}" for m in messages[:upto_idx])


def prepare_signatures(signatures_df, cleaner, manual_signatures):
    filtered = signatures_df[signatures_df["signature"].astype(bool)]
    filtered.loc[:, "signature"] = filtered["signature"].apply(cleaner.clean)
    staff_signatures = defaultdict(set)
    for sid, sig in zip(filtered["id"], filtered["signature"]):
        if sig:
            staff_signatures[sid].add(sig)
    for sid, sig in manual_signatures.items():
        staff_signatures[sid].add(sig)
    return staff_signatures


def preprocess_messages(df, cleaner):
    df = df.fillna("")
    df.loc[:, "subject"] = df["subject"].apply(cleaner.clean)
    df.loc[:, "ticket_message"] = df["ticket_message"].apply(cleaner.clean)
    df.loc[:, "reply_message"] = df["reply_message"].apply(cleaner.clean)
    return df


def build_dialogs_and_meta(replies_df, staff_signatures, cleaner):
    replies_df.loc[:, "reply_message"] = replies_df.apply(
        lambda row: remove_signature_from_message(
            row["reply_message"], row["staffid"], staff_signatures
        ),
        axis=1,
    )
    ticket_dialogs = defaultdict(list)
    ticket_messages = {}
    ticket_categories = {}
    ticket_subjects = {}
    for _, row in replies_df.iterrows():
        role = "Пользователь" if int(row.staffid) == 0 else "Оператор"
        message = cleaner.clean(row.reply_message)
        if message:
            ticket_dialogs[row.ticket_id].append({"role": role, "message": message})
        if row.ticket_id not in ticket_messages:
            ticket_messages[row.ticket_id] = cleaner.clean(row.ticket_message)
        if row.ticket_id not in ticket_categories:
            ticket_categories[row.ticket_id] = row.category_title.strip()
        if row.ticket_id not in ticket_subjects:
            ticket_subjects[row.ticket_id] = row.subject
    return ticket_dialogs, ticket_messages, ticket_categories, ticket_subjects


def build_dataset(
    ticket_dialogs, ticket_messages, ticket_categories, ticket_subjects, unwanted_labels
):
    dataset = []
    for ticket_id, messages in ticket_dialogs.items():
        idx = max(i for i, m in enumerate(messages) if m["role"] == "Оператор")
        last_operator_msg = messages[idx]
        label = normalize_label(last_operator_msg["message"].strip())
        label = sentence_normalize(label)
        if label.strip() in unwanted_labels:
            continue
        dialogue = build_dialogue(messages, idx)
        category = ticket_categories.get(ticket_id, "неизвестно")
        subject = ticket_subjects.get(ticket_id, "Без темы")
        user_msg = ticket_messages.get(ticket_id, "")
        full_text = (
            f"Категория: {category}\n"
            f"Тема: {subject}\n"
            f"Пользователь: {user_msg}\n"
            f"{dialogue + chr(10) if dialogue else ''}Оператор: "
        )
        dataset.append(
            {
                "text": full_text,
                "label": label,
                "category_title": category,
                "ticket_id": ticket_id,
                "subject": subject,
                "ticket_message": user_msg,
            }
        )
    return dataset


def print_label_stats(dataset):
    label_counter = Counter(entry["label"] for entry in dataset)
    sorted_label_counts = sorted(
        label_counter.items(), key=lambda x: x[1], reverse=True
    )
    print("\n🔢 Топ повторяющихся ответов (label):")
    for label, count in sorted_label_counts[:20]:
        print(f"{count:4} × {label}")


def main():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SET NAMES utf8mb3"))
        replies_df = pd.read_sql(repiles_query(), conn)
        signatures_df = pd.read_sql("SELECT id, signature FROM hesk_users", conn)

    cleaner = TextCleaner()
    manual_signatures = {
        21: "С уважением, начальник отдела сопровождения и поддержки Попков Александр Юрьевич",
        26: "Юлия Митузина Отдел сопровождения и поддержки Центра разработки и внедрения Тел.: (3822) 900-157, внутр. 1130",
        33: "Начальник ЦИТС ТУСУР. г. Томск, ул. Красноармейская, 146, каб. 804 Тел. (3822) 701-515 (внутр. 2436)",
    }
    staff_signatures = prepare_signatures(signatures_df, cleaner, manual_signatures)
    replies_df = preprocess_messages(replies_df, cleaner)
    ticket_dialogs, ticket_messages, ticket_categories, ticket_subjects = (
        build_dialogs_and_meta(replies_df, staff_signatures, cleaner)
    )
    unwanted_labels = {
        ":).",
        ".",
        "[URL] [URL].",
        "[URL].",
        "123.",
        "2.",
        "Ааааааааа, поправил).",
        "Анна Владимировна, здравствуйте! Все готово).",
        "Анонс этой конференции уже был выложен на сайте [URL].",
        "Баннер в процессе, ищем картинку.",
        "Баннер разместили.",
        "Баннер убрал, текст поправил).",
        "Баннер удалил.",
        "Баннеры заменил.",
        "Вернее тут картинку поменял, а по прошлой задаче ошибку исправил)).",
        "Вернул).",
        "Все готово.",
        "Всё готово.",
        "Все поправил.",
        "Всё, нашла. Удалила, через пару минут отобразится.",
        "Выполнено.",
        "Готово [URL].",
        "Готово, поправил.",
        "Готово, после обновления отобразится.",
        "Готово, таблицу перенес.",
        "Готово: [URL].",
        "Готово!",
        "Готово.",
        "Готово).",
        "Да.",
        "Дату заменили.",
        "Даты исправил.",
        "Даша, привет! [URL].",
        "Добавил, сейчас появится.",
        "Добавил.",
        "Добавил).",
        "Добавила, скоро отобразится.",
        "Добавила, чуть позже появится!",
        "Добавила.",
        "Добавили.",
        "Доброе утро, убрал.",
        "Доброе утро! [URL].",
        "Доброе утро! Все готово).",
        "Доброе утро! Все изменения внесены.",
        "Доброе утро! Вчерашнее) [URL].",
        "Доброе утро! Выложили [URL].",
        "Доброе утро! Готово [URL].",
        "Доброе утро! Готово: [URL] Ссылка не открывается ни в одном браузере. А у вас?",
        "Доброе утро! Готово: [URL].",
        "Доброе утро! Готово.",
        "Доброе утро! Добавил.",
        "Доброе утро! Добавил).",
        "Доброе утро! Заменил.",
        "Доброе утро! Источник поправил.",
        "Доброе утро! Картинку добавил.",
        "Доброе утро! Опубликовано: [URL].",
        "Доброе утро! Первый баннер разместил.",
        "Доброе утро! Письмо-согласие заменил.",
        "Доброе утро! Поправил.",
        "Доброе утро! Программу разместили.",
        "Доброе утро! Разместил.",
        "Доброе утро! Текст поправил.",
        "Доброе утро! Убрала.",
        "Доброе утро) готово.",
        "Доброе утро) картинку поставил на место).",
        "Добрый вечер! Исправили) В течение 15 минут появится.",
        "Добрый день :) готово!",
        "Добрый день [URL].",
        "Добрый день, готово.",
        "Добрый день! [URL] Картинка из вложения не открывается.",
        "Добрый день! [URL].",
        "Добрый день! Баннеры добавили.",
        "Добрый день! Все готово [URL].",
        "Добрый день! Выложили в новостную ленту [URL].",
        "Добрый день! Готово [URL].",
        "Добрый день! Готово, отобразится чуть позже.",
        "Добрый день! Готово: [URL].",
        "Добрый день! Готово!",
        "Добрый день! Готово.",
        "Добрый день! Готово).",
        "Добрый день! Дату изменил.",
        "Добрый день! Доступ восстановили.",
        "Добрый день! Заменил.",
        "Добрый день! Заменила, скоро отобразится.",
        "Добрый день! Заменила. Внутри тоже исправила.",
        "Добрый день! Заменили.",
        "Добрый день! Ошибку исправили.",
        "Добрый день! Починили.",
        "Дубль.",
        "Заменил.",
        "Здравствуйте, готово [URL].",
        "Здравствуйте, готово.",
        "Здравствуйте! [URL] [URL].",
        "Здравствуйте! [URL].",
        "Здравствуйте! Вернули.",
        "Здравствуйте! Восстановили.",
        "Здравствуйте! Все готово.",
        "Здравствуйте! Готов [URL].",
        "Здравствуйте! Готово - [URL].",
        "Здравствуйте! Готово [URL].",
        "Здравствуйте! Готово.",
        "Здравствуйте! Добавил.",
        "Здравствуйте! Добавили.",
        "Здравствуйте! Заменил.",
        "Здравствуйте! Заменили.",
        "Здравствуйте! Исправили.",
        "Здравствуйте! Исправлено.",
        "Здравствуйте! Оформили.",
        "Здравствуйте! Ошибку исправили.",
        "Здравствуйте! Ошибку исправили.",
        "Здравствуйте! Поправил.",
        "Здравствуйте! Поправили.",
        "Здравствуйте! Права добавили.",
        "Здравствуйте! Разместил.",
        "Здравствуйте! Роль выдали.",
        "Здравствуйте! Убрал.",
        "Здравствуйте! Удалил.",
        "Здравствуйте!",
        "Исправили.",
        "Исправлено.",
        "Новость размещена [URL].",
        "Ну всё всё.",
        "Ответ.",
        "Отправлено!",
        "Отчислил.",
        "Ошибку исправили.",
        "Полностью поддерживаю)))).",
        "Поправил.",
        "Права добавлены.",
        "Принято.",
        "Проблема исправлена.",
        "Проблема решена.",
        "Продублировал.",
        "Решена.",
        "Связь проверена.",
        "Удалил.",
        "Хорошо.",
    }
    dataset = build_dataset(
        ticket_dialogs,
        ticket_messages,
        ticket_categories,
        ticket_subjects,
        unwanted_labels,
    )
    with open("dialogue_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ Сохранено {len(dataset)} диалогов")
    print_label_stats(dataset)


if __name__ == "__main__":
    main()
