import os
import json
import dotenv
import re
import pandas as pd
from sqlalchemy import create_engine, text
from collections import defaultdict
import html

EMAIL_REGEX = r'(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))'
URL_REGEX = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)"


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


def clean_text(text_data):
    if not isinstance(text_data, str) or not text_data.strip():
        return ""
    text = html.unescape(text_data)
    substitutions = [
        (r"<[^>]+>", " "),
        (r"<br\s*/?>", " "),
        (EMAIL_REGEX, "[EMAIL]"),
        (r"\[SIGNATURE\]", ""),
        (URL_REGEX, "[URL]"),
        (r"[\r\n]+", " "),
        (r"\s{2,}", " "),
        (r"\s+\.", "."),
    ]
    for pat, repl in substitutions:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text.strip()


def remove_signature_from_message(row):
    message = row["reply_message"]
    staffid = row["staffid"]
    if staffid != 0:
        signature = staff_signatures.get(staffid)
        if signature and signature.strip():
            message = message.replace(signature, "")
    return message


def build_dialogue(messages, upto_idx):
    if upto_idx == 0:
        return ""
    return "\n".join(f"{m['role']}: {m['message']}" for m in messages[:upto_idx])


if __name__ == "__main__":
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text("SET NAMES utf8mb3"))
        replies_df = pd.read_sql(repiles_query(), conn)
        signatures_df = pd.read_sql("SELECT id, signature FROM hesk_users", conn)

    # убрать пустые
    filtered_signatures_df = signatures_df[signatures_df["signature"].astype(bool)]
    filtered_signatures_df.loc[:, "signature"] = filtered_signatures_df[
        "signature"
    ].apply(clean_text)
    staff_signatures = dict(
        zip(filtered_signatures_df["id"], filtered_signatures_df["signature"])
    )

    replies_df = replies_df.fillna("")

    replies_df.loc[:, "subject"] = replies_df["subject"].apply(clean_text)
    replies_df.loc[:, "ticket_message"] = replies_df["ticket_message"].apply(clean_text)
    replies_df.loc[:, "reply_message"] = replies_df["reply_message"].apply(clean_text)
    replies_df.loc[:, "reply_message"] = replies_df.apply(
        remove_signature_from_message, axis=1
    )

    ticket_dialogs = defaultdict(list)
    ticket_messages = {}
    ticket_categories = {}
    ticket_subjects = {}

    for _, row in replies_df.iterrows():
        role = "Пользователь" if int(row.staffid) == 0 else "Оператор"
        message = clean_text(row.reply_message)
        if message:
            ticket_dialogs[row.ticket_id].append({"role": role, "message": message})
        if row.ticket_id not in ticket_messages:
            ticket_messages[row.ticket_id] = clean_text(row.ticket_message)
        if row.ticket_id not in ticket_categories:
            ticket_categories[row.ticket_id] = row.category_title.strip()
        if row.ticket_id not in ticket_subjects:
            ticket_subjects[row.ticket_id] = row.subject

    dataset = []
    for ticket_id, messages in ticket_dialogs.items():
        idx = max(i for i, m in enumerate(messages) if m["role"] == "Оператор")

        last_operator_msg = messages[idx]
        if last_operator_msg["message"].strip() == "[URL]":
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
                "label": last_operator_msg["message"],
                "category_title": category,
                "ticket_id": ticket_id,
                "subject": subject,
                "ticket_message": user_msg,
            }
        )

    with open("dialogue_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Сохранено {len(dataset)} диалогов в dialogue_dataset.json")

    with open("dialogue_dataset.json", encoding="utf-8") as f:
        raw_data = json.load(f)
