import os
import json
import dotenv
import re
import pandas as pd
from sqlalchemy import create_engine, text
from collections import defaultdict
from html2text import HTML2Text

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


def define_sql_function(engine):
    sql_function = """
CREATE FUNCTION IF NOT EXISTS clean_text(input_text LONGTEXT)
    RETURNS LONGTEXT
    DETERMINISTIC
    BEGIN
        DECLARE cleaned LONGTEXT;
        SET cleaned = REGEXP_REPLACE(input_text, '<[^>]+>', ' ');
        SET cleaned = REGEXP_REPLACE(cleaned, '<br />', ' ');
        SET cleaned = REGEXP_REPLACE(cleaned, '[\r\n]+', ' ');
        SET cleaned = REGEXP_REPLACE(cleaned, '\\s{2,}', ' ');
        SET cleaned = REGEXP_REPLACE(cleaned, ' \\.', '\\.');
        SET cleaned = TRIM(cleaned);
        RETURN cleaned;
    END
"""
    with engine.connect() as conn:
        conn.execute(text(sql_function))
        conn.commit()


def repiles_query():
    return """
SELECT
    messages.id,
    tickets.id AS ticket_id,
    tickets.subject AS subject,
    clean_text(tickets.message) AS ticket_message,
    COALESCE(
        REPLACE(
            clean_text(messages.message),
            clean_text(staff.signature),
            '[SIGNATURE]'
        ),
        clean_text(messages.message)
    ) AS reply_message,
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
    if type(text_data) is not str:
        return ""
    h = HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.single_line = True
    h.body_width = 0
    result = h.handle(text_data).strip()
    result = re.sub(EMAIL_REGEX, "[EMAIL]", result)
    result = re.sub(r"\[SIGNATURE\]", "", result)
    result = re.sub(r"\[URL\]", "", result)
    result = re.sub(URL_REGEX, "", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip()


if __name__ == "__main__":
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text("SET NAMES utf8mb3"))
        replies_df = pd.read_sql(repiles_query(), conn)

    replies_df = replies_df.fillna("")

    ticket_dialogs = defaultdict(list)
    ticket_messages = {}
    ticket_categories = {}

    for _, row in replies_df.iterrows():
        role = "Пользователь" if row.staffid == 0 else "Оператор"
        message = clean_text(row.reply_message)
        if message:
            ticket_dialogs[row.ticket_id].append({"role": role, "message": message})
        if row.ticket_id not in ticket_messages:
            ticket_messages[row.ticket_id] = clean_text(row.ticket_message)
        if row.ticket_id not in ticket_categories:
            ticket_categories[row.ticket_id] = row.category_title.strip()

    dataset = []
    for ticket_id, messages in ticket_dialogs.items():
        last = messages[-1]
        if last["role"] == "Оператор":
            dialogue = "\n".join(
                [f"{m['role']}: {m['message']}" for m in messages[:-1]]
            )
            category = ticket_categories.get(ticket_id, "неизвестно")
            full_text = f"Категория: {category}\nПервичное сообщение: {ticket_messages.get(ticket_id, '')}\n{dialogue}"
            dataset.append(
                {
                    "text": full_text,
                    "label": last["message"],
                    "category_title": category,
                    "ticket_id": ticket_id,
                }
            )

    with open("dialogue_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Сохранено {len(dataset)} диалогов в dialogue_dataset.json")
