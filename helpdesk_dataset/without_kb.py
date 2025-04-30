import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from html2text import HTML2Text
import re
from datasets import Dataset

EMAIL_REGEX = r'(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))'
URL_REGEX = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)"


def load_db_config():
    load_dotenv()
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
        connect_args={
            "charset": "utf8mb3",
            "collation": "utf8mb3_unicode_ci",
        },
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
        SET cleaned = REGEXP_REPLACE(cleaned, '\\\\s{2,}', ' ');
        SET cleaned = REGEXP_REPLACE(cleaned, ' \\\\.', '\\\\.');
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
    hesk_tickets.id AS ticket_id,
    hesk_tickets.subject AS subject,
    clean_text(hesk_tickets.message) AS ticket_message,
    COALESCE(
        REPLACE(
            clean_text(messages.message),
            clean_text(hesk_users.signature),
            '[SIGNATURE]'
        ),
        clean_text(messages.message)
    ) AS reply_message,
    hesk_categories.name AS category_title
FROM hesk_replies AS messages
INNER JOIN hesk_tickets ON hesk_tickets.id = messages.replyto AS tickets
INNER JOIN hesk_categories ON hesk_categories.id = hesk_tickets.category AS category
INNER JOIN hesk_users ON messages.staffid = hesk_users.id AS staff
WHERE hesk_tickets.status = 3 AND messages.staffid != 0 AND messages.rating == 5
"""


def clean_text(text_data):
    if type(text_data) is not str:
        return
    h = HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.single_line = True
    h.body_width = 0
    result = text_data
    result = h.handle(result).strip()
    result = re.sub(EMAIL_REGEX, "[EMAIL]", result)
    result = re.sub(URL_REGEX, "[URL]", result)

    return result


def main():
    engine = get_engine()
    define_sql_function(engine)

    replies_df = pd.read_sql(repiles_query(), engine)

    replies_df_clean = replies_df.map(clean_text)

    replies_dataset = Dataset.from_pandas(replies_df_clean)
    replies_dataset = replies_dataset.map(
        lambda x: {
            "text": f"Тикет: {x['ticket_message']}\nОтвет: {x['reply_message']}",
            "metadata": {
                "category": x["category_title"],
                "subject": x["subject"],
                "ticket_id": x["ticket_id"],
                "message_id": x["id"],
            },
        }
    )

    replies_dataset.save_to_disk(os.getenv("PWD") + "/ticket_replies_dataset")

if __name__ == '__main__':
    main()
