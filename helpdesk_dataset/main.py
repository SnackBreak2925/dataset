import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from html2text import HTML2Text
import re
from datasets import Dataset, DatasetDict

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
    hesk_replies.id,
    hesk_tickets.id AS ticket_id,
    hesk_tickets.subject AS subject,
    clean_text(hesk_tickets.message) AS ticket_message,
    COALESCE(
        REPLACE(
            clean_text(hesk_replies.message),
            clean_text(hesk_users.signature),
            '[SIGNATURE]'
        ),
        clean_text(hesk_replies.message)
    ) AS reply_message,
    hesk_categories.name AS category_title
FROM hesk_replies
INNER JOIN hesk_tickets ON hesk_tickets.id = hesk_replies.replyto
INNER JOIN hesk_categories ON hesk_categories.id = hesk_tickets.category
LEFT JOIN hesk_users ON hesk_replies.staffid = hesk_users.id
"""


def articles_query():
    return """
SELECT
    hesk_kb_articles.id,
    hesk_kb_articles.subject as subject,
    clean_text(hesk_kb_articles.content) as instruction,
    hesk_kb_articles.keywords as keywords,
    hesk_kb_categories.name AS category_title
FROM hesk_kb_articles
INNER JOIN hesk_kb_categories ON hesk_kb_categories.id = hesk_kb_articles.catid
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
    knowledge_df = pd.read_sql(articles_query(), engine)

    replies_df_clean = replies_df.map(clean_text)
    knowledge_df_clean = knowledge_df.map(clean_text)

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

    kb_dataset = Dataset.from_pandas(knowledge_df_clean)
    kb_dataset = kb_dataset.map(
        lambda x: {
            "text": f"Тема: {x['subject']}\nИнструкция: {x['instruction']}",
            "metadata": {
                "article_id": x["id"],
                "category": x["category_title"],
                "keywords": x["keywords"],
            },
        }
    )

    dataset_dict = DatasetDict(
        {"tickets": replies_dataset, "knowledge_base": kb_dataset}
    )
    dataset_dict.save_to_disk(os.getenv("PWD") + "/ticket_replies_dataset")
