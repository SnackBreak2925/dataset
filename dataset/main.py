import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from html2text import HTML2Text
import re

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
    return create_engine(connection_string)

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
        SET cleaned = REGEXP_REPLACE(cleaned, '[ ]+', ' ');
        SET cleaned = REGEXP_REPLACE(cleaned, ' \\\\.', '\\\\.');
        SET cleaned = TRIM(cleaned);

        RETURN cleaned;
    END
"""
    with engine.connect() as conn:
        conn.execute(text(sql_function))
        conn.commit()


def main():
    engine = get_engine()
    define_sql_function(engine)

    query = """
SELECT DISTINCT
    hesk_tickets.id,
    clean_text(hesk_tickets.message) AS ticket_message,
    REPLACE(
        clean_text(hesk_replies.message),
        clean_text(hesk_users.signature),
        '[SIGNATURE]'
    ) AS reply_message,
    hesk_categories.name AS category_title
FROM hesk_tickets
INNER JOIN hesk_replies ON hesk_replies.replyto = hesk_tickets.id
INNER JOIN hesk_categories ON hesk_categories.id = hesk_tickets.category
LEFT JOIN hesk_users ON hesk_replies.staffid = hesk_users.id OR hesk_replies.staffid = 0
WHERE hesk_tickets.replies > 2 AND hesk_tickets.id = 6620 OR hesk_tickets.id = 6870
ORDER BY hesk_tickets.id DESC
LIMIT 100
"""

    email_regex = r'(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))'
    url_regex = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)"

    df = pd.read_sql(query, engine)
    print(df)
    h = HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.single_line = True
    h.body_width = 0
    for row in df.itertuples():
        text = row.reply_message
        text = h.handle(text).strip()
        text = re.sub(email_regex, "[EMAIL]", text)
        text = re.sub(url_regex, "[URL]", text)
        print(row.id, repr(text))
