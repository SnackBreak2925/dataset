import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from html2text import HTML2Text
import re


def main():
    load_dotenv()
    db_config = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "dialect": os.getenv("DB_DIALECT"),
    }

    connection_string = (
        f"{db_config['dialect']}://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(connection_string)

    query = """
SELECT DISTINCT
    hesk_tickets.message AS ticket_message,
    REPLACE(
        REPLACE(
            hesk_replies.message, '<br />', ''
        ), hesk_users.signature, ''
    ) AS reply_message,
    hesk_categories.name
FROM hesk_tickets
INNER JOIN hesk_replies ON hesk_replies.replyto = hesk_tickets.id
INNER JOIN hesk_categories ON hesk_categories.id = hesk_tickets.category
LEFT JOIN hesk_users ON hesk_replies.staffid = hesk_users.id OR hesk_replies.staffid = 0
WHERE hesk_tickets.id = 6870
"""

    email_regex = r'(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))'
    url_regex = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)"

    df = pd.read_sql(query, engine)
    print(df)
    h = HTML2Text()
    h.ignore_links = True
    for row in df.itertuples():
        text = row.reply_message
        text = re.sub(email_regex, "[EMAIL]", text)
        text = re.sub(url_regex, "[URL]", text)
        text = h.handle(text)
        print(" ".join(text.split()) + "\n")
