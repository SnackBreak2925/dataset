import pandas as pd
from sqlalchemy import create_engine


def main():
    db_connection_string = "mysql+pymysql://snack:password@localhost:3306/help"

    engine = create_engine(db_connection_string)

    query = """
    SELECT hesk_tickets.message AS ticket_message, hesk_replies.message AS reply_message, hesk_categories.name FROM hesk_tickets
    INNER JOIN hesk_replies ON hesk_replies.replyto = hesk_tickets.id
    INNER JOIN hesk_categories ON hesk_categories.id = hesk_tickets.category
    WHERE hesk_tickets.id = 6870
    """

    df = pd.read_sql(query, engine)
    print(df.to_dict())
