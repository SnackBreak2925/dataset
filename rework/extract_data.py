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
LOGIN_REGEX = r'(логин|login)[^\wа-яё]*:?\s*([\w.@!#$%^&*()_+=\-\[\]{}:;"\'<>.,?/~]+)'
PASSWORD_REGEX = (
    r'(пароль|password)[^\wа-яё]*:?\s*([\w!@#$%^&*()_+=\-\[\]{}:;"\'<>.,?/~]+)'
)
IGNORE_GROUP_PATTERN = re.compile(
    r"\((\s*логин\s*(?:и|или|,|/)?\s*пароль\s*|\s*пароль\s*(?:и|или|,|/)?\s*логин\s*)\)",
    flags=re.IGNORECASE,
)


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

    ignore_spans = [m.span() for m in IGNORE_GROUP_PATTERN.finditer(text)]

    def in_ignore_spans(pos):
        for s, e in ignore_spans:
            if pos >= s and pos < e:
                return True
        return False

    def mask_login(match):
        start = match.start(1)
        if in_ignore_spans(start):
            return match.group(0)
        value = match.group(2).strip()
        s = match.string
        pos = match.start(1)
        left_context = s[pos - 1] if pos > 0 else ""
        right_context = s[match.end(2)] if match.end(2) < len(s) else ""
        # Не маскируем если value — это просто союз или разделитель, либо value пустой
        if not value or value in ("и", "или", "/", ",", "и/или"):
            return match.group(0)
        if left_context in "\"'«" or right_context in "\"'»":
            return match.group(0)
        if len(value.split()) > 1:
            return match.group(0)
        if value.upper() in ("[LOGIN]", "[EMAIL]") or (
            match.end(2) < len(match.string) and match.string[match.end(2)] == "]"
        ):
            return match.group(0)
        # Доработка — если значение начинается с большой буквы (и короткое), не маскировать (скорее всего это подпись)
        if value in {"С", "С уважением"} or (value.istitle() and len(value) <= 3):
            return match.group(0)
        return f"{match.group(1)}: [LOGIN]"

    def mask_password(match):
        start = match.start(1)
        if in_ignore_spans(start):
            return match.group(0)
        value = match.group(2).strip()
        s = match.string
        pos = match.start(1)
        left_context = s[pos - 1] if pos > 0 else ""
        right_context = s[match.end(2)] if match.end(2) < len(s) else ""
        if not value or value in ("и", "или", "/", ",", "и/или"):
            return match.group(0)
        if left_context in "\"'«" or right_context in "\"'»":
            return match.group(0)
        if len(value.split()) > 1:
            return match.group(0)
        if value.upper() == "[PASSWORD]" or (
            match.end(2) < len(match.string) and match.string[match.end(2)] == "]"
        ):
            return match.group(0)
        # Доработка — если после "пароль" идёт "С уважением" или "С", не маскировать!
        if value in {"С", "С уважением"} or (value.istitle() and len(value) <= 3):
            return match.group(0)
        return f"{match.group(1)}: [PASSWORD]"

    text = re.sub(LOGIN_REGEX, mask_login, text, flags=re.IGNORECASE)
    text = re.sub(PASSWORD_REGEX, mask_password, text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
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

    manual_signatures = {
        21: "С уважением, начальник отдела сопровождения и поддержки Попков Александр Юрьевич",
        26: "Юлия Митузина Отдел сопровождения и поддержки Центра разработки и внедрения Тел.: (3822) 900-157, внутр. 1130",
    }

    staff_signatures.update(manual_signatures)

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
        if (
            last_operator_msg["message"].strip() == "[URL]"
            or last_operator_msg["message"].strip() == "[URL] [URL]"
        ):
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
