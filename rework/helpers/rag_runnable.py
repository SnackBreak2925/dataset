from rag_pipeline_by_category import auto_detect_category, generate_with_contexts


def get_rag_answer(ticket_message):
    category = auto_detect_category(ticket_message)
    if not category:
        return "[Категория не определена]"
    answers, _ = generate_with_contexts(category, ticket_message, top_k=1)
    return answers[0] if answers else "[Нет ответа]"

def get_rag_beams(ticket_message, top_k=5):
    category = auto_detect_category(ticket_message)
    if not category:
        return ["[Категория не определена]"]
    answers, _ = generate_with_contexts(category, ticket_message, top_k=top_k)
    return answers if answers else ["[Нет ответа]"]
