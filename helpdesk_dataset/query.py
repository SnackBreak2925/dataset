from rag_engine import rag_answer
from load_kb import load_kb


def ask_rag(question, category):
    kb = load_kb()
    return rag_answer(question, category, kb)
