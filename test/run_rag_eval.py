# 📤 Запуск RAG
from rag_engine import rag_answer
from load_kb import load_kb

kb = load_kb()
q = "Как восстановить пароль?"
print(rag_answer(q, "ИТ", kb))
