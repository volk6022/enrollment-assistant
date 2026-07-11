from __future__ import annotations

from app.orchestrator_v11 import V11Orchestrator


def main() -> None:
    orchestrator = V11Orchestrator()
    questions = [
        "Сколько баллов нужно для поступления на специалитет?",
        "Какие документы нужны для поступления?",
        "Как подать документы?",
        "Можно ли поступить без ЕГЭ?",
    ]
    for q in questions:
        print("=" * 80)
        print(q)
        result = orchestrator.answer(q)
        print(result["answer"])
        print(result.get("meta"))


if __name__ == "__main__":
    main()
