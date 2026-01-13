import json
import time
from pathlib import Path

from src.rag_pkg.qa import answer_repo

EVAL_FILE = Path("data/rag_eval_questions.jsonl")


def main() -> None:
    # Load questions from JSONL, skip blank lines
    with EVAL_FILE.open("r") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    # Limit questions to avoid OpenAI rate limits
    questions = questions[:1]  # increase later if you want more

    total = len(questions)

    for q in questions:
        question = q["question"]
        print(f"Q: {question}")

        # answer_repo returns a string, not a dict
        ans = answer_repo(question)

        print("---")
        print(ans)
        print("===")
        print()

        # Sleep to avoid 429 rate limit errors
        time.sleep(25)

    print(f"Eval questions run: {total}")


if __name__ == "__main__":
    main()
