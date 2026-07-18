from __future__ import annotations

import sys

from app.agent import run_agent


def main() -> None:
    question = " ".join(sys.argv[1:])
    if not question:
        question = "Which ticket mentions a missing trailing slash and what was the resolution?"

    answer = run_agent(question)
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()
