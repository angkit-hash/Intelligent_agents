"""Command-line entry point for the RFP response assistant.

Usage:
    python app/main.py Do you support SSO and SAML for enterprise customers?

If no question is given as arguments, a default sample question is used so
the demo works out of the box with no arguments.
"""

from __future__ import annotations

import sys

from app.agent import run_agent


def main() -> None:
    """Read the RFP question from argv (or use a default) and print the draft."""
    question = " ".join(sys.argv[1:])
    if not question:
        question = "What uptime SLA do you guarantee, and what happens if you miss it?"

    answer = run_agent(question)
    print("\nSuggested draft / review flag:\n")
    print(answer)


if __name__ == "__main__":
    main()
