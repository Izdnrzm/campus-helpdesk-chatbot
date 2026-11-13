"""Command-line interface for the Campus IT Helpdesk RAG chatbot.

Run:  python chatbot.py
Commands: 'sources' toggles showing retrieved FAQ rows, 'reset' clears chat history, 'quit' exits.
"""

import sys

from rag import RagChatbot


def main():
    try:  # load GEMINI_API_KEY from a local .env file if python-dotenv is installed
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    bot = RagChatbot()
    show_sources = True
    print(f"Campus IT Helpdesk bot [{bot.mode}] - {len(bot.df)} FAQ entries indexed.")
    print("Ask a question. Commands: sources | reset | quit\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        cmd = query.lower()
        if cmd in {"quit", "exit", "q"}:
            break
        if cmd == "reset":
            bot.reset()
            print("Bot: Conversation cleared.\n")
            continue
        if cmd == "sources":
            show_sources = not show_sources
            print(f"Bot: Source display {'on' if show_sources else 'off'}.\n")
            continue

        reply = bot.ask(query)
        print(f"Bot: {reply.text}")
        if show_sources and reply.sources:
            refs = "; ".join(
                f"#{i + 2} \"{bot.df.loc[i, 'Question']}\" ({s:.2f})" for i, s in reply.sources
            )
            print(f"     [sources: faq.csv row {refs}]")
        print()


if __name__ == "__main__":
    sys.exit(main())
