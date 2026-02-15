import argparse
import sys

from dotenv import load_dotenv

from agent.core import make_agent


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="CLI Chat Agent")
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5-20251001",
        help="Model string, e.g. openai:gpt-4o, anthropic:claude-haiku-4-5-20251001, google_genai:gemini-2.5-flash",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Custom system prompt",
    )
    args = parser.parse_args()

    agent = make_agent(args.model, args.system)
    messages = []

    print("Chat started. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        ai_msg = result["messages"][-1]
        print(f"\nAssistant: {ai_msg.content}\n")
        messages = result["messages"]


if __name__ == "__main__":
    main()
