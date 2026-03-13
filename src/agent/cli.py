import argparse

from dotenv import load_dotenv

from agent.core import make_agent
from agent.memory import get_memory_strategy


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
    parser.add_argument(
        "--memory",
        default="none",
        choices=["none", "history", "summary", "semantic"],
        help="Cross-conversation memory strategy (default: none)",
    )
    parser.add_argument(
        "--user",
        default="default",
        help="User ID for memory storage and retrieval (default: 'default')",
    )
    args = parser.parse_args()

    strategy = get_memory_strategy(args.memory)
    messages = []

    print(f"Chat started (memory: {args.memory}, user: {args.user}). Type 'quit' to exit.\n")

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

        memory_context = strategy.retrieve(args.user, user_input)
        agent = make_agent(args.model, args.system, memory_context or None)

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        ai_msg = result["messages"][-1]
        print(f"\nAssistant: {ai_msg.content}\n")
        messages = result["messages"]

        new_turn = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_msg.content},
        ]
        strategy.store(args.user, new_turn)


if __name__ == "__main__":
    main()
