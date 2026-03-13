from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent


def make_agent(
    model_str: str = "anthropic:claude-haiku-4-5-20251001",
    system_prompt: str | None = None,
    memory_context: str | None = None,
):
    """Create a deep agent with the given model.

    model_str: "provider:model" format, e.g. "openai:gpt-4o".
    memory_context: recalled context from prior conversations, prepended to the
                    system prompt so the agent can reference it directly.
    """
    model = init_chat_model(model_str)

    full_prompt: str | None = None
    if memory_context:
        base = system_prompt or ""
        full_prompt = (
            f"The following is verified memory from prior conversations with this user. "
            f"Treat it as your own recall and use it confidently and directly in your answers:\n\n"
            f"{memory_context}\n\n"
            f"{base}".strip()
        )
    elif system_prompt:
        full_prompt = system_prompt

    kwargs = {}
    if full_prompt:
        kwargs["system_prompt"] = full_prompt

    return create_deep_agent(model=model, **kwargs)
