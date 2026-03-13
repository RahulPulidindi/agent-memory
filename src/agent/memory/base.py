from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryStrategy(Protocol):
    """Interface for cross-conversation memory strategies."""

    def retrieve(self, user_id: str, query: str) -> str:
        """Return prior context to inject into the system prompt.

        query is used by semantic memory for relevance scoring; other strategies ignore it.
        Returns an empty string if no memory exists.
        """
        ...

    def store(self, user_id: str, messages: list) -> None:
        """Persist the current turn. Pass only the new user/assistant exchange."""
        ...

    def clear(self, user_id: str) -> None:
        """Delete all stored memory for a user."""
        ...


class NoMemory:
    """Null object — no storage, no retrieval. Preserves stateless baseline behaviour."""

    def __init__(self, **kwargs):
        pass

    def retrieve(self, user_id: str, query: str) -> str:
        return ""

    def store(self, user_id: str, messages: list) -> None:
        pass

    def clear(self, user_id: str) -> None:
        pass
