from agent.memory.agentic import AgenticMemory
from agent.memory.base import MemoryStrategy, NoMemory
from agent.memory.history import FullHistoryMemory
from agent.memory.semantic import SemanticMemory
from agent.memory.summary import SummaryMemory


def get_memory_strategy(name: str, **kwargs) -> MemoryStrategy:
    """Instantiate a memory strategy by name. kwargs are passed to the constructor."""
    strategies = {
        "none": NoMemory,
        "history": FullHistoryMemory,
        "summary": SummaryMemory,
        "semantic": SemanticMemory,
        "agentic": AgenticMemory,
    }
    if name not in strategies:
        raise ValueError(f"Unknown memory strategy '{name}'. Choose from: {list(strategies)}")
    return strategies[name](**kwargs)


__all__ = [
    "MemoryStrategy",
    "NoMemory",
    "FullHistoryMemory",
    "SummaryMemory",
    "SemanticMemory",
    "AgenticMemory",
    "get_memory_strategy",
]