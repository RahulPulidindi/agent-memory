import os
import tempfile

import pytest
from dotenv import load_dotenv

from agent.core import make_agent
from agent.memory import AgenticMemory, FullHistoryMemory, SemanticMemory, SummaryMemory, get_memory_strategy

load_dotenv()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent():
    return make_agent()


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a fresh temporary SQLite database path per test."""
    return str(tmp_path / "test_memory.db")


# ---------------------------------------------------------------------------
# Baseline agent tests (unchanged from starter)
# ---------------------------------------------------------------------------


def test_agent_responds(agent):
    """Agent should return a non-empty response to a simple question."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 2 + 2?"}]}
    )
    assert len(result["messages"]) > 1
    ai_msg = result["messages"][-1]
    assert ai_msg.content
    assert "4" in ai_msg.content


def test_agent_multi_turn(agent):
    """Agent should handle multi-turn conversation."""
    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Alice."}]}
    )
    msgs = r1["messages"]
    msgs.append({"role": "user", "content": "What is my name?"})
    r2 = agent.invoke({"messages": msgs})
    ai_msg = r2["messages"][-1]
    assert "Alice" in ai_msg.content


# ---------------------------------------------------------------------------
# Cross-conversation memory tests
# ---------------------------------------------------------------------------


def _run_session_1(strategy, user_id: str, model_str: str) -> None:
    """Plant facts in Session 1 and store them via the strategy."""
    messages = []
    turns = [
        "My name is Jordan. I'm a staff engineer at Meridian Labs.",
        "Our team works in Python exclusively. We use FastAPI for our services and deploy to GCP.",
    ]
    for turn in turns:
        agent = make_agent(model_str)
        messages.append({"role": "user", "content": turn})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        ai_content = messages[-1].content
        strategy.store(user_id, [
            {"role": "user", "content": turn},
            {"role": "assistant", "content": ai_content},
        ])


def _run_session_2_question(strategy, user_id: str, question: str, model_str: str) -> str:
    """Start a fresh session (no prior messages) and ask a recall question."""
    memory_context = strategy.retrieve(user_id, question)
    agent = make_agent(model_str, memory_context=memory_context or None)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


MODEL = "anthropic:claude-haiku-4-5-20251001"
USER = "test_user"


class TestFullHistoryMemory:
    def test_recalls_name_across_sessions(self, tmp_db):
        strategy = FullHistoryMemory(db_path=tmp_db)
        _run_session_1(strategy, USER, MODEL)

        response = _run_session_2_question(
            strategy, USER, "What is my name?", MODEL
        )
        assert "Jordan" in response

    def test_recalls_company_across_sessions(self, tmp_db):
        strategy = FullHistoryMemory(db_path=tmp_db)
        _run_session_1(strategy, USER, MODEL)

        response = _run_session_2_question(
            strategy, USER, "What company do I work at?", MODEL
        )
        assert "Meridian" in response

    def test_recalls_tech_stack_across_sessions(self, tmp_db):
        strategy = FullHistoryMemory(db_path=tmp_db)
        _run_session_1(strategy, USER, MODEL)

        response = _run_session_2_question(
            strategy, USER, "What programming language and framework does my team use?", MODEL
        )
        assert "Python" in response or "FastAPI" in response

    def test_clear_removes_memory(self, tmp_db):
        strategy = FullHistoryMemory(db_path=tmp_db)
        _run_session_1(strategy, USER, MODEL)
        strategy.clear(USER)

        memory = strategy.retrieve(USER, "What is my name?")
        assert memory == ""

    def test_user_isolation(self, tmp_db):
        strategy = FullHistoryMemory(db_path=tmp_db)
        _run_session_1(strategy, USER, MODEL)

        # A different user should have no memory
        other_memory = strategy.retrieve("other_user", "What is my name?")
        assert other_memory == ""


class TestSummaryMemory:
    def test_recalls_name_across_sessions(self, tmp_db):
        strategy = SummaryMemory(db_path=tmp_db, model_str=MODEL)
        _run_session_1(strategy, USER, MODEL)

        response = _run_session_2_question(
            strategy, USER, "What is my name?", MODEL
        )
        assert "Jordan" in response

    def test_recalls_tech_stack_across_sessions(self, tmp_db):
        strategy = SummaryMemory(db_path=tmp_db, model_str=MODEL)
        _run_session_1(strategy, USER, MODEL)

        response = _run_session_2_question(
            strategy, USER, "What language and framework does my team use?", MODEL
        )
        assert "Python" in response or "FastAPI" in response

    def test_clear_removes_memory(self, tmp_db):
        strategy = SummaryMemory(db_path=tmp_db, model_str=MODEL)
        _run_session_1(strategy, USER, MODEL)
        strategy.clear(USER)

        memory = strategy.retrieve(USER, "What is my name?")
        assert memory == ""

    def test_summary_is_compact(self, tmp_db):
        """Summary should be significantly shorter than full message history."""
        strategy_summary = SummaryMemory(db_path=tmp_db, model_str=MODEL)
        history_db = tmp_db.replace(".db", "_hist.db")
        strategy_history = FullHistoryMemory(db_path=history_db)

        _run_session_1(strategy_summary, USER, MODEL)
        _run_session_1(strategy_history, USER, MODEL)

        summary_ctx = strategy_summary.retrieve(USER, "anything")
        history_ctx = strategy_history.retrieve(USER, "anything")

        assert len(summary_ctx) < len(history_ctx)


class TestSemanticMemory:
    def test_recalls_name_across_sessions(self, tmp_db):
        strategy = SemanticMemory(db_path=tmp_db)
        _run_session_1(strategy, USER, MODEL)

        response = _run_session_2_question(
            strategy, USER, "What is my name?", MODEL
        )
        assert "Jordan" in response

    def test_recalls_tech_stack_across_sessions(self, tmp_db):
        strategy = SemanticMemory(db_path=tmp_db)
        _run_session_1(strategy, USER, MODEL)

        response = _run_session_2_question(
            strategy, USER, "What programming language and cloud provider does my team use?", MODEL
        )
        assert "Python" in response or "GCP" in response

    def test_clear_removes_memory(self, tmp_db):
        strategy = SemanticMemory(db_path=tmp_db)
        _run_session_1(strategy, USER, MODEL)
        strategy.clear(USER)

        memory = strategy.retrieve(USER, "What is my name?")
        assert memory == ""

    def test_user_isolation(self, tmp_db):
        strategy = SemanticMemory(db_path=tmp_db)
        _run_session_1(strategy, USER, MODEL)

        other_memory = strategy.retrieve("other_user", "What is my name?")
        assert other_memory == ""


class TestMemoryFactory:
    def test_factory_none(self):
        s = get_memory_strategy("none")
        assert s.retrieve("u", "q") == ""

    def test_factory_history(self, tmp_db):
        s = get_memory_strategy("history", db_path=tmp_db)
        assert isinstance(s, FullHistoryMemory)

    def test_factory_summary(self, tmp_db):
        s = get_memory_strategy("summary", db_path=tmp_db)
        assert isinstance(s, SummaryMemory)

    def test_factory_semantic(self, tmp_db):
        s = get_memory_strategy("semantic", db_path=tmp_db)
        assert isinstance(s, SemanticMemory)

    def test_factory_invalid(self):
        with pytest.raises(ValueError, match="Unknown memory strategy"):
            get_memory_strategy("nonexistent")

    def test_factory_agentic(self, tmp_db):
        s = get_memory_strategy("agentic", db_path=tmp_db)
        assert isinstance(s, AgenticMemory)


# ---------------------------------------------------------------------------
# Agentic memory tests
# ---------------------------------------------------------------------------


def _run_session_1_agentic(strategy, user_id: str, model_str: str) -> None:
    """Plant facts via agentic memory, waiting for async core updates."""
    messages = []
    turns = [
        "My name is Jordan. I'm a staff engineer at Meridian Labs.",
        "Our team works in Python exclusively. We use FastAPI for our services and deploy to GCP.",
    ]
    for turn in turns:
        agent = make_agent(model_str)
        messages.append({"role": "user", "content": turn})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        ai_content = messages[-1].content
        strategy.store(user_id, [
            {"role": "user", "content": turn},
            {"role": "assistant", "content": ai_content},
        ])
        strategy.wait_for_core_update()


class TestAgenticMemory:
    def test_recalls_name_across_sessions(self, tmp_db):
        strategy = AgenticMemory(db_path=tmp_db)
        _run_session_1_agentic(strategy, USER, MODEL)

        response = _run_session_2_question(
            strategy, USER, "What is my name?", MODEL
        )
        assert "Jordan" in response

    def test_recalls_company_across_sessions(self, tmp_db):
        strategy = AgenticMemory(db_path=tmp_db)
        _run_session_1_agentic(strategy, USER, MODEL)

        response = _run_session_2_question(
            strategy, USER, "What company do I work at?", MODEL
        )
        assert "Meridian" in response

    def test_recalls_tech_stack_across_sessions(self, tmp_db):
        strategy = AgenticMemory(db_path=tmp_db)
        _run_session_1_agentic(strategy, USER, MODEL)

        response = _run_session_2_question(
            strategy, USER, "What programming language and framework does my team use?", MODEL
        )
        assert "Python" in response or "FastAPI" in response

    def test_extracts_structured_facts(self, tmp_db):
        """Facts should be stored as discrete extracted statements, not raw turns."""
        strategy = AgenticMemory(db_path=tmp_db)
        _run_session_1_agentic(strategy, USER, MODEL)

        import sqlite3
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT fact FROM agentic_facts WHERE user_id = ?", (USER,)
        ).fetchall()
        conn.close()

        facts = [row["fact"] for row in rows]
        assert len(facts) > 0
        # Facts should be concise extractions, not full conversation turns
        for fact in facts:
            assert len(fact) < 500, f"Fact too long — probably raw turn: {fact[:100]}"

    def test_core_memory_block_exists(self, tmp_db):
        """After storing, a core memory block should exist for the user."""
        strategy = AgenticMemory(db_path=tmp_db)
        _run_session_1_agentic(strategy, USER, MODEL)

        core = strategy._get_core(USER)
        assert core, "Core memory block should not be empty after storing facts"
        assert "Jordan" in core or "Meridian" in core

    def test_clear_removes_all(self, tmp_db):
        strategy = AgenticMemory(db_path=tmp_db)
        _run_session_1_agentic(strategy, USER, MODEL)
        strategy.clear(USER)

        memory = strategy.retrieve(USER, "What is my name?")
        assert memory == ""

    def test_user_isolation(self, tmp_db):
        strategy = AgenticMemory(db_path=tmp_db)
        _run_session_1_agentic(strategy, USER, MODEL)

        other_memory = strategy.retrieve("other_user", "What is my name?")
        assert other_memory == ""

    def test_retrieve_has_both_sections(self, tmp_db):
        """Retrieve output should contain both core memory and relevant facts."""
        strategy = AgenticMemory(db_path=tmp_db)
        _run_session_1_agentic(strategy, USER, MODEL)

        memory = strategy.retrieve(USER, "Tell me about Jordan's team")
        assert "Core Memory" in memory
        assert "Relevant Facts" in memory