"""Tests for LangGraphAdapter — all mocked, no LangGraph install needed."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from normlayer.adapters.langgraph_adapter import LangGraphAdapter, _WrappedGraph
from normlayer.base_policy import AgentMessage, BasePolicy, PolicyResult
from normlayer.engine import EnforcementError, PolicyEngine


# ---------------------------------------------------------------------------
# Minimal policy stubs
# ---------------------------------------------------------------------------


class AlwaysPassPolicy(BasePolicy):
    name = "AlwaysPass"

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        return PolicyResult(
            passed=True,
            violation_score=0.0,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="low",
        )


class AlwaysBlockPolicy(BasePolicy):
    name = "AlwaysBlock"

    def __init__(self) -> None:
        super().__init__(handler="block")

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        return PolicyResult(
            passed=False,
            violation_score=1.0,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="high",
            details="Always blocked.",
        )


class AlwaysWarnPolicy(BasePolicy):
    name = "AlwaysWarn"

    def __init__(self) -> None:
        super().__init__(handler="warn")

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        return PolicyResult(
            passed=False,
            violation_score=0.7,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="medium",
            details="Always warned.",
        )


# ---------------------------------------------------------------------------
# Mock LangGraph objects
# ---------------------------------------------------------------------------


class MockBaseMessage:
    """Mimics a LangGraph BaseMessage."""

    def __init__(
        self,
        content: str,
        type: str = "ai",
        name: str | None = None,
        additional_kwargs: dict | None = None,
    ) -> None:
        self.content = content
        self.type = type
        self.name = name
        self.additional_kwargs = additional_kwargs or {}


class MockCompiledGraph:
    """Mimics a compiled LangGraph StateGraph."""

    def __init__(
        self,
        input_msgs: list[MockBaseMessage],
        output_msgs: list[MockBaseMessage],
        messages_key: str = "messages",
    ) -> None:
        self._input_msgs = input_msgs
        self._output_msgs = output_msgs
        self._messages_key = messages_key
        self.some_attr = "graph_attr"

    def invoke(self, state: dict, **kwargs: Any) -> dict:
        result = dict(state)
        existing = list(state.get(self._messages_key, []))
        result[self._messages_key] = existing + self._output_msgs
        return result

    async def ainvoke(self, state: dict, **kwargs: Any) -> dict:
        return self.invoke(state, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLangGraphAdapterInit:
    def test_stores_engine(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = LangGraphAdapter(engine)
        assert adapter.engine is engine

    def test_default_messages_key(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = LangGraphAdapter(engine)
        assert adapter.messages_key == "messages"

    def test_custom_messages_key(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = LangGraphAdapter(engine, messages_key="chat_history")
        assert adapter.messages_key == "chat_history"


class TestWrapReturnsProxy:
    def test_wrap_returns_wrapped_graph(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = LangGraphAdapter(engine)
        graph = MockCompiledGraph([], [])
        wrapped = adapter.wrap(graph)
        assert isinstance(wrapped, _WrappedGraph)

    def test_getattr_delegation(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = LangGraphAdapter(engine)
        graph = MockCompiledGraph([], [])
        wrapped = adapter.wrap(graph)
        assert wrapped.some_attr == "graph_attr"


class TestToAgentMessage:
    def test_basic_conversion(self) -> None:
        msg = MockBaseMessage(content="Hello", type="human", name="user_agent")
        result = LangGraphAdapter._to_agent_message(msg)
        assert result.content == "Hello"
        assert result.sender == "user_agent"

    def test_falls_back_to_type_when_no_name(self) -> None:
        msg = MockBaseMessage(content="Hello", type="ai")
        result = LangGraphAdapter._to_agent_message(msg)
        assert result.sender == "ai"

    def test_additional_kwargs_as_metadata(self) -> None:
        msg = MockBaseMessage(
            content="Hello",
            type="ai",
            additional_kwargs={"tool_calls": [{"name": "search"}]},
        )
        result = LangGraphAdapter._to_agent_message(msg)
        assert result.metadata == {"tool_calls": [{"name": "search"}]}

    def test_empty_additional_kwargs(self) -> None:
        msg = MockBaseMessage(content="Hello", type="ai")
        result = LangGraphAdapter._to_agent_message(msg)
        assert result.metadata == {}


class TestInvokePassthrough:
    def test_passthrough_no_violations(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = LangGraphAdapter(engine)
        input_msgs = [MockBaseMessage("Hi", type="human", name="user")]
        output_msgs = [MockBaseMessage("Hello!", type="ai", name="assistant")]
        graph = MockCompiledGraph(input_msgs, output_msgs)
        wrapped = adapter.wrap(graph)

        state = {"messages": input_msgs}
        result = wrapped.invoke(state)
        assert len(result["messages"]) == 2
        assert len(engine.violations) == 0

    def test_empty_messages(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = LangGraphAdapter(engine)
        graph = MockCompiledGraph([], [])
        wrapped = adapter.wrap(graph)

        state = {"messages": []}
        result = wrapped.invoke(state)
        assert result["messages"] == []

    def test_no_messages_key_in_state(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = LangGraphAdapter(engine)
        graph = MockCompiledGraph([], [])
        wrapped = adapter.wrap(graph)

        result = wrapped.invoke({})
        assert result.get("messages", []) == []


class TestInvokeWithViolations:
    def test_block_raises_enforcement_error(self) -> None:
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])
        adapter = LangGraphAdapter(engine)
        output_msgs = [MockBaseMessage("Bad output", type="ai", name="agent")]
        graph = MockCompiledGraph([], output_msgs)
        wrapped = adapter.wrap(graph)

        with pytest.raises(EnforcementError):
            wrapped.invoke({"messages": []})

    def test_warn_allows_through(self) -> None:
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        adapter = LangGraphAdapter(engine)
        output_msgs = [MockBaseMessage("Warn output", type="ai", name="agent")]
        graph = MockCompiledGraph([], output_msgs)
        wrapped = adapter.wrap(graph)

        result = wrapped.invoke({"messages": []})
        assert len(result["messages"]) == 1
        assert len(engine.violations) == 1

    def test_multiple_new_messages_checked(self) -> None:
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        adapter = LangGraphAdapter(engine)
        output_msgs = [
            MockBaseMessage("Msg 1", type="ai", name="a1"),
            MockBaseMessage("Msg 2", type="ai", name="a2"),
        ]
        graph = MockCompiledGraph([], output_msgs)
        wrapped = adapter.wrap(graph)

        wrapped.invoke({"messages": []})
        assert len(engine.violations) == 2


class TestCustomMessagesKey:
    def test_custom_key(self) -> None:
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        adapter = LangGraphAdapter(engine, messages_key="chat_history")

        class CustomGraph:
            some_attr = "val"

            def invoke(self, state: dict, **kwargs: Any) -> dict:
                result = dict(state)
                result["chat_history"] = state.get("chat_history", []) + [
                    MockBaseMessage("Output", type="ai", name="bot")
                ]
                return result

        wrapped = adapter.wrap(CustomGraph())
        result = wrapped.invoke({"chat_history": []})
        assert len(result["chat_history"]) == 1
        assert len(engine.violations) == 1


class TestAinvoke:
    def test_ainvoke_passthrough(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = LangGraphAdapter(engine)
        output_msgs = [MockBaseMessage("Async hello", type="ai", name="bot")]
        graph = MockCompiledGraph([], output_msgs)
        wrapped = adapter.wrap(graph)

        result = asyncio.get_event_loop().run_until_complete(
            wrapped.ainvoke({"messages": []})
        )
        assert len(result["messages"]) == 1
        assert len(engine.violations) == 0

    def test_ainvoke_block(self) -> None:
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])
        adapter = LangGraphAdapter(engine)
        output_msgs = [MockBaseMessage("Bad", type="ai", name="bot")]
        graph = MockCompiledGraph([], output_msgs)
        wrapped = adapter.wrap(graph)

        with pytest.raises(EnforcementError):
            asyncio.get_event_loop().run_until_complete(
                wrapped.ainvoke({"messages": []})
            )
