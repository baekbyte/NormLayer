"""Tests for AutoGenAdapter — all mocked, no AutoGen install needed."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from normlayer.adapters.autogen_adapter import AutoGenAdapter, _WrappedAgent
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
# Mock AutoGen objects
# ---------------------------------------------------------------------------


class MockTextMessage:
    """Mimics an AutoGen TextMessage."""

    def __init__(self, content: str, source: str) -> None:
        self.content = content
        self.source = source


class MockToolCallMessage:
    """Mimics an AutoGen ToolCallMessage (no content str)."""

    def __init__(self, source: str) -> None:
        self.content = {"tool": "search", "args": {}}  # dict, not str
        self.source = source


class MockResponse:
    """Mimics an AutoGen Response."""

    def __init__(
        self,
        chat_message: Any = None,
        inner_messages: list[Any] | None = None,
    ) -> None:
        self.chat_message = chat_message
        self.inner_messages = inner_messages or []


class MockAutoGenAgent:
    """Mimics an AutoGen agent with async on_messages."""

    def __init__(self, name: str, response: MockResponse) -> None:
        self.name = name
        self._response = response
        self.some_attr = "agent_attr"

    async def on_messages(
        self,
        messages: list[Any],
        cancellation_token: Any = None,
    ) -> MockResponse:
        return self._response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAutoGenAdapterInit:
    def test_stores_engine(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = AutoGenAdapter(engine)
        assert adapter.engine is engine


class TestWrapReturnsProxy:
    def test_wrap_returns_wrapped_agent(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = AutoGenAdapter(engine)
        response = MockResponse(chat_message=MockTextMessage("Hi", "bot"))
        agent = MockAutoGenAgent("bot", response)
        wrapped = adapter.wrap(agent)
        assert isinstance(wrapped, _WrappedAgent)

    def test_getattr_delegation(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = AutoGenAdapter(engine)
        response = MockResponse()
        agent = MockAutoGenAgent("bot", response)
        wrapped = adapter.wrap(agent)
        assert wrapped.some_attr == "agent_attr"
        assert wrapped.name == "bot"


class TestToAgentMessage:
    def test_text_message_conversion(self) -> None:
        msg = MockTextMessage(content="Hello", source="agent_1")
        result = AutoGenAdapter._to_agent_message(msg)
        assert result is not None
        assert result.content == "Hello"
        assert result.sender == "agent_1"

    def test_unsupported_message_returns_none(self) -> None:
        msg = MockToolCallMessage(source="agent_1")
        result = AutoGenAdapter._to_agent_message(msg)
        assert result is None

    def test_no_source_returns_none(self) -> None:
        class NoSourceMsg:
            content = "Hello"

        result = AutoGenAdapter._to_agent_message(NoSourceMsg())
        assert result is None

    def test_no_content_returns_none(self) -> None:
        class NoContentMsg:
            source = "agent_1"

        result = AutoGenAdapter._to_agent_message(NoContentMsg())
        assert result is None


class TestOnMessagesPassthrough:
    def test_passthrough_no_violations(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = AutoGenAdapter(engine)
        chat_msg = MockTextMessage("Response", "bot")
        response = MockResponse(chat_message=chat_msg)
        agent = MockAutoGenAgent("bot", response)
        wrapped = adapter.wrap(agent)

        incoming = [MockTextMessage("Question", "user")]
        result = asyncio.get_event_loop().run_until_complete(
            wrapped.on_messages(incoming)
        )
        assert result is response
        assert len(engine.violations) == 0

    def test_no_chat_message_in_response(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = AutoGenAdapter(engine)
        response = MockResponse(chat_message=None)
        agent = MockAutoGenAgent("bot", response)
        wrapped = adapter.wrap(agent)

        result = asyncio.get_event_loop().run_until_complete(
            wrapped.on_messages([])
        )
        assert result is response

    def test_empty_incoming_messages(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = AutoGenAdapter(engine)
        chat_msg = MockTextMessage("Hi", "bot")
        response = MockResponse(chat_message=chat_msg)
        agent = MockAutoGenAgent("bot", response)
        wrapped = adapter.wrap(agent)

        result = asyncio.get_event_loop().run_until_complete(
            wrapped.on_messages([])
        )
        assert result is response


class TestIncomingViolations:
    def test_incoming_block_raises(self) -> None:
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])
        adapter = AutoGenAdapter(engine)
        response = MockResponse(chat_message=MockTextMessage("Ok", "bot"))
        agent = MockAutoGenAgent("bot", response)
        wrapped = adapter.wrap(agent)

        incoming = [MockTextMessage("Bad input", "user")]
        with pytest.raises(EnforcementError):
            asyncio.get_event_loop().run_until_complete(
                wrapped.on_messages(incoming)
            )

    def test_incoming_warn_allows_through(self) -> None:
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        adapter = AutoGenAdapter(engine)
        chat_msg = MockTextMessage("Response", "bot")
        response = MockResponse(chat_message=chat_msg)
        agent = MockAutoGenAgent("bot", response)
        wrapped = adapter.wrap(agent)

        incoming = [MockTextMessage("Warn input", "user")]
        result = asyncio.get_event_loop().run_until_complete(
            wrapped.on_messages(incoming)
        )
        assert result is response
        # 1 incoming warn + 1 outgoing warn = 2
        assert len(engine.violations) == 2


class TestOutgoingViolations:
    def test_outgoing_block_raises(self) -> None:
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])
        adapter = AutoGenAdapter(engine)
        chat_msg = MockTextMessage("Bad output", "bot")
        response = MockResponse(chat_message=chat_msg)
        agent = MockAutoGenAgent("bot", response)
        wrapped = adapter.wrap(agent)

        # Use unsupported message type to skip incoming check
        incoming = [MockToolCallMessage(source="user")]
        with pytest.raises(EnforcementError):
            asyncio.get_event_loop().run_until_complete(
                wrapped.on_messages(incoming)
            )


class TestUnsupportedMessageSkipped:
    def test_tool_call_message_skipped(self) -> None:
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])
        adapter = AutoGenAdapter(engine)
        # Response has no chat_message, so outgoing check also skipped
        response = MockResponse(chat_message=None)
        agent = MockAutoGenAgent("bot", response)
        wrapped = adapter.wrap(agent)

        incoming = [MockToolCallMessage(source="user")]
        result = asyncio.get_event_loop().run_until_complete(
            wrapped.on_messages(incoming)
        )
        assert result is response
        assert len(engine.violations) == 0


class TestCancellationTokenForwarded:
    def test_cancellation_token_passed(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = AutoGenAdapter(engine)

        received_token = None

        class TokenCapturingAgent:
            name = "bot"
            some_attr = "val"

            async def on_messages(self, messages, cancellation_token=None):
                nonlocal received_token
                received_token = cancellation_token
                return MockResponse(chat_message=None)

        wrapped = adapter.wrap(TokenCapturingAgent())
        token = object()
        asyncio.get_event_loop().run_until_complete(
            wrapped.on_messages([], cancellation_token=token)
        )
        assert received_token is token
