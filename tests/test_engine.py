"""Tests for PolicyEngine — check, async_check, handlers, enforce/wrap decorators."""

import pytest

from normlayer.base_policy import AgentMessage, BasePolicy, PolicyResult
from normlayer.engine import EnforcementError, PolicyEngine
from normlayer.testing import MockMessage

# ---------------------------------------------------------------------------
# Minimal concrete policy stubs for engine tests
# ---------------------------------------------------------------------------


class AlwaysPassPolicy(BasePolicy):
    """Policy that always passes — used to verify engine plumbing."""

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
    """Policy that always fires with handler='block'."""

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
    """Policy that always fires with handler='warn'."""

    name = "AlwaysWarn"

    def __init__(self) -> None:
        super().__init__(handler="warn")

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        return PolicyResult(
            passed=False,
            violation_score=0.5,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="medium",
            details="Always warns.",
        )


class AlwaysEscalatePolicy(BasePolicy):
    """Policy that always fires with handler='escalate'."""

    name = "AlwaysEscalate"

    def __init__(self) -> None:
        super().__init__(handler="escalate")

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        return PolicyResult(
            passed=False,
            violation_score=0.7,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="high",
            details="Always escalates.",
        )


class AlwaysLogPolicy(BasePolicy):
    """Policy that always fires with handler='log'."""

    name = "AlwaysLog"

    def __init__(self) -> None:
        super().__init__(handler="log")

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        return PolicyResult(
            passed=False,
            violation_score=0.3,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="low",
            details="Always logs.",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(content: str = "hello", sender: str = "agent_a") -> AgentMessage:
    return MockMessage(content=content, sender=sender).to_agent_message()


# ---------------------------------------------------------------------------
# check() — basic result shape
# ---------------------------------------------------------------------------


class TestEngineCheck:
    def test_returns_one_result_per_policy(self):
        engine = PolicyEngine(policies=[AlwaysPassPolicy(), AlwaysPassPolicy()])
        results = engine.check(_msg())
        assert len(results) == 2

    def test_all_pass_when_no_violations(self):
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        results = engine.check(_msg())
        assert all(r.passed for r in results)

    def test_empty_context_defaults_to_dict(self):
        """check() with no context arg should not raise."""
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        results = engine.check(_msg())
        assert results[0].passed is True

    def test_explicit_context_is_passed_through(self):
        """check() with an explicit context dict should not raise."""
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        results = engine.check(_msg(), context={"history": []})
        assert len(results) == 1


# ---------------------------------------------------------------------------
# check() — handler dispatch
# ---------------------------------------------------------------------------


class TestEngineHandlers:
    def test_block_raises_enforcement_error(self):
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])
        with pytest.raises(EnforcementError) as exc_info:
            engine.check(_msg())
        err = exc_info.value
        assert err.result.policy_name == "AlwaysBlock"
        assert err.event.handler_dispatched == "block"

    def test_warn_allows_message_through(self, capsys):
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        results = engine.check(_msg())
        assert len(results) == 1
        assert results[0].passed is False
        captured = capsys.readouterr()
        assert "WARN" in captured.out

    def test_log_allows_message_through_silently(self, capsys):
        engine = PolicyEngine(policies=[AlwaysLogPolicy()])
        results = engine.check(_msg())
        assert results[0].passed is False
        captured = capsys.readouterr()
        assert captured.out == ""  # log handler is silent

    def test_escalate_calls_supervisor(self):
        escalated = []
        engine = PolicyEngine(
            policies=[AlwaysEscalatePolicy()],
            supervisor_agent=lambda event: escalated.append(event),
        )
        engine.check(_msg())
        assert len(escalated) == 1
        assert escalated[0].policy_violated == "AlwaysEscalate"

    def test_escalate_without_supervisor_prints_warning(self, capsys):
        engine = PolicyEngine(policies=[AlwaysEscalatePolicy()])
        engine.check(_msg())
        captured = capsys.readouterr()
        assert "ESCALATE" in captured.out


# ---------------------------------------------------------------------------
# violations log
# ---------------------------------------------------------------------------


class TestEngineViolationsLog:
    def test_violation_appended_on_block(self):
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])
        try:
            engine.check(_msg())
        except EnforcementError:
            pass
        assert len(engine.violations) == 1

    def test_violation_appended_on_warn(self):
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        engine.check(_msg())
        assert len(engine.violations) == 1

    def test_violation_appended_on_log(self):
        engine = PolicyEngine(policies=[AlwaysLogPolicy()])
        engine.check(_msg())
        assert len(engine.violations) == 1

    def test_violations_is_a_copy(self):
        """Mutating the returned list should not affect internal state."""
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        engine.check(_msg())
        copy = engine.violations
        copy.clear()
        assert len(engine.violations) == 1

    def test_multiple_violations_accumulated(self):
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        engine.check(_msg())
        engine.check(_msg())
        assert len(engine.violations) == 2

    def test_no_violations_when_all_pass(self):
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        engine.check(_msg())
        assert len(engine.violations) == 0

    def test_violation_event_fields(self):
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        engine.check(_msg(sender="test_sender"))
        event = engine.violations[0]
        assert event.agent_id == "test_sender"
        assert event.policy_violated == "AlwaysWarn"
        assert event.handler_dispatched == "warn"


# ---------------------------------------------------------------------------
# async_check()
# ---------------------------------------------------------------------------


class TestEngineAsyncCheck:
    @pytest.mark.asyncio
    async def test_async_check_returns_results(self):
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        results = await engine.async_check(_msg())
        assert len(results) == 1
        assert results[0].passed is True

    @pytest.mark.asyncio
    async def test_async_check_raises_on_block(self):
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])
        with pytest.raises(EnforcementError):
            await engine.async_check(_msg())

    @pytest.mark.asyncio
    async def test_async_check_accumulates_violations(self):
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        await engine.async_check(_msg())
        assert len(engine.violations) == 1


# ---------------------------------------------------------------------------
# enforce / wrap decorators
# ---------------------------------------------------------------------------


class TestEngineEnforce:
    def test_enforce_allows_passing_message(self):
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])

        @engine.enforce
        def agent(message: AgentMessage, context: dict | None = None) -> str:
            return "response"

        result = agent(_msg())
        assert result == "response"

    def test_enforce_blocks_violating_message(self):
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])

        @engine.enforce
        def agent(message: AgentMessage, context: dict | None = None) -> str:
            return "response"  # should never reach here

        with pytest.raises(EnforcementError):
            agent(_msg())

    def test_wrap_is_alias_for_enforce(self):
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])

        def raw_agent(message: AgentMessage, context: dict | None = None) -> str:
            return "ok"

        safe_agent = engine.wrap(raw_agent)
        assert safe_agent(_msg()) == "ok"

    @pytest.mark.asyncio
    async def test_async_enforce_allows_passing_message(self):
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])

        @engine.async_enforce
        async def agent(message: AgentMessage, context: dict | None = None) -> str:
            return "async response"

        result = await agent(_msg())
        assert result == "async response"

    @pytest.mark.asyncio
    async def test_async_enforce_blocks_violating_message(self):
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])

        @engine.async_enforce
        async def agent(message: AgentMessage, context: dict | None = None) -> str:
            return "should not get here"

        with pytest.raises(EnforcementError):
            await agent(_msg())
