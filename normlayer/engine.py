"""PolicyEngine — the runtime enforcement layer for NormLayer."""

from __future__ import annotations

import functools
from typing import Any, Callable

from normlayer.base_policy import (
    AgentMessage,
    BasePolicy,
    PolicyResult,
    ViolationEvent,
)


class EnforcementError(Exception):
    """Raised when a policy with handler='block' detects a violation.

    Attributes:
        result: The PolicyResult that triggered the block.
        event: The structured ViolationEvent that was logged.
    """

    def __init__(self, result: PolicyResult, event: ViolationEvent) -> None:
        self.result = result
        self.event = event
        super().__init__(
            f"[NormLayer BLOCK] Policy '{result.policy_name}' blocked message "
            f"from '{result.agent_id}': {result.details}"
        )


class PolicyEngine:
    """Runtime enforcement engine that applies a stack of policies to agent messages.

    Intercepts agent messages, runs them through every registered policy, and
    dispatches the appropriate handler when a violation is detected:

    - ``block``    — raises :class:`EnforcementError`, stopping the message.
    - ``warn``     — prints a warning and allows the message through.
    - ``escalate`` — calls `supervisor_agent` with the :class:`ViolationEvent`
                     (falls back to a warning print if no supervisor is set).
    - ``log``      — records silently; no user-visible side effect.

    Violations are always appended to `self.violations` regardless of handler.
    If `aws_bucket` is provided, violations are also shipped to S3 via boto3.

    Args:
        policies: Ordered list of :class:`BasePolicy` instances to enforce.
        aws_bucket: S3 bucket name for violation logging. Optional during local dev.
        aws_region: AWS region string (e.g. ``"us-east-1"``). Required when
            `aws_bucket` is set.
        supervisor_agent: Callable invoked for ``"escalate"`` violations.
            Receives a single :class:`ViolationEvent` argument.
        violation_logger: An explicit :class:`ViolationLogger` instance. If
            provided, it takes precedence over ``aws_bucket``/``aws_region``.
            If not provided but ``aws_bucket`` is set, a logger is
            auto-constructed.
    """

    def __init__(
        self,
        policies: list[BasePolicy],
        aws_bucket: str | None = None,
        aws_region: str | None = None,
        supervisor_agent: Callable[[ViolationEvent], Any] | None = None,
        violation_logger: Any | None = None,
    ) -> None:
        self.policies = policies
        self.aws_bucket = aws_bucket
        self.aws_region = aws_region
        self.supervisor_agent = supervisor_agent
        self._violation_log: list[ViolationEvent] = []

        # Resolve logger: explicit > auto-construct from bucket > None
        if violation_logger is not None:
            self._logger = violation_logger
        elif aws_bucket is not None:
            from normlayer.logging.violation_logger import ViolationLogger

            self._logger = ViolationLogger(
                bucket=aws_bucket,
                region=aws_region or "us-east-1",
            )
        else:
            self._logger = None

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    def check(
        self,
        message: AgentMessage,
        context: dict | None = None,
    ) -> list[PolicyResult]:
        """Run all policies against a message synchronously.

        Policies are evaluated in registration order. The first ``"block"``
        violation raises immediately; subsequent policies are not evaluated.

        Args:
            message: The :class:`AgentMessage` to evaluate.
            context: Optional dict with conversation history, role definitions,
                and any other policy-specific keys.

        Returns:
            List of :class:`PolicyResult`, one per registered policy.

        Raises:
            EnforcementError: If any policy fires with ``handler="block"``.
        """
        context = context or {}
        results: list[PolicyResult] = []
        for policy in self.policies:
            result = policy.evaluate(message, context)
            results.append(result)
            if not result.passed:
                self._dispatch(result, message, context)
        return results

    async def async_check(
        self,
        message: AgentMessage,
        context: dict | None = None,
    ) -> list[PolicyResult]:
        """Run all policies against a message asynchronously.

        Policies are awaited in registration order. Each policy's
        :meth:`BasePolicy.async_evaluate` is called; the default
        implementation falls back to synchronous evaluation.

        Args:
            message: The :class:`AgentMessage` to evaluate.
            context: Optional context dict.

        Returns:
            List of :class:`PolicyResult`, one per registered policy.

        Raises:
            EnforcementError: If any policy fires with ``handler="block"``.
        """
        context = context or {}
        results: list[PolicyResult] = []
        for policy in self.policies:
            result = await policy.async_evaluate(message, context)
            results.append(result)
            if not result.passed:
                self._dispatch(result, message, context)
        return results

    # ------------------------------------------------------------------
    # Decorator / wrap API
    # ------------------------------------------------------------------

    def enforce(self, func: Callable) -> Callable:
        """Decorator that wraps a synchronous agent function with policy enforcement.

        The decorated function must accept ``(message: AgentMessage, context: dict)``
        as its first two positional arguments. Policy checks run before the
        function body executes.

        Args:
            func: The sync agent callable to wrap.

        Returns:
            A wrapped callable with the same signature.

        Example::

            @engine.enforce
            def planner(message: AgentMessage, context: dict) -> AgentMessage:
                ...
        """
        @functools.wraps(func)
        def wrapper(
            message: AgentMessage,
            context: dict | None = None,
            **kwargs: Any,
        ) -> Any:
            self.check(message, context)
            return func(message, context, **kwargs)

        return wrapper

    def async_enforce(self, func: Callable) -> Callable:
        """Decorator that wraps an async agent function with policy enforcement.

        Args:
            func: The async agent callable to wrap.

        Returns:
            A wrapped async callable with the same signature.

        Example::

            @engine.async_enforce
            async def planner(message: AgentMessage, context: dict) -> AgentMessage:
                ...
        """
        @functools.wraps(func)
        async def wrapper(
            message: AgentMessage,
            context: dict | None = None,
            **kwargs: Any,
        ) -> Any:
            await self.async_check(message, context)
            return await func(message, context, **kwargs)

        return wrapper

    def wrap(self, agent: Callable) -> Callable:
        """Wrap an existing sync agent callable with policy enforcement.

        Alias for :meth:`enforce` — convenient when wrapping agents you
        didn't define yourself.

        Args:
            agent: Any sync callable representing an agent.

        Returns:
            A wrapped callable.

        Example::

            safe_agent = engine.wrap(existing_agent)
        """
        return self.enforce(agent)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def violations(self) -> list[ViolationEvent]:
        """All violation events recorded in this engine's lifetime.

        Returns a copy so callers cannot mutate internal state.
        """
        return list(self._violation_log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        result: PolicyResult,
        message: AgentMessage,
        context: dict,
    ) -> None:
        """Record a violation and execute the appropriate handler.

        Args:
            result: The failing PolicyResult.
            message: The message that triggered the violation.
            context: The context dict at evaluation time.

        Raises:
            EnforcementError: When ``result.handler == "block"``.
        """
        event = ViolationEvent.from_policy_result(result, message, context)
        self._violation_log.append(event)
        self._ship_to_s3(event)

        if result.handler == "block":
            raise EnforcementError(result, event)
        elif result.handler == "warn":
            print(
                f"[NormLayer WARN] {result.policy_name} violated by "
                f"'{result.agent_id}': {result.details}"
            )
        elif result.handler == "escalate":
            if self.supervisor_agent is not None:
                self.supervisor_agent(event)
            else:
                print(
                    f"[NormLayer ESCALATE] {result.policy_name} violated by "
                    f"'{result.agent_id}' — no supervisor_agent configured."
                )
        elif result.handler == "log":
            pass  # Already appended to _violation_log and shipped to S3.

    def _ship_to_s3(self, event: ViolationEvent) -> None:
        """Ship a violation event to S3 via the configured ViolationLogger.

        No-op when no logger is configured. Failures are printed but
        never re-raised — a logging error must never interrupt the pipeline.

        Args:
            event: The :class:`ViolationEvent` to persist.
        """
        if self._logger is None:
            return
        try:
            self._logger.ship(event)
        except Exception as exc:
            print(f"[NormLayer] S3 logging failed: {exc}")

    def flush_violations(self) -> int:
        """Flush any buffered violation events to S3.

        Returns:
            Number of events flushed. Returns 0 if no logger is configured.
        """
        if self._logger is None:
            return 0
        return self._logger.flush()
