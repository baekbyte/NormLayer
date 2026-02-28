"""NormLayer — runtime behavioral policy enforcement for multi-agent pipelines."""

from normlayer.base_policy import (
    AgentMessage,
    BasePolicy,
    HandlerType,
    PolicyResult,
    SeverityLevel,
    ViolationEvent,
)
from normlayer.engine import EnforcementError, PolicyEngine
from normlayer import policies

__all__ = [
    # Core abstractions
    "AgentMessage",
    "BasePolicy",
    "HandlerType",
    "PolicyResult",
    "SeverityLevel",
    "ViolationEvent",
    # Engine
    "PolicyEngine",
    "EnforcementError",
    # Policy namespace
    "policies",
]
