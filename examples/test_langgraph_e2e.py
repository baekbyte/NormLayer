"""End-to-end integration test for LangGraph adapter with real Claude LLM calls.

Prerequisites:
    pip install normlayer[all] langgraph langchain-anthropic
    export ANTHROPIC_API_KEY="your-key-here"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, TypedDict

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from normlayer import AgentMessage, EnforcementError, PolicyEngine, policies
from normlayer.adapters import LangGraphAdapter


class GraphState(TypedDict):
    messages: list[Any]


def make_planner_node(llm: ChatAnthropic):
    """Create a planner node that generates a plan using Claude."""

    def planner(state: GraphState) -> GraphState:
        system = SystemMessage(content="You are a planner agent. Create a brief plan.")
        response = llm.invoke([system, *state["messages"]])
        response.name = "planner_agent"
        return {"messages": state["messages"] + [response]}

    return planner


def make_executor_node(llm: ChatAnthropic):
    """Create an executor node that summarizes/executes the plan."""

    def executor(state: GraphState) -> GraphState:
        system = SystemMessage(
            content="You are an executor agent. Summarize the plan you received "
            "and describe how you would execute it. Be concise."
        )
        response = llm.invoke([system, *state["messages"]])
        response.name = "executor_agent"
        return {"messages": state["messages"] + [response]}

    return executor


def build_graph(llm: ChatAnthropic) -> Any:
    """Build and compile a 2-node planner→executor graph."""
    graph = StateGraph(GraphState)
    graph.add_node("planner", make_planner_node(llm))
    graph.add_node("executor", make_executor_node(llm))
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", END)
    return graph.compile()


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=512)

    # Build the policy engine
    engine = PolicyEngine(
        policies=[
            policies.RoleRespect(
                role_definitions={
                    "planner": ["plan", "assign", "schedule", "organize"],
                    "executor": ["execute", "run", "complete", "implement"],
                },
                agent_roles={
                    "planner_agent": "planner",
                    "executor_agent": "executor",
                },
                strict=True,
                handler="warn",
            ),
            policies.LoopDetection(max_repetitions=2, handler="warn"),
            policies.ResponseProportionality(max_ratio=10.0, handler="warn"),
        ],
    )

    # Wrap the graph
    adapter = LangGraphAdapter(engine, messages_key="messages")
    compiled_graph = build_graph(llm)
    wrapped_graph = adapter.wrap(compiled_graph)

    print("=" * 60)
    print("LangGraph E2E Integration Test")
    print("=" * 60)

    # Invoke with a real prompt
    initial_state: GraphState = {
        "messages": [HumanMessage(content="Plan how to organize a team meeting")],
    }

    try:
        result = wrapped_graph.invoke(initial_state)
    except EnforcementError as e:
        print(f"\nBLOCKED by {e.result.policy_name}: {e.result.details}")
        return

    # Print messages
    print("\n--- Messages ---")
    for msg in result["messages"]:
        name = getattr(msg, "name", None) or msg.__class__.__name__
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        print(f"\n[{name}]: {content[:300]}...")

    # Print violations
    print("\n--- Violations ---")
    violations = engine.violations
    if not violations:
        print("No violations detected.")
    else:
        for v in violations:
            print(
                f"  Policy: {v.policy_violated}, Agent: {v.agent_id}, "
                f"Severity: {v.severity}, Handler: {v.handler_dispatched}"
            )
            print(f"  Details: {v.details[:200]}")

    print("\nLangGraph E2E test completed successfully.")


if __name__ == "__main__":
    main()
