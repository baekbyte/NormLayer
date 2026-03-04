"""End-to-end integration test for AutoGen adapter with real Claude LLM calls.

Prerequisites:
    pip install normlayer[all] pyautogen
    export ANTHROPIC_API_KEY="your-key-here"

Note: AutoGen (ag2) uses async-only APIs for the new agent interface.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

from normlayer import EnforcementError, PolicyEngine, policies
from normlayer.adapters import AutoGenAdapter


async def run_test() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Create AutoGen model client for Claude
    model_client = AnthropicChatCompletionClient(
        model="claude-sonnet-4-20250514",
        api_key=api_key,
        max_tokens=512,
    )

    # Create an AutoGen AssistantAgent
    agent = AssistantAgent(
        name="assistant_agent",
        model_client=model_client,
        system_message="You are a helpful assistant. Answer questions concisely.",
    )

    # Build the policy engine
    engine = PolicyEngine(
        policies=[
            policies.RoleRespect(
                role_definitions={
                    "assistant": ["answer", "help", "explain", "summarize"],
                },
                agent_roles={"assistant_agent": "assistant"},
                handler="warn",
            ),
            policies.ResponseProportionality(max_ratio=10.0, handler="warn"),
        ],
    )

    # Wrap the agent
    adapter = AutoGenAdapter(engine)
    wrapped_agent = adapter.wrap(agent)

    print("=" * 60)
    print("AutoGen E2E Integration Test")
    print("=" * 60)

    # Send a message
    message = TextMessage(
        content="What are 3 key benefits of multi-agent AI systems?",
        source="user",
    )

    try:
        response = await wrapped_agent.on_messages(
            [message],
            cancellation_token=None,
        )
    except EnforcementError as e:
        print(f"\nBLOCKED by {e.result.policy_name}: {e.result.details}")
        return

    # Print response
    print("\n--- Response ---")
    if response.chat_message:
        content = response.chat_message.content
        if isinstance(content, str):
            print(f"[{response.chat_message.source}]: {content[:500]}")
        else:
            print(f"[{response.chat_message.source}]: {content}")
    else:
        print("No chat message in response.")

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

    print("\nAutoGen E2E test completed successfully.")


def main() -> None:
    asyncio.run(run_test())


if __name__ == "__main__":
    main()
