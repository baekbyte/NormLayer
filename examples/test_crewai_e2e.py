"""End-to-end integration test for CrewAI adapter with real Claude LLM calls.

Prerequisites:
    pip install normlayer[all] crewai crewai-tools
    export ANTHROPIC_API_KEY="your-key-here"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from crewai import Agent, Crew, Process, Task

from normlayer import EnforcementError, PolicyEngine, policies
from normlayer.adapters import CrewAIAdapter


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Create CrewAI agents using Claude
    researcher = Agent(
        role="research analyst",
        goal="Find relevant information about the given topic",
        backstory="You are an experienced research analyst who excels at finding "
        "and synthesizing information. Be concise in your findings.",
        llm="anthropic/claude-sonnet-4-20250514",
        max_iter=2,
        verbose=True,
    )

    writer = Agent(
        role="content writer",
        goal="Write a clear summary based on research findings",
        backstory="You are a skilled content writer who creates clear, concise "
        "summaries from research data.",
        llm="anthropic/claude-sonnet-4-20250514",
        max_iter=2,
        verbose=True,
    )

    # Define tasks
    research_task = Task(
        description="Research the key benefits of multi-agent AI systems. "
        "List 3 main benefits in 2-3 sentences each.",
        expected_output="A list of 3 key benefits with brief explanations.",
        agent=researcher,
    )

    writing_task = Task(
        description="Based on the research findings, write a concise 1-paragraph "
        "summary of multi-agent AI system benefits.",
        expected_output="A single paragraph summarizing the benefits.",
        agent=writer,
    )

    # Build the crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    # Build the policy engine
    engine = PolicyEngine(
        policies=[
            policies.NoUnsanctionedAction(
                permissions={
                    "research analyst": ["create", "update", "analyze", "search"],
                    "content writer": ["create", "publish", "write", "summarize"],
                },
                global_forbidden=["delete", "override"],
                handler="warn",
            ),
            policies.EscalateOnConflict(
                conflict_threshold=3,
                to="supervisor",
                handler="warn",
            ),
            policies.ResponseProportionality(max_ratio=15.0, handler="warn"),
        ],
    )

    # Wrap the crew
    adapter = CrewAIAdapter(engine)
    wrapped_crew = adapter.wrap(crew)

    print("=" * 60)
    print("CrewAI E2E Integration Test")
    print("=" * 60)

    try:
        result = wrapped_crew.kickoff()
    except EnforcementError as e:
        print(f"\nBLOCKED by {e.result.policy_name}: {e.result.details}")
        return

    # Print results
    print("\n--- Crew Output ---")
    print(f"Raw output: {str(result)[:500]}")

    if hasattr(result, "tasks_output"):
        print(f"\nTasks completed: {len(result.tasks_output)}")
        for i, task_output in enumerate(result.tasks_output):
            raw = getattr(task_output, "raw", str(task_output))
            print(f"\n  Task {i}: {raw[:300]}...")

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

    print("\nCrewAI E2E test completed successfully.")


if __name__ == "__main__":
    main()
