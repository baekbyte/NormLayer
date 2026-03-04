"""End-to-end integration test for AWS S3 violation logging.

Prerequisites:
    pip install normlayer[aws]
    export AWS_DEFAULT_REGION="us-east-1"
    # AWS credentials configured via ~/.aws/credentials, env vars, or IAM role

    # Create a test bucket:
    aws s3 mb s3://normlayer-violations-$(whoami) --region us-east-1
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

from normlayer import AgentMessage, EnforcementError, PolicyEngine, policies
from normlayer.logging import ViolationLogger


def main() -> None:
    bucket = os.environ.get("NORMLAYER_S3_BUCKET")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    if not bucket:
        print(
            "ERROR: Set NORMLAYER_S3_BUCKET environment variable\n"
            "  Example: export NORMLAYER_S3_BUCKET=normlayer-violations-myname"
        )
        sys.exit(1)

    print("=" * 60)
    print("AWS S3 Violation Logging E2E Integration Test")
    print("=" * 60)
    print(f"Bucket: {bucket}")
    print(f"Region: {region}")

    # Create a violation logger
    logger = ViolationLogger(
        bucket=bucket,
        region=region,
        prefix="violations/",
        batch_size=10,
    )

    # Create engine with a strict policy that will trigger violations
    engine = PolicyEngine(
        policies=[
            policies.RoleRespect(
                role_definitions={
                    "reader": ["read", "view", "list"],
                },
                agent_roles={"deployer_agent": "reader"},
                strict=True,
                handler="warn",
            ),
            policies.NoUnsanctionedAction(
                permissions={
                    "deployer_agent": ["read"],
                },
                global_forbidden=["delete"],
                handler="warn",
            ),
        ],
        violation_logger=logger,
    )

    # --- Test 1: Send a message that should trigger RoleRespect ---
    print("\n--- Test 1: RoleRespect violation ---")
    msg1 = AgentMessage(
        content="I will deploy the application to production and restart all services.",
        sender="deployer_agent",
        recipient="supervisor_agent",
    )

    try:
        results = engine.check(msg1)
    except EnforcementError as e:
        print(f"  Blocked: {e.result.policy_name}")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.policy_name}: score={r.violation_score:.3f}, {r.details[:100]}")

    # --- Test 2: Send a message with a globally forbidden action ---
    print("\n--- Test 2: NoUnsanctionedAction violation ---")
    msg2 = AgentMessage(
        content="I need to delete all old records from the database.",
        sender="deployer_agent",
        recipient="db_agent",
    )

    try:
        results = engine.check(msg2)
    except EnforcementError as e:
        print(f"  Blocked: {e.result.policy_name}")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.policy_name}: score={r.violation_score:.3f}, {r.details[:100]}")

    # --- Flush violations to S3 ---
    print("\n--- Flushing violations to S3 ---")
    flushed = engine.flush_violations()
    print(f"  Flushed {flushed} violation(s) to S3")

    # --- Fetch violations back from S3 ---
    print("\n--- Fetching violations from S3 ---")
    all_events = logger.fetch_all()
    print(f"  Total events in S3: {len(all_events)}")

    deployer_events = logger.fetch_all(agent_id="deployer_agent")
    print(f"  Events for deployer_agent: {len(deployer_events)}")

    for event in deployer_events:
        print(
            f"  - {event.timestamp}: policy={event.policy_violated}, "
            f"severity={event.severity}, handler={event.handler_dispatched}"
        )
        print(f"    snippet: {event.message_snippet[:100]}")

    print(f"\nAWS S3 E2E test completed. Bucket: s3://{bucket}/violations/")


if __name__ == "__main__":
    main()
