"""Master integration test runner.

Runs all E2E integration tests in the recommended order.
Set SKIP_AWS=1 to skip AWS tests, SKIP_SAGEMAKER=1 to skip only SageMaker.

Prerequisites:
    pip install normlayer[all] langgraph langchain-anthropic crewai crewai-tools pyautogen
    export ANTHROPIC_API_KEY="your-key-here"
    export AWS_DEFAULT_REGION="us-east-1"
    export NORMLAYER_S3_BUCKET="your-bucket-name"        # for AWS tests
    export SAGEMAKER_ROLE_ARN="arn:aws:iam::...:role/..."  # for SageMaker test
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv is optional

EXAMPLES_DIR = Path(__file__).parent

TESTS = [
    ("LangGraph Adapter", "test_langgraph_e2e.py", False),
    ("CrewAI Adapter", "test_crewai_e2e.py", False),
    ("AutoGen Adapter", "test_autogen_e2e.py", False),
    ("AWS S3 Logging", "test_aws_e2e.py", True),
    ("SageMaker Audit Job", "test_sagemaker_e2e.py", True),
]


def main() -> None:
    skip_aws = os.environ.get("SKIP_AWS", "0") == "1"
    skip_sagemaker = os.environ.get("SKIP_SAGEMAKER", "0") == "1"

    print("=" * 60)
    print("NormLayer End-to-End Integration Tests")
    print("=" * 60)

    results: list[tuple[str, str]] = []

    for name, script, is_aws in TESTS:
        if is_aws and skip_aws:
            print(f"\n>>> SKIPPING {name} (SKIP_AWS=1)")
            results.append((name, "SKIPPED"))
            continue
        if script == "test_sagemaker_e2e.py" and skip_sagemaker:
            print(f"\n>>> SKIPPING {name} (SKIP_SAGEMAKER=1)")
            results.append((name, "SKIPPED"))
            continue

        print(f"\n{'=' * 60}")
        print(f">>> Running: {name}")
        print("=" * 60)

        script_path = EXAMPLES_DIR / script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(EXAMPLES_DIR.parent),
        )

        if result.returncode == 0:
            results.append((name, "PASSED"))
        else:
            results.append((name, f"FAILED (exit {result.returncode})"))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results:
        icon = "+" if status == "PASSED" else ("-" if "SKIPPED" in status else "X")
        print(f"  [{icon}] {name}: {status}")

    failed = sum(1 for _, s in results if s.startswith("FAILED"))
    if failed:
        print(f"\n{failed} test(s) failed.")
        sys.exit(1)
    else:
        print("\nAll tests passed (or skipped).")


if __name__ == "__main__":
    main()
