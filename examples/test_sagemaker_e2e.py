"""End-to-end integration test for SageMaker audit job.

Prerequisites:
    pip install normlayer[aws]
    export AWS_DEFAULT_REGION="us-east-1"
    export NORMLAYER_S3_BUCKET="your-bucket-name"
    export SAGEMAKER_ROLE_ARN="arn:aws:iam::ACCOUNT:role/NormLayerSageMakerRole"

    Run test_aws_e2e.py first to populate S3 with violations.

Cost note: This launches a real SageMaker Processing Job on ml.t3.medium (~$0.05).
    The job will likely fail quickly (no audit script), but we're testing the API call.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from normlayer.logging import SageMakerAuditJob


def main() -> None:
    bucket = os.environ.get("NORMLAYER_S3_BUCKET")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")

    if not bucket or not role_arn:
        print(
            "ERROR: Set these environment variables:\n"
            "  NORMLAYER_S3_BUCKET — your S3 bucket with violations\n"
            "  SAGEMAKER_ROLE_ARN — your SageMaker execution role ARN\n"
            "\n"
            "Example:\n"
            "  export NORMLAYER_S3_BUCKET=normlayer-violations-myname\n"
            '  export SAGEMAKER_ROLE_ARN="arn:aws:iam::123456789:role/NormLayerSageMakerRole"'
        )
        sys.exit(1)

    input_s3_uri = f"s3://{bucket}/violations/"
    output_s3_uri = f"s3://{bucket}/audit-results/"

    print("=" * 60)
    print("SageMaker Audit Job E2E Integration Test")
    print("=" * 60)
    print(f"Bucket:    {bucket}")
    print(f"Region:    {region}")
    print(f"Role ARN:  {role_arn}")
    print(f"Input:     {input_s3_uri}")
    print(f"Output:    {output_s3_uri}")

    # Create the audit job
    job = SageMakerAuditJob(
        role_arn=role_arn,
        input_s3_uri=input_s3_uri,
        output_s3_uri=output_s3_uri,
        instance_type="ml.t3.medium",
        region=region,
        max_runtime_seconds=600,
    )

    # Launch the job
    print("\n--- Launching SageMaker Processing Job ---")
    try:
        job_name = job.run()
    except Exception as e:
        print(f"  ERROR launching job: {e}")
        print("  This may indicate missing IAM permissions or invalid role ARN.")
        sys.exit(1)

    print(f"  Job name: {job_name}")

    # Poll status a few times
    print("\n--- Polling job status ---")
    max_polls = 6
    poll_interval = 10

    for i in range(max_polls):
        try:
            status = job.status(job_name)
        except Exception as e:
            print(f"  Poll {i + 1}: ERROR — {e}")
            break

        print(f"  Poll {i + 1}: {status}")

        if status in ("Completed", "Failed", "Stopped"):
            break

        if i < max_polls - 1:
            print(f"  Waiting {poll_interval}s...")
            time.sleep(poll_interval)

    print(f"\nSageMaker E2E test completed. Final status: {status}")
    print(
        "Note: 'Failed' is expected if no audit script was provided — "
        "the goal is to verify the API call succeeds."
    )


if __name__ == "__main__":
    main()
