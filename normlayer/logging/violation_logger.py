"""ViolationLogger — S3 shipping via boto3. Stub, planned for Week 3-4."""

from __future__ import annotations

import json

from normlayer.base_policy import ViolationEvent


class ViolationLogger:
    """Ships violation events to AWS S3 and supports batch retrieval.

    Violations are written as individual JSON objects under a structured
    S3 key prefix: ``violations/<timestamp>/<agent_id>/<policy_name>.json``.

    Batch audits are run as SageMaker Processing Jobs that consume the S3
    log prefix and produce aggregated reports.

    .. note::
        This class is a stub. Full implementation is planned for Week 3-4.

    Args:
        bucket: S3 bucket name.
        region: AWS region string (e.g. ``"us-east-1"``).
        prefix: Key prefix for all violation objects (default ``"violations/"``).
    """

    def __init__(
        self,
        bucket: str,
        region: str,
        prefix: str = "violations/",
    ) -> None:
        self.bucket = bucket
        self.region = region
        self.prefix = prefix

    def ship(self, event: ViolationEvent) -> None:
        """Ship a single ViolationEvent to S3.

        Args:
            event: The violation to persist.

        Raises:
            NotImplementedError: Full implementation planned for Week 3-4.
        """
        raise NotImplementedError(
            "ViolationLogger.ship will be implemented in Week 3-4."
        )

    def fetch_all(self, agent_id: str | None = None) -> list[ViolationEvent]:
        """Fetch stored violation events from S3, optionally filtered by agent.

        Args:
            agent_id: If provided, only return events for this agent.

        Returns:
            List of ViolationEvent objects.

        Raises:
            NotImplementedError: Full implementation planned for Week 3-4.
        """
        raise NotImplementedError(
            "ViolationLogger.fetch_all will be implemented in Week 3-4."
        )
