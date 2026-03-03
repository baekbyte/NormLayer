"""ViolationLogger — buffered S3 shipping of violation events via boto3."""

from __future__ import annotations

import json
import time
from typing import Any

from normlayer.base_policy import ViolationEvent


class ViolationLogger:
    """Ships violation events to AWS S3 with buffered batch writes.

    Events are buffered internally and flushed to S3 either when the buffer
    reaches ``batch_size`` or when ``flush_interval_seconds`` has elapsed
    since the last flush.

    Single events are written as ``.json`` files; batches are written as
    ``.jsonl`` (one JSON object per line).

    S3 key format:
        - Individual: ``{prefix}{date}/{agent_id}/{timestamp}_{policy}.json``
        - Batch: ``{prefix}batches/{date}/{timestamp}.jsonl``

    Args:
        bucket: S3 bucket name.
        region: AWS region string (e.g. ``"us-east-1"``).
        prefix: Key prefix for all violation objects (default ``"violations/"``).
        batch_size: Flush after this many buffered events (default 10).
        flush_interval_seconds: Time-based flush trigger in seconds (default 60.0).
    """

    def __init__(
        self,
        bucket: str,
        region: str,
        prefix: str = "violations/",
        batch_size: int = 10,
        flush_interval_seconds: float = 60.0,
    ) -> None:
        self.bucket = bucket
        self.region = region
        self.prefix = prefix
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self._buffer: list[ViolationEvent] = []
        self._client: Any = None
        self._last_flush: float = time.monotonic()

    def _get_client(self) -> Any:
        """Return a lazily-initialized boto3 S3 client.

        Raises:
            ImportError: If boto3 is not installed.
        """
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3 violation logging. "
                    "Install it with: pip install normlayer[aws]"
                )
            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def _should_flush(self) -> bool:
        """Check whether the buffer should be flushed.

        Returns:
            True if buffer size >= batch_size or flush interval has elapsed.
        """
        if len(self._buffer) >= self.batch_size:
            return True
        elapsed = time.monotonic() - self._last_flush
        if elapsed >= self.flush_interval_seconds:
            return True
        return False

    def ship(self, event: ViolationEvent) -> None:
        """Buffer a violation event and flush if thresholds are met.

        This method never raises — S3 failures are caught and printed.

        Args:
            event: The violation to buffer/ship.
        """
        self._buffer.append(event)
        if self._should_flush():
            self.flush()

    def ship_immediate(self, event: ViolationEvent) -> None:
        """Ship a single event directly to S3, bypassing the buffer.

        Use this for critical violations that must be persisted immediately.

        Args:
            event: The violation to persist.
        """
        try:
            client = self._get_client()
            date = event.timestamp[:10]  # YYYY-MM-DD
            key = (
                f"{self.prefix}{date}/{event.agent_id}/"
                f"{event.timestamp}_{event.policy_violated}.json"
            )
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(event.model_dump()),
                ContentType="application/json",
            )
        except Exception as exc:
            print(f"[NormLayer] S3 immediate ship failed: {exc}")

    def flush(self) -> int:
        """Flush all buffered events to S3.

        A single event is written as ``.json``; multiple events are written
        as a ``.jsonl`` file (one JSON object per line).

        Returns:
            Number of events successfully shipped. Returns 0 on failure.
        """
        if not self._buffer:
            return 0
        events = list(self._buffer)
        self._buffer.clear()
        self._last_flush = time.monotonic()
        try:
            client = self._get_client()
            if len(events) == 1:
                event = events[0]
                date = event.timestamp[:10]
                key = (
                    f"{self.prefix}{date}/{event.agent_id}/"
                    f"{event.timestamp}_{event.policy_violated}.json"
                )
                client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=json.dumps(event.model_dump()),
                    ContentType="application/json",
                )
            else:
                timestamp = events[0].timestamp
                date = timestamp[:10]
                key = f"{self.prefix}batches/{date}/{timestamp}.jsonl"
                body = "\n".join(json.dumps(e.model_dump()) for e in events)
                client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=body,
                    ContentType="application/x-ndjson",
                )
            return len(events)
        except Exception as exc:
            print(f"[NormLayer] S3 flush failed: {exc}")
            return 0

    def fetch_all(
        self, agent_id: str | None = None
    ) -> list[ViolationEvent]:
        """Fetch stored violation events from S3.

        Paginates ``list_objects_v2``, downloads and parses ``.json`` and
        ``.jsonl`` files under the configured prefix.

        Args:
            agent_id: If provided, only return events for this agent.

        Returns:
            List of ViolationEvent objects.
        """
        try:
            client = self._get_client()
            events: list[ViolationEvent] = []
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    resp = client.get_object(Bucket=self.bucket, Key=key)
                    body = resp["Body"].read().decode("utf-8")
                    if key.endswith(".jsonl"):
                        for line in body.strip().split("\n"):
                            if line.strip():
                                events.append(
                                    ViolationEvent.model_validate_json(line)
                                )
                    elif key.endswith(".json"):
                        events.append(
                            ViolationEvent.model_validate_json(body)
                        )
            if agent_id is not None:
                events = [e for e in events if e.agent_id == agent_id]
            return events
        except Exception as exc:
            print(f"[NormLayer] S3 fetch failed: {exc}")
            return []
