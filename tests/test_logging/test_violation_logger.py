"""Tests for ViolationLogger — buffered S3 shipping, flush, fetch_all."""

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from normlayer.base_policy import ViolationEvent
from normlayer.logging.violation_logger import ViolationLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    agent_id: str = "agent_a",
    policy: str = "TestPolicy",
    severity: str = "medium",
    timestamp: str = "2026-03-02T12:00:00+00:00",
) -> ViolationEvent:
    return ViolationEvent(
        timestamp=timestamp,
        agent_id=agent_id,
        policy_violated=policy,
        severity=severity,
        message_snippet="test message content",
        context_window_hash="abc123",
        handler_dispatched="warn",
        details="test violation",
    )


def _mock_paginator(objects: list[dict]) -> MagicMock:
    """Build a mock paginator that yields a single page of S3 objects."""
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": objects}]
    return paginator


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestViolationLoggerInit:
    def test_init_stores_config(self):
        logger = ViolationLogger(bucket="my-bucket", region="us-east-1")
        assert logger.bucket == "my-bucket"
        assert logger.region == "us-east-1"
        assert logger.prefix == "violations/"
        assert logger.batch_size == 10
        assert logger.flush_interval_seconds == 60.0

    def test_init_custom_params(self):
        logger = ViolationLogger(
            bucket="b", region="eu-west-1", prefix="logs/",
            batch_size=5, flush_interval_seconds=30.0,
        )
        assert logger.prefix == "logs/"
        assert logger.batch_size == 5
        assert logger.flush_interval_seconds == 30.0

    def test_no_boto3_import_on_init(self):
        """boto3 should not be imported until _get_client() is called."""
        logger = ViolationLogger(bucket="b", region="us-east-1")
        assert logger._client is None


# ---------------------------------------------------------------------------
# Lazy client
# ---------------------------------------------------------------------------


class TestLazyClient:
    @patch("normlayer.logging.violation_logger.boto3", create=True)
    def test_get_client_creates_s3_client(self, mock_boto3):
        mock_boto3.client.return_value = MagicMock()
        logger = ViolationLogger(bucket="b", region="us-west-2")
        # Patch the import inside _get_client
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            client = logger._get_client()
        assert client is not None
        mock_boto3.client.assert_called_once_with("s3", region_name="us-west-2")

    @patch("normlayer.logging.violation_logger.boto3", create=True)
    def test_get_client_caches(self, mock_boto3):
        mock_boto3.client.return_value = MagicMock()
        logger = ViolationLogger(bucket="b", region="us-east-1")
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            c1 = logger._get_client()
            c2 = logger._get_client()
        assert c1 is c2
        assert mock_boto3.client.call_count == 1


# ---------------------------------------------------------------------------
# ship() buffering
# ---------------------------------------------------------------------------


class TestShipBuffering:
    def test_ship_adds_to_buffer(self):
        logger = ViolationLogger(bucket="b", region="r", batch_size=100)
        event = _make_event()
        logger.ship(event)
        assert len(logger._buffer) == 1

    @patch.object(ViolationLogger, "flush", return_value=5)
    def test_ship_triggers_flush_at_batch_size(self, mock_flush):
        logger = ViolationLogger(bucket="b", region="r", batch_size=3)
        for _ in range(3):
            logger.ship(_make_event())
        mock_flush.assert_called()

    @patch.object(ViolationLogger, "flush", return_value=1)
    def test_ship_triggers_flush_on_time_interval(self, mock_flush):
        logger = ViolationLogger(
            bucket="b", region="r", batch_size=100,
            flush_interval_seconds=0.0,
        )
        logger.ship(_make_event())
        mock_flush.assert_called()

    def test_ship_no_flush_below_threshold(self):
        logger = ViolationLogger(
            bucket="b", region="r", batch_size=100,
            flush_interval_seconds=9999.0,
        )
        logger.ship(_make_event())
        # Buffer should still hold the event (no flush)
        assert len(logger._buffer) == 1


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_empty_buffer_returns_zero(self):
        logger = ViolationLogger(bucket="b", region="r")
        assert logger.flush() == 0

    def test_flush_single_event_writes_json(self):
        mock_client = MagicMock()
        logger = ViolationLogger(bucket="my-bucket", region="us-east-1")
        logger._client = mock_client
        event = _make_event()
        logger._buffer.append(event)

        count = logger.flush()

        assert count == 1
        mock_client.put_object.assert_called_once()
        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "my-bucket"
        assert call_kwargs["Key"].endswith(".json")
        assert call_kwargs["ContentType"] == "application/json"
        assert len(logger._buffer) == 0

    def test_flush_multiple_events_writes_jsonl(self):
        mock_client = MagicMock()
        logger = ViolationLogger(bucket="my-bucket", region="us-east-1")
        logger._client = mock_client
        logger._buffer.extend([_make_event(), _make_event(agent_id="agent_b")])

        count = logger.flush()

        assert count == 2
        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["Key"].endswith(".jsonl")
        assert call_kwargs["ContentType"] == "application/x-ndjson"
        body = call_kwargs["Body"]
        lines = body.strip().split("\n")
        assert len(lines) == 2

    def test_flush_clears_buffer(self):
        mock_client = MagicMock()
        logger = ViolationLogger(bucket="b", region="r")
        logger._client = mock_client
        logger._buffer.append(_make_event())
        logger.flush()
        assert len(logger._buffer) == 0

    def test_flush_on_s3_failure_returns_zero(self, capsys):
        mock_client = MagicMock()
        mock_client.put_object.side_effect = RuntimeError("S3 down")
        logger = ViolationLogger(bucket="b", region="r")
        logger._client = mock_client
        logger._buffer.append(_make_event())

        count = logger.flush()

        assert count == 0
        captured = capsys.readouterr()
        assert "S3 flush failed" in captured.out

    def test_flush_s3_key_format_single(self):
        mock_client = MagicMock()
        logger = ViolationLogger(bucket="b", region="r", prefix="violations/")
        logger._client = mock_client
        event = _make_event(
            agent_id="planner", policy="NoDeception",
            timestamp="2026-03-02T12:00:00+00:00",
        )
        logger._buffer.append(event)
        logger.flush()

        key = mock_client.put_object.call_args[1]["Key"]
        assert key.startswith("violations/2026-03-02/planner/")
        assert "NoDeception" in key
        assert key.endswith(".json")

    def test_flush_s3_key_format_batch(self):
        mock_client = MagicMock()
        logger = ViolationLogger(bucket="b", region="r", prefix="violations/")
        logger._client = mock_client
        logger._buffer.extend([_make_event(), _make_event()])
        logger.flush()

        key = mock_client.put_object.call_args[1]["Key"]
        assert key.startswith("violations/batches/2026-03-02/")
        assert key.endswith(".jsonl")


# ---------------------------------------------------------------------------
# ship_immediate()
# ---------------------------------------------------------------------------


class TestShipImmediate:
    def test_ship_immediate_bypasses_buffer(self):
        mock_client = MagicMock()
        logger = ViolationLogger(bucket="b", region="r")
        logger._client = mock_client
        event = _make_event()

        logger.ship_immediate(event)

        assert len(logger._buffer) == 0
        mock_client.put_object.assert_called_once()

    def test_ship_immediate_failure_does_not_raise(self, capsys):
        mock_client = MagicMock()
        mock_client.put_object.side_effect = RuntimeError("boom")
        logger = ViolationLogger(bucket="b", region="r")
        logger._client = mock_client

        logger.ship_immediate(_make_event())  # should not raise

        captured = capsys.readouterr()
        assert "S3 immediate ship failed" in captured.out


# ---------------------------------------------------------------------------
# fetch_all()
# ---------------------------------------------------------------------------


class TestFetchAll:
    def test_fetch_all_parses_json_files(self):
        event = _make_event()
        event_json = event.model_dump_json()

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = _mock_paginator(
            [{"Key": "violations/2026-03-02/agent_a/event.json"}]
        )
        mock_client.get_object.return_value = {
            "Body": io.BytesIO(event_json.encode("utf-8"))
        }

        logger = ViolationLogger(bucket="b", region="r")
        logger._client = mock_client

        results = logger.fetch_all()
        assert len(results) == 1
        assert results[0].agent_id == "agent_a"

    def test_fetch_all_parses_jsonl_files(self):
        e1 = _make_event(agent_id="a1")
        e2 = _make_event(agent_id="a2")
        body = e1.model_dump_json() + "\n" + e2.model_dump_json()

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = _mock_paginator(
            [{"Key": "violations/batches/2026-03-02/batch.jsonl"}]
        )
        mock_client.get_object.return_value = {
            "Body": io.BytesIO(body.encode("utf-8"))
        }

        logger = ViolationLogger(bucket="b", region="r")
        logger._client = mock_client

        results = logger.fetch_all()
        assert len(results) == 2

    def test_fetch_all_filters_by_agent_id(self):
        e1 = _make_event(agent_id="keep")
        e2 = _make_event(agent_id="skip")
        body = e1.model_dump_json() + "\n" + e2.model_dump_json()

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = _mock_paginator(
            [{"Key": "violations/batches/batch.jsonl"}]
        )
        mock_client.get_object.return_value = {
            "Body": io.BytesIO(body.encode("utf-8"))
        }

        logger = ViolationLogger(bucket="b", region="r")
        logger._client = mock_client

        results = logger.fetch_all(agent_id="keep")
        assert len(results) == 1
        assert results[0].agent_id == "keep"

    def test_fetch_all_empty_bucket(self):
        mock_client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{}]  # no Contents key
        mock_client.get_paginator.return_value = paginator

        logger = ViolationLogger(bucket="b", region="r")
        logger._client = mock_client

        results = logger.fetch_all()
        assert results == []

    def test_fetch_all_on_failure_returns_empty(self, capsys):
        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = RuntimeError("S3 down")

        logger = ViolationLogger(bucket="b", region="r")
        logger._client = mock_client

        results = logger.fetch_all()
        assert results == []
        captured = capsys.readouterr()
        assert "S3 fetch failed" in captured.out
