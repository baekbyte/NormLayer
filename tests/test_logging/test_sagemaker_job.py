"""Tests for SageMakerAuditJob — mocked boto3, no AWS credentials needed."""

from unittest.mock import MagicMock, patch

import pytest

from normlayer.logging.sagemaker_job import SageMakerAuditJob, _SKLEARN_CONTAINER_URIS


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestSageMakerAuditJobInit:
    def test_init_stores_config(self):
        job = SageMakerAuditJob(
            role_arn="arn:aws:iam::123:role/test",
            input_s3_uri="s3://bucket/input",
            output_s3_uri="s3://bucket/output",
        )
        assert job.role_arn == "arn:aws:iam::123:role/test"
        assert job.input_s3_uri == "s3://bucket/input"
        assert job.output_s3_uri == "s3://bucket/output"
        assert job.instance_type == "ml.t3.medium"
        assert job.region == "us-east-1"
        assert job.script_s3_uri is None
        assert job.max_runtime_seconds == 3600

    def test_init_custom_params(self):
        job = SageMakerAuditJob(
            role_arn="arn",
            input_s3_uri="s3://in",
            output_s3_uri="s3://out",
            instance_type="ml.m5.large",
            region="eu-west-1",
            script_s3_uri="s3://scripts/audit.py",
            max_runtime_seconds=7200,
        )
        assert job.instance_type == "ml.m5.large"
        assert job.region == "eu-west-1"
        assert job.script_s3_uri == "s3://scripts/audit.py"
        assert job.max_runtime_seconds == 7200

    def test_no_boto3_import_on_init(self):
        job = SageMakerAuditJob(
            role_arn="arn", input_s3_uri="s3://in", output_s3_uri="s3://out",
        )
        assert job._client is None


# ---------------------------------------------------------------------------
# Job name generation
# ---------------------------------------------------------------------------


class TestJobNameGeneration:
    def test_job_name_format(self):
        job = SageMakerAuditJob(
            role_arn="arn", input_s3_uri="s3://in", output_s3_uri="s3://out",
        )
        name = job._generate_job_name()
        assert name.startswith("normlayer-audit-")
        # Format: normlayer-audit-YYYYMMDD-HHMMSS
        parts = name.split("-")
        assert len(parts) == 4  # normlayer, audit, YYYYMMDD, HHMMSS


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------


class TestContainerImage:
    def test_known_region_returns_uri(self):
        job = SageMakerAuditJob(
            role_arn="arn", input_s3_uri="s3://in", output_s3_uri="s3://out",
            region="us-east-1",
        )
        uri = job._get_container_image()
        assert "us-east-1" in uri
        assert "sagemaker-scikit-learn" in uri

    def test_unknown_region_raises_value_error(self):
        job = SageMakerAuditJob(
            role_arn="arn", input_s3_uri="s3://in", output_s3_uri="s3://out",
            region="mars-west-1",
        )
        with pytest.raises(ValueError, match="mars-west-1"):
            job._get_container_image()


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_calls_create_processing_job(self):
        mock_client = MagicMock()
        job = SageMakerAuditJob(
            role_arn="arn:aws:iam::123:role/test",
            input_s3_uri="s3://bucket/violations/",
            output_s3_uri="s3://bucket/audit-results/",
            region="us-east-1",
        )
        job._client = mock_client

        job_name = job.run()

        assert job_name.startswith("normlayer-audit-")
        mock_client.create_processing_job.assert_called_once()
        call_kwargs = mock_client.create_processing_job.call_args[1]
        assert call_kwargs["ProcessingJobName"] == job_name
        assert call_kwargs["RoleArn"] == "arn:aws:iam::123:role/test"
        assert call_kwargs["ProcessingResources"]["ClusterConfig"]["InstanceType"] == "ml.t3.medium"
        assert call_kwargs["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"] == 10

    def test_run_includes_script_input_when_provided(self):
        mock_client = MagicMock()
        job = SageMakerAuditJob(
            role_arn="arn",
            input_s3_uri="s3://in",
            output_s3_uri="s3://out",
            script_s3_uri="s3://scripts/audit.py",
            region="us-east-1",
        )
        job._client = mock_client

        job.run()

        call_kwargs = mock_client.create_processing_job.call_args[1]
        inputs = call_kwargs["ProcessingInputs"]
        assert len(inputs) == 2
        assert inputs[1]["InputName"] == "code"

    def test_run_without_script_has_single_input(self):
        mock_client = MagicMock()
        job = SageMakerAuditJob(
            role_arn="arn",
            input_s3_uri="s3://in",
            output_s3_uri="s3://out",
            region="us-east-1",
        )
        job._client = mock_client

        job.run()

        call_kwargs = mock_client.create_processing_job.call_args[1]
        inputs = call_kwargs["ProcessingInputs"]
        assert len(inputs) == 1


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_returns_job_status(self):
        mock_client = MagicMock()
        mock_client.describe_processing_job.return_value = {
            "ProcessingJobStatus": "Completed"
        }
        job = SageMakerAuditJob(
            role_arn="arn", input_s3_uri="s3://in", output_s3_uri="s3://out",
        )
        job._client = mock_client

        result = job.status("normlayer-audit-20260302-120000")
        assert result == "Completed"
        mock_client.describe_processing_job.assert_called_once_with(
            ProcessingJobName="normlayer-audit-20260302-120000"
        )

    def test_status_in_progress(self):
        mock_client = MagicMock()
        mock_client.describe_processing_job.return_value = {
            "ProcessingJobStatus": "InProgress"
        }
        job = SageMakerAuditJob(
            role_arn="arn", input_s3_uri="s3://in", output_s3_uri="s3://out",
        )
        job._client = mock_client

        assert job.status("some-job") == "InProgress"
