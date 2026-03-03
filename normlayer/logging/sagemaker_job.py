"""SageMaker Processing Job wrapper for batch violation audits."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# Hardcoded SageMaker sklearn container URIs for common regions.
_SKLEARN_CONTAINER_URIS: dict[str, str] = {
    "us-east-1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "us-east-2": "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "us-west-1": "746614075791.dkr.ecr.us-west-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "us-west-2": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "eu-west-1": "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "eu-central-1": "492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "ap-northeast-1": "354813040037.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "ap-southeast-1": "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
}


class SageMakerAuditJob:
    """Launches an AWS SageMaker Processing Job to batch-audit stored violation logs.

    The job reads violation JSON files from an S3 input prefix, runs a
    configurable audit script, and writes aggregated reports to an S3 output
    prefix. CloudWatch logs are captured automatically by SageMaker.

    Args:
        role_arn: IAM role ARN with ``sagemaker:CreateProcessingJob``,
            ``s3:GetObject``, and ``s3:PutObject`` permissions.
        input_s3_uri: S3 URI of the violation log prefix to audit.
        output_s3_uri: S3 URI where audit results will be written.
        instance_type: SageMaker instance type (default ``"ml.t3.medium"``).
        region: AWS region string.
        script_s3_uri: S3 URI of the audit script to run. Optional.
        max_runtime_seconds: Maximum runtime for the job (default 3600).
    """

    def __init__(
        self,
        role_arn: str,
        input_s3_uri: str,
        output_s3_uri: str,
        instance_type: str = "ml.t3.medium",
        region: str = "us-east-1",
        script_s3_uri: str | None = None,
        max_runtime_seconds: int = 3600,
    ) -> None:
        self.role_arn = role_arn
        self.input_s3_uri = input_s3_uri
        self.output_s3_uri = output_s3_uri
        self.instance_type = instance_type
        self.region = region
        self.script_s3_uri = script_s3_uri
        self.max_runtime_seconds = max_runtime_seconds
        self._client: Any = None

    def _get_client(self) -> Any:
        """Return a lazily-initialized boto3 SageMaker client.

        Raises:
            ImportError: If boto3 is not installed.
        """
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for SageMaker audit jobs. "
                    "Install it with: pip install normlayer[aws]"
                )
            self._client = boto3.client("sagemaker", region_name=self.region)
        return self._client

    def _generate_job_name(self) -> str:
        """Generate a unique job name based on the current UTC timestamp.

        Returns:
            Job name in format ``normlayer-audit-YYYYMMDD-HHMMSS``.
        """
        now = datetime.now(timezone.utc)
        return f"normlayer-audit-{now.strftime('%Y%m%d-%H%M%S')}"

    def _get_container_image(self) -> str:
        """Return the SageMaker sklearn container URI for the configured region.

        Returns:
            Container image URI string.

        Raises:
            ValueError: If the region is not in the supported list.
        """
        if self.region not in _SKLEARN_CONTAINER_URIS:
            raise ValueError(
                f"No SageMaker sklearn container URI configured for region "
                f"'{self.region}'. Supported regions: "
                f"{sorted(_SKLEARN_CONTAINER_URIS.keys())}"
            )
        return _SKLEARN_CONTAINER_URIS[self.region]

    def run(self) -> str:
        """Launch the SageMaker Processing Job.

        Returns:
            The generated job name for status tracking.
        """
        client = self._get_client()
        job_name = self._generate_job_name()
        image_uri = self._get_container_image()

        processing_inputs = [
            {
                "InputName": "violations",
                "S3Input": {
                    "S3Uri": self.input_s3_uri,
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ]

        if self.script_s3_uri is not None:
            processing_inputs.append(
                {
                    "InputName": "code",
                    "S3Input": {
                        "S3Uri": self.script_s3_uri,
                        "LocalPath": "/opt/ml/processing/code",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                }
            )

        client.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=self.role_arn,
            AppSpecification={
                "ImageUri": image_uri,
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": self.instance_type,
                    "VolumeSizeInGB": 10,
                }
            },
            ProcessingInputs=processing_inputs,
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "audit-results",
                        "S3Output": {
                            "S3Uri": self.output_s3_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            StoppingCondition={
                "MaxRuntimeInSeconds": self.max_runtime_seconds,
            },
        )
        return job_name

    def status(self, job_name: str) -> str:
        """Query the status of a SageMaker Processing Job.

        Args:
            job_name: The job name returned by :meth:`run`.

        Returns:
            Job status string (e.g. ``"InProgress"``, ``"Completed"``, ``"Failed"``).
        """
        client = self._get_client()
        resp = client.describe_processing_job(ProcessingJobName=job_name)
        return resp["ProcessingJobStatus"]
