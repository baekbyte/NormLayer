"""SageMaker Processing Job wrapper for batch violation audits. Stub, planned for Week 3-4."""

from __future__ import annotations


class SageMakerAuditJob:
    """Launches an AWS SageMaker Processing Job to batch-audit stored violation logs.

    The job reads violation JSON files from an S3 input prefix, runs a
    configurable audit script, and writes aggregated reports to an S3 output
    prefix. CloudWatch logs are captured automatically by SageMaker.

    Recommended instance type during development: ``ml.t3.medium`` to keep
    costs low. Violations should be batched and not streamed individually.

    .. note::
        This class is a stub. Full implementation is planned for Week 3-4.

    Args:
        role_arn: IAM role ARN with ``sagemaker:CreateProcessingJob``,
            ``s3:GetObject``, and ``s3:PutObject`` permissions.
        input_s3_uri: S3 URI of the violation log prefix to audit.
        output_s3_uri: S3 URI where audit results will be written.
        instance_type: SageMaker instance type (default ``"ml.t3.medium"``).
        region: AWS region string.
    """

    def __init__(
        self,
        role_arn: str,
        input_s3_uri: str,
        output_s3_uri: str,
        instance_type: str = "ml.t3.medium",
        region: str = "us-east-1",
    ) -> None:
        self.role_arn = role_arn
        self.input_s3_uri = input_s3_uri
        self.output_s3_uri = output_s3_uri
        self.instance_type = instance_type
        self.region = region

    def run(self) -> str:
        """Launch the SageMaker Processing Job and return its job name.

        Returns:
            The SageMaker job name for status tracking.

        Raises:
            NotImplementedError: Full implementation planned for Week 3-4.
        """
        raise NotImplementedError(
            "SageMakerAuditJob.run will be implemented in Week 3-4."
        )
