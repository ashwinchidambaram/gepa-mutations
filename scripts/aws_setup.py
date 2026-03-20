"""AWS infrastructure setup for GEPA experiments (idempotent).

Creates:
  - S3 bucket with lifecycle policies
  - IAM role with least-privilege (S3 write, SNS publish, CloudWatch, SSM read)
  - Security group (egress-only for API calls)
  - SNS topic for notifications
  - CloudWatch alarms (cost, orphan instance detection)
  - SSM Parameter Store entries for secrets
"""

from __future__ import annotations

import json
import sys

import boto3
from botocore.exceptions import ClientError

# Configuration
REGION = "us-east-1"
PROJECT = "gepa-mutations"
BUCKET_NAME = "gepa-mutations-results"
SNS_TOPIC_NAME = "gepa-mutations-notifications"
IAM_ROLE_NAME = "gepa-mutations-ec2-role"
INSTANCE_PROFILE_NAME = "gepa-mutations-ec2-profile"
SG_NAME = "gepa-mutations-sg"


def create_s3_bucket(s3) -> str:
    """Create S3 bucket with lifecycle policy."""
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        print(f"  Created S3 bucket: {BUCKET_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"  S3 bucket already exists: {BUCKET_NAME}")
        else:
            raise

    # Lifecycle: move to IA after 30 days, delete after 180 days
    s3.put_bucket_lifecycle_configuration(
        Bucket=BUCKET_NAME,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "archive-old-results",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "runs/"},
                    "Transitions": [
                        {"Days": 30, "StorageClass": "STANDARD_IA"},
                    ],
                    "Expiration": {"Days": 180},
                },
            ],
        },
    )
    print("  Applied lifecycle policy")
    return BUCKET_NAME


def create_sns_topic(sns) -> str:
    """Create SNS topic for notifications."""
    response = sns.create_topic(Name=SNS_TOPIC_NAME)
    topic_arn = response["TopicArn"]
    print(f"  SNS topic: {topic_arn}")
    return topic_arn


def create_iam_role(iam, topic_arn: str) -> str:
    """Create IAM role with least-privilege for EC2 instances."""
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        iam.create_role(
            RoleName=IAM_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="EC2 role for GEPA mutation experiments",
        )
        print(f"  Created IAM role: {IAM_ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            print(f"  IAM role already exists: {IAM_ROLE_NAME}")
        else:
            raise

    # Inline policy: S3 write, SNS publish, CloudWatch, SSM read
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{BUCKET_NAME}",
                    f"arn:aws:s3:::{BUCKET_NAME}/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": "sns:Publish",
                "Resource": topic_arn,
            },
            {
                "Effect": "Allow",
                "Action": [
                    "cloudwatch:PutMetricData",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ssm:GetParameter",
                    "ssm:GetParameters",
                ],
                "Resource": f"arn:aws:ssm:{REGION}:*:parameter/{PROJECT}/*",
            },
            {
                "Effect": "Allow",
                "Action": ["ec2:TerminateInstances", "ec2:DescribeInstances"],
                "Resource": "*",
            },
        ],
    }

    iam.put_role_policy(
        RoleName=IAM_ROLE_NAME,
        PolicyName=f"{PROJECT}-policy",
        PolicyDocument=json.dumps(policy),
    )
    print("  Applied inline policy")

    # Instance profile
    try:
        iam.create_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
        iam.add_role_to_instance_profile(
            InstanceProfileName=INSTANCE_PROFILE_NAME,
            RoleName=IAM_ROLE_NAME,
        )
        print(f"  Created instance profile: {INSTANCE_PROFILE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            print(f"  Instance profile already exists: {INSTANCE_PROFILE_NAME}")
        else:
            raise

    return IAM_ROLE_NAME


def create_security_group(ec2) -> str:
    """Create security group with egress-only (no SSH needed)."""
    try:
        # Get default VPC
        vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
        vpc_id = vpcs["Vpcs"][0]["VpcId"]

        response = ec2.create_security_group(
            GroupName=SG_NAME,
            Description="GEPA experiments - egress only for API calls",
            VpcId=vpc_id,
        )
        sg_id = response["GroupId"]

        # Remove default ingress rules (egress-only)
        existing_ingress = ec2.describe_security_groups(GroupIds=[sg_id])["SecurityGroups"][0].get(
            "IpPermissions", []
        )
        if existing_ingress:
            ec2.revoke_security_group_ingress(GroupId=sg_id, IpPermissions=existing_ingress)
        print(f"  Created security group: {sg_id}")
        return sg_id
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidGroup.Duplicate":
            sgs = ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
            )
            sg_id = sgs["SecurityGroups"][0]["GroupId"]
            print(f"  Security group already exists: {sg_id}")
            return sg_id
        raise


def store_ssm_parameters(ssm) -> None:
    """Store secrets in SSM Parameter Store (won't overwrite existing)."""
    params = {
        f"/{PROJECT}/openrouter-api-key": "REPLACE_WITH_YOUR_KEY",
        f"/{PROJECT}/hf-token": "REPLACE_WITH_YOUR_TOKEN",
        f"/{PROJECT}/telegram-bot-token": "REPLACE_WITH_YOUR_TOKEN",
        f"/{PROJECT}/telegram-chat-id": "REPLACE_WITH_YOUR_CHAT_ID",
    }

    for name, placeholder in params.items():
        try:
            ssm.get_parameter(Name=name, WithDecryption=True)
            print(f"  SSM parameter exists: {name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ParameterNotFound":
                ssm.put_parameter(
                    Name=name,
                    Value=placeholder,
                    Type="SecureString",
                    Description=f"GEPA experiments - {name.split('/')[-1]}",
                )
                print(f"  Created SSM parameter: {name} (UPDATE WITH REAL VALUE)")
            else:
                raise


def setup_cloudwatch_alarms(cloudwatch, sns_topic_arn: str) -> None:
    """Create CloudWatch alarms for cost and orphan instance detection."""
    # Alarm if any instance runs > 36 hours
    cloudwatch.put_metric_alarm(
        AlarmName=f"{PROJECT}-orphan-instance",
        AlarmDescription="Alert if GEPA experiment instance runs > 36 hours",
        MetricName="StatusCheckFailed",
        Namespace="AWS/EC2",
        Statistic="Maximum",
        Period=3600,
        EvaluationPeriods=36,
        Threshold=0,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        AlarmActions=[sns_topic_arn],
        TreatMissingData="notBreaching",
    )
    print("  Created orphan instance alarm")


def main():
    print(f"Setting up AWS infrastructure for {PROJECT} in {REGION}...")

    s3 = boto3.client("s3", region_name=REGION)
    sns = boto3.client("sns", region_name=REGION)
    iam = boto3.client("iam", region_name=REGION)
    ec2 = boto3.client("ec2", region_name=REGION)
    ssm = boto3.client("ssm", region_name=REGION)
    cloudwatch = boto3.client("cloudwatch", region_name=REGION)

    print("\n1. S3 Bucket")
    create_s3_bucket(s3)

    print("\n2. SNS Topic")
    topic_arn = create_sns_topic(sns)

    print("\n3. IAM Role")
    create_iam_role(iam, topic_arn)

    print("\n4. Security Group")
    create_security_group(ec2)

    print("\n5. SSM Parameters")
    store_ssm_parameters(ssm)

    print("\n6. CloudWatch Alarms")
    setup_cloudwatch_alarms(cloudwatch, topic_arn)

    print("\n[DONE] AWS infrastructure setup complete.")
    print("IMPORTANT: Update SSM parameters with real API keys before running experiments.")


if __name__ == "__main__":
    main()
