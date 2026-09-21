terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.64"
    }
  }

  # Bucket, key, and region come from the gitignored backend.hcl (backend.hcl.example shows
  # its shape): tofu init -backend-config=backend.hcl. The lock is an S3 object written with
  # a conditional put, so no DynamoDB table exists.
  backend "s3" {
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      project = "ces-revisions"
    }
  }
}

# The account the trust policies name, read at plan time so no account ID is committed.
data "aws_caller_identity" "current" {}
