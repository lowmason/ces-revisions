# The S3 bucket that holds infra/env's state. This root's own state stays local, in the
# gitignored terraform.tfstate beside this file.

terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.64"
    }
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

variable "region" {
  description = "The region Req 2 chose, pinned in pinned.auto.tfvars."
  type        = string
}

resource "aws_s3_bucket" "state" {
  # A prefix rather than a name: AWS appends a unique suffix, so no bucket name is committed.
  bucket_prefix = "ces-revisions-tofu-state-"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

output "bucket" {
  description = "The state bucket's generated name, for the gitignored infra/env/backend.hcl."
  value       = aws_s3_bucket.state.bucket
  # Kept out of plan and apply output; tofu output -raw bucket still prints it.
  sensitive = true
}
