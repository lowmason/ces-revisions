# A monthly cost budget over the costs tagged project = ces-revisions. At 100% of actual
# spend, a budget action stops the instance through a role that can do only that. All four
# resources wait for budget_enabled, which follows the tag's activation.

resource "aws_budgets_budget" "monthly" {
  count = var.budget_enabled ? 1 : 0

  name         = "ces-revisions-monthly"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Budgets name a user-defined cost allocation tag as user:<key>$<value>.
  cost_filter {
    name   = "TagKeyValue"
    values = ["user:project$ces-revisions"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }
}

resource "aws_iam_role" "budget_action" {
  count = var.budget_enabled ? 1 : 0

  name = "ces-revisions-budget-stop"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "budgets.amazonaws.com" }
      Condition = {
        StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
        ArnLike      = { "aws:SourceArn" = "arn:aws:budgets::${data.aws_caller_identity.current.account_id}:budget/*" }
      }
    }]
  })
}

# The EC2 stop half of AWS's AWSBudgetsActions_RolePolicyForResourceAdministrationWithSSM,
# narrowed to this instance.
resource "aws_iam_role_policy" "budget_action" {
  count = var.budget_enabled ? 1 : 0

  name = "stop-the-vm"
  role = aws_iam_role.budget_action[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "RunTheStopAutomation"
        Effect = "Allow"
        Action = "ssm:StartAutomationExecution"
        Resource = [
          "arn:aws:ssm:*:*:document/AWS-StopEC2Instance",
          "arn:aws:ssm:*:*:automation-definition/AWS-StopEC2Instance:*",
          "arn:aws:ssm:*:*:automation-execution/*",
        ]
      },
      {
        Sid       = "StopTheVm"
        Effect    = "Allow"
        Action    = "ec2:StopInstances"
        Resource  = aws_instance.vm.arn
        Condition = { "ForAnyValue:StringEquals" = { "aws:CalledVia" = ["ssm.amazonaws.com"] } }
      },
      {
        Sid       = "ReadInstanceStatus"
        Effect    = "Allow"
        Action    = "ec2:DescribeInstanceStatus"
        Resource  = "*"
        Condition = { "ForAnyValue:StringEquals" = { "aws:CalledVia" = ["ssm.amazonaws.com"] } }
      },
    ]
  })
}

resource "aws_budgets_budget_action" "stop_vm" {
  count = var.budget_enabled ? 1 : 0

  budget_name        = aws_budgets_budget.monthly[0].name
  action_type        = "RUN_SSM_DOCUMENTS"
  approval_model     = "AUTOMATIC"
  notification_type  = "ACTUAL"
  execution_role_arn = aws_iam_role.budget_action[0].arn

  action_threshold {
    action_threshold_type  = "PERCENTAGE"
    action_threshold_value = 100
  }

  definition {
    ssm_action_definition {
      action_sub_type = "STOP_EC2_INSTANCES"
      region          = var.region
      instance_ids    = [aws_instance.vm.id]
    }
  }

  subscriber {
    address           = var.budget_email
    subscription_type = "EMAIL"
  }
}
