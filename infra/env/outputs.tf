output "instance_id" {
  description = "The instance, which keeps this ID across sizes."
  value       = aws_instance.vm.id
}

output "region" {
  description = "The region, for AWS CLI commands."
  value       = var.region
}

output "availability_zone" {
  description = "The instance's zone."
  value       = aws_instance.vm.availability_zone
}

output "instance_type" {
  description = "The instance type the current size maps to."
  value       = aws_instance.vm.instance_type
}

output "root_volume_id" {
  description = "The root volume, which keeps this ID across sizes."
  value       = aws_instance.vm.root_block_device[0].volume_id
}

output "security_group_id" {
  description = "The instance's security group, which has no ingress rules."
  value       = aws_security_group.vm.id
}

output "budget_name" {
  description = "The monthly cost budget; null until budget_enabled is true."
  value       = one(aws_budgets_budget.monthly[*].name)
}

output "budget_action_id" {
  description = "The budget action that stops the instance, for resetting it; null until budget_enabled is true."
  value       = one(aws_budgets_budget_action.stop_vm[*].action_id)
}
