variable "region" {
  description = "The region Req 2 chose, pinned in pinned.auto.tfvars."
  type        = string
}

variable "availability_zone" {
  description = "The zone Req 2 chose, pinned in pinned.auto.tfvars."
  type        = string
}

variable "ami_id" {
  description = "Canonical's Ubuntu 24.04 LTS amd64 image, read once from its public SSM parameter and pinned in pinned.auto.tfvars."
  type        = string
}

variable "size" {
  description = "dev (m7i.xlarge), l4 (g6.xlarge), l40s (g6e.xlarge), a10g (g5.xlarge), or h100 (p5.4xlarge). infra/bin/vm size records the last applied size in the gitignored size.auto.tfvars."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "l4", "l40s", "a10g", "h100"], var.size)
    error_message = "The size must be dev, l4, l40s, a10g, or h100."
  }
}

variable "ssh_public_key_path" {
  description = "The public half of the VM's dedicated SSH key pair; the private half stays on the Mac."
  type        = string
  default     = "~/.ssh/ces-revisions-vm.pub"
}

variable "budget_email" {
  description = "Where AWS Budgets sends alerts. A personal value, so it lives in the gitignored terraform.tfvars."
  type        = string
  sensitive   = true
}

variable "budget_enabled" {
  description = "Whether the budget and its stop action exist. A budget filters on an activated cost allocation tag, and the project tag can be activated only after resources carry it, so this starts false."
  type        = bool
  default     = false
}

variable "monthly_budget_usd" {
  description = "The monthly cost ceiling in US dollars, revisited at roadmap Stage 6."
  type        = string
  default     = "150"
}
