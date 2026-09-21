# The VM: one instance on one root volume, whose type follows var.size.

locals {
  instance_types = {
    dev  = "m7i.xlarge"
    l4   = "g6.xlarge"
    h100 = "p5.4xlarge"
  }

  # The files cloud-init writes under /opt/ces-revisions at first boot, with their modes,
  # named by their paths under infra/vm.
  vm_files = {
    "first-boot.sh"                = "0755"
    "guards/install.sh"            = "0755"
    "guards/idle_stop.py"          = "0755"
    "guards/idle-stop.env"         = "0644"
    "guards/ces-idle-stop.service" = "0644"
    "guards/ces-idle-stop.timer"   = "0644"
    "guards/ces-gpu-cap.service"   = "0644"
  }

  # cloud-init applies this once per instance: the SSH key for ubuntu, the files above, and
  # first-boot.sh, which runs after cloud-init's own package stage.
  cloud_config = {
    ssh_authorized_keys = [trimspace(file(pathexpand(var.ssh_public_key_path)))]
    write_files = [
      for name, permissions in local.vm_files : {
        path        = "/opt/ces-revisions/${name}"
        permissions = permissions
        content     = file("${path.module}/../vm/${name}")
      }
    ]
    runcmd = [["/opt/ces-revisions/first-boot.sh"]]
  }

  user_data = "#cloud-config\n${yamlencode(local.cloud_config)}"
}

resource "aws_iam_role" "vm" {
  name = "ces-revisions-vm"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

# The instance's only AWS permission: registering with Systems Manager.
resource "aws_iam_role_policy_attachment" "vm_ssm" {
  role       = aws_iam_role.vm.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "vm" {
  name = "ces-revisions-vm"
  role = aws_iam_role.vm.name
}

resource "aws_instance" "vm" {
  ami                                  = var.ami_id
  instance_type                        = local.instance_types[var.size]
  subnet_id                            = aws_subnet.public.id
  vpc_security_group_ids               = [aws_security_group.vm.id]
  iam_instance_profile                 = aws_iam_instance_profile.vm.name
  instance_initiated_shutdown_behavior = "stop"
  user_data                            = local.user_data

  metadata_options {
    http_tokens = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 100
    encrypted             = true
    delete_on_termination = true
  }

  tags = {
    Name = "ces-revisions-vm"
  }

  lifecycle {
    prevent_destroy = true
    # cloud-init runs once per instance; later changes reach the VM through setup.sh.
    ignore_changes = [user_data]

    precondition {
      condition     = length(local.user_data) <= 16384
      error_message = "EC2 accepts at most 16 KB of user data."
    }
  }
}
