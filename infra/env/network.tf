# One public subnet and no inbound rules. The instance's public IPv4 address carries only
# outbound traffic; Session Manager reaches the instance over a connection its agent opens.

resource "aws_vpc" "main" {
  cidr_block = "10.42.0.0/16"

  tags = {
    Name = "ces-revisions"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = var.availability_zone
  map_public_ip_on_launch = true

  tags = {
    Name = "ces-revisions-public"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "ces-revisions"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "ces-revisions-public"
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "vm" {
  name        = "ces-revisions-vm"
  description = "No ingress: Session Manager uses an outbound connection from the agent"
  vpc_id      = aws_vpc.main.id

  tags = {
    Name = "ces-revisions-vm"
  }
}

# OpenTofu removes the allow-all egress rule AWS gives a new group, so it is declared here.
resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.vm.id
  description       = "All outbound IPv4: packages, GitHub, PyPI, and Systems Manager"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}
