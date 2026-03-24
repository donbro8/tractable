terraform {
  required_version = ">= 1.0"
}

module "network" {
  source  = "./modules/network"
  version = "1.0.0"
}

module "remote_module" {
  source  = "hashicorp/consul/aws"
  version = "0.1.0"
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_security_group" "sg" {
  name   = "example-sg"
  vpc_id = aws_vpc.main.id
}

resource "aws_instance" "web" {
  ami           = "ami-0abcdef1234567890"
  instance_type = "t3.micro"

  depends_on = [aws_security_group.sg]
}

variable "environment" {
  type        = string
  description = "Deployment environment"
  default     = "dev"
}

variable "instance_count" {
  type        = number
  description = "Number of instances to create"
}

output "vpc_id" {
  description = "The ID of the VPC"
  value       = aws_vpc.main.id
}

output "instance_ids" {
  description = "List of instance IDs"
  value       = [aws_instance.web.id]
}

data "aws_ami" "ubuntu" {
  owners = ["example-owner"]
}
