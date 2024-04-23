variable "aws_region" {
  default = "ap-southeast-1"
  type    = string
}

variable "ami_id" {
  default = "ami-06c4be2792f419b7b"
  type    = string
}

variable "instance_type" {
  default = "t2.medium"
  type    = string
}

variable "key_name" {
  default = "terraform-key"
  type    = string
}

variable "bucket" {
  default = "jenkins-s3-bucket-jupi155"
  type    = string
}

variable "acl" {
  default = "private"
  type    = string
}
