variable "project_id" {
  description = "The project ID of the cluster"
  default     = "human-pose-estimation-418913"
}

variable "region" {
  description = "The region the cluster in"
  default     = "asia-southeast1-b"
}

variable "k8s" {
  description = "Kubernetes cluster for human pose estimation system"
  default     = "human-pose-estimation"
}
