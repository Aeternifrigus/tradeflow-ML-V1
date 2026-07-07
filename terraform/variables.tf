variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "europe-central2" # Warsaw
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "tradeflow-api"
}

variable "artifact_repo_name" {
  description = "Name of the Artifact Registry Docker repository"
  type        = string
  default     = "tradeflow-ml"
}

variable "image_tag" {
  description = "Container image tag to deploy (set by CI to the git SHA)"
  type        = string
  default     = "latest"
}

variable "min_instances" {
  type    = number
  default = 0
}

variable "max_instances" {
  type    = number
  default = 3
}
