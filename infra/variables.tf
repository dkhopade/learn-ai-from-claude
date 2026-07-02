variable "region" {
  description = "OCI region identifier, e.g. us-phoenix-1"
  type        = string
}

variable "compartment_ocid" {
  description = "OCID of the compartment to deploy into"
  type        = string
}

variable "project_name" {
  description = "Short name prefix for all resources"
  type        = string
  default     = "learn-ai"
}

variable "enable_gpu" {
  description = "Whether to create the GPU nodepool (the expensive part)"
  type        = bool
  default     = false
}

variable "kubernetes_version" {
  description = "OKE Kubernetes version"
  type        = string
  default     = "v1.35.2"
}

variable "tenancy_ocid" {
  description = "OCID of the tenancy (root compartment) — required for IAM"
  type        = string
}

variable "ocir_auth_token" {
  description = "OCIR auth token for image pulls (sensitive)"
  type        = string
  sensitive   = true
}

variable "ocir_username" {
  description = "OCIR username (email or federated username)"
  type        = string
  default     = "deepak.khopade@oracle.com"
}

variable "ocir_region_key" {
  description = "OCIR region key (e.g. iad for Ashburn, phx for Phoenix)"
  type        = string
  default     = "iad"
}
