output "cluster_id" {
  description = "OKE cluster OCID"
  value       = oci_containerengine_cluster.main.id
}

output "cluster_name" {
  description = "OKE cluster name"
  value       = oci_containerengine_cluster.main.name
}

output "kubeconfig_command" {
  description = "Run this to configure kubectl access to the cluster"
  value = "oci ce cluster create-kubeconfig --cluster-id ${oci_containerengine_cluster.main.id} --file $HOME/.kube/config --region ${var.region} --token-version 2.0.0 --kube-endpoint PUBLIC_ENDPOINT"
}

output "ocir_backend_repo" {
  description = "Backend image repository path"
  value       = oci_artifacts_container_repository.backend.display_name
}

output "ocir_frontend_repo" {
  description = "Frontend image repository path"
  value       = oci_artifacts_container_repository.frontend.display_name
}
