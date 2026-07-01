provider "oci" {
  # Uses your ~/.oci/config "DEFAULT" profile by default.
  # Region is overridable via variable for portability.
  region = var.region
}

# Fetch the cluster's kubeconfig content for the k8s provider
data "oci_containerengine_cluster_kube_config" "main" {
  cluster_id = oci_containerengine_cluster.main.id
}

locals {
  kubeconfig = yamldecode(data.oci_containerengine_cluster_kube_config.main.content)
}

provider "kubernetes" {
  host                   = local.kubeconfig["clusters"][0]["cluster"]["server"]
  cluster_ca_certificate = base64decode(local.kubeconfig["clusters"][0]["cluster"]["certificate-authority-data"])

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "oci"
    args        = local.kubeconfig["users"][0]["user"]["exec"]["args"]
  }
}
