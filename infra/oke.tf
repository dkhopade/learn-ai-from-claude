# --- Get availability domains for node placement ---
data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# --- Find the correct x86 Oracle Linux 8 image for our K8s version ---
data "oci_containerengine_node_pool_option" "options" {
  node_pool_option_id = oci_containerengine_cluster.main.id
  compartment_id      = var.compartment_ocid
}

locals {
  # pick the plain x86 OL8 image matching our k8s version:
  # exclude aarch64 (ARM) and GPU images
  k8s_ver_short = replace(var.kubernetes_version, "v", "")

  compatible_images = [
    for src in data.oci_containerengine_node_pool_option.options.sources : src
    if can(regex("Oracle-Linux-8", src.source_name))
    && !can(regex("aarch64", src.source_name))
    && !can(regex("GPU", src.source_name))
    && can(regex(local.k8s_ver_short, src.source_name))
  ]

  node_image_id = local.compatible_images[0].image_id
}

# --- The OKE cluster ---
resource "oci_containerengine_cluster" "main" {
  compartment_id     = var.compartment_ocid
  kubernetes_version = var.kubernetes_version
  name               = "${var.project_name}-oke"
  vcn_id             = oci_core_vcn.main.id
  type               = "ENHANCED_CLUSTER"

  endpoint_config {
    subnet_id            = oci_core_subnet.public.id
    is_public_ip_enabled = true
  }

  options {
    service_lb_subnet_ids = [oci_core_subnet.public.id]
    add_ons {
      is_kubernetes_dashboard_enabled = false
      is_tiller_enabled               = false
    }
  }
}

# --- CPU nodepool for FastAPI, Qdrant, frontend ---
resource "oci_containerengine_node_pool" "cpu_pool" {
  cluster_id         = oci_containerengine_cluster.main.id
  compartment_id     = var.compartment_ocid
  kubernetes_version = var.kubernetes_version
  name               = "${var.project_name}-cpu-pool"
  node_shape         = "VM.Standard.E5.Flex"

  node_shape_config {
    ocpus         = 2
    memory_in_gbs = 16
  }

  node_config_details {
    size = 2
    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.private.id
    }
  }

  node_source_details {
    image_id    = local.node_image_id
    source_type = "IMAGE"
  }
}
