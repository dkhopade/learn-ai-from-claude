# --- GPU nodepool (toggleable via enable_gpu) ---
# Find the GPU-compatible OL8 image matching our k8s version
locals {
  gpu_compatible_images = [
    for src in data.oci_containerengine_node_pool_option.options.sources : src
    if can(regex("Oracle-Linux-8", src.source_name))
    && can(regex("GPU", src.source_name))
    && !can(regex("aarch64", src.source_name))
    && can(regex(local.k8s_ver_short, src.source_name))
  ]
  gpu_node_image_id = length(local.gpu_compatible_images) > 0 ? local.gpu_compatible_images[0].image_id : ""
}

resource "oci_containerengine_node_pool" "gpu_pool" {
  count              = var.enable_gpu ? 1 : 0
  cluster_id         = oci_containerengine_cluster.main.id
  compartment_id     = var.compartment_ocid
  kubernetes_version = var.kubernetes_version
  name               = "${var.project_name}-gpu-pool"
  node_shape         = "VM.GPU.A10.1"

  node_config_details {
    size = 1
    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.private.id
    }
  }

  node_source_details {
    image_id                = local.gpu_node_image_id
    source_type             = "IMAGE"
    boot_volume_size_in_gbs = 500
  }

  node_metadata = {
    user_data = base64encode(<<-EOF
      #!/bin/bash
      curl --fail -H "Authorization: Bearer Oracle" -L0 http://169.254.169.254/opc/v2/instance/metadata/oke_init_script | base64 --decode >/var/run/oke-init.sh
      bash /var/run/oke-init.sh
      /usr/libexec/oci-growfs -y
    EOF
    )
  }

  # Taint so only GPU workloads schedule here
  initial_node_labels {
    key   = "node-type"
    value = "gpu"
  }
}
