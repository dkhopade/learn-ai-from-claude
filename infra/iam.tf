# --- Dynamic group matching the OKE cluster + nodes in this compartment ---
resource "oci_identity_dynamic_group" "oke_nodes" {
  compartment_id = var.tenancy_ocid
  name           = "${var.project_name}-oke-dg"
  description    = "Dynamic group for OKE cluster and worker nodes"
  matching_rule  = "ALL {instance.compartment.id = '${var.compartment_ocid}'}"
}

# --- Policy granting the cluster what it needs ---
resource "oci_identity_policy" "oke_policy" {
  compartment_id = var.tenancy_ocid
  name           = "${var.project_name}-oke-policy"
  description    = "Permissions for OKE to manage load balancers, volumes, and registry"

  statements = [
    "Allow dynamic-group ${oci_identity_dynamic_group.oke_nodes.name} to manage load-balancers in compartment id ${var.compartment_ocid}",
    "Allow dynamic-group ${oci_identity_dynamic_group.oke_nodes.name} to use virtual-network-family in compartment id ${var.compartment_ocid}",
    "Allow dynamic-group ${oci_identity_dynamic_group.oke_nodes.name} to manage volume-family in compartment id ${var.compartment_ocid}",
    "Allow dynamic-group ${oci_identity_dynamic_group.oke_nodes.name} to read repos in compartment id ${var.compartment_ocid}",
  ]
}
