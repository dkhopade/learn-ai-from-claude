resource "oci_artifacts_container_repository" "backend" {
  compartment_id = var.compartment_ocid
  display_name   = "${var.project_name}/ai-backend"
  is_public      = false
}

resource "oci_artifacts_container_repository" "frontend" {
  compartment_id = var.compartment_ocid
  display_name   = "${var.project_name}/ai-frontend"
  is_public      = false
}
