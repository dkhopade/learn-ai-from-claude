# Get the tenancy namespace for the OCIR server path
data "oci_objectstorage_namespace" "ns" {
  compartment_id = var.compartment_ocid
}

resource "kubernetes_secret_v1" "ocir" {
  metadata {
    name = "ocir-secret"
  }

  type = "kubernetes.io/dockerconfigjson"

  data = {
    ".dockerconfigjson" = jsonencode({
      auths = {
        "${var.ocir_region_key}.ocir.io" = {
          username = "${data.oci_objectstorage_namespace.ns.namespace}/${var.ocir_username}"
          password = var.ocir_auth_token
          auth     = base64encode("${data.oci_objectstorage_namespace.ns.namespace}/${var.ocir_username}:${var.ocir_auth_token}")
        }
      }
    })
  }
}
