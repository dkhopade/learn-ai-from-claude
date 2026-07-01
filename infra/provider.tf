provider "oci" {
  # Uses your ~/.oci/config "DEFAULT" profile by default.
  # Region is overridable via variable for portability.
  region = var.region
}
