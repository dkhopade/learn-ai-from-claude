# --- Security list for the PUBLIC subnet (K8s API endpoint + load balancers) ---
resource "oci_core_security_list" "public" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.project_name}-public-seclist"

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  # Workers -> API server (6443): required for node registration
  ingress_security_rules {
    protocol = "6" # TCP
    source   = "10.0.1.0/24" # private subnet
    tcp_options {
      min = 6443
      max = 6443
    }
  }

  # Workers -> control plane (12250): OKE worker-to-cp communication
  ingress_security_rules {
    protocol = "6"
    source   = "10.0.1.0/24"
    tcp_options {
      min = 12250
      max = 12250
    }
  }

  # Path MTU discovery from workers (ICMP type 3 code 4)
  ingress_security_rules {
    protocol = "1" # ICMP
    source   = "10.0.1.0/24"
    icmp_options {
      type = 3
      code = 4
    }
  }

  # External access to the K8s API endpoint (6443)
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 6443
      max = 6443
    }
  }

  # HTTP/HTTPS for load balancer traffic
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 80
      max = 80
    }
  }
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 443
      max = 443
    }
  }
}

# --- Security list for the PRIVATE subnet (worker nodes) ---
resource "oci_core_security_list" "private" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.project_name}-private-seclist"

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  # All traffic from within the VCN (node-to-node, cp-to-node)
  ingress_security_rules {
    protocol = "all"
    source   = "10.0.0.0/16"
  }

  # API endpoint -> workers (10250 kubelet): required for node registration
  ingress_security_rules {
    protocol = "6"
    source   = "10.0.0.0/24" # public subnet (where API endpoint lives)
    tcp_options {
      min = 10250
      max = 10250
    }
  }

  # Path MTU discovery from control plane
  ingress_security_rules {
    protocol = "1" # ICMP
    source   = "10.0.0.0/24"
    icmp_options {
      type = 3
      code = 4
    }
  }
}
