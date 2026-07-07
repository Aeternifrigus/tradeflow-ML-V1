terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- Artifact Registry: stores built Docker images -------------------------
resource "google_artifact_registry_repository" "tradeflow_repo" {
  location      = var.region
  repository_id = var.artifact_repo_name
  format        = "DOCKER"
  description   = "Container images for TradeFlow-ML"
}

# --- Cloud Storage: model artifacts, training data, MLflow artifacts -------
resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-tradeflow-artifacts"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# --- Service account for the Cloud Run service ------------------------------
resource "google_service_account" "tradeflow_run_sa" {
  account_id   = "tradeflow-run-sa"
  display_name = "TradeFlow-ML Cloud Run runtime SA"
}

resource "google_storage_bucket_iam_member" "run_sa_bucket_access" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.tradeflow_run_sa.email}"
}

# --- Cloud Run service -------------------------------------------------------
resource "google_cloud_run_v2_service" "tradeflow_api" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.tradeflow_run_sa.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repo_name}/tradeflow-api:${var.image_tag}"

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "ARTIFACT_BUCKET"
        value = google_storage_bucket.artifacts.name
      }

      ports {
        container_port = 8080
      }
    }
  }

  depends_on = [google_artifact_registry_repository.tradeflow_repo]
}

# Allow unauthenticated invocations (drop this + add IAM bindings for a
# private, auth-required deployment instead)
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  location = google_cloud_run_v2_service.tradeflow_api.location
  name     = google_cloud_run_v2_service.tradeflow_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
