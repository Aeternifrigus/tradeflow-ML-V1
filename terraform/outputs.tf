output "cloud_run_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_v2_service.tradeflow_api.uri
}

output "artifact_bucket_name" {
  value = google_storage_bucket.artifacts.name
}

output "artifact_registry_repo" {
  value = google_artifact_registry_repository.tradeflow_repo.name
}
