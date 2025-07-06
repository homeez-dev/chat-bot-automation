use axum::{
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use serde::Serialize;
use std::collections::HashMap;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    timestamp: String,
    service: String,
    version: String,
}

#[derive(Serialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    message: String,
}

// Health check endpoint
async fn health() -> Result<Json<ApiResponse<HealthResponse>>, StatusCode> {
    let health_data = HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        service: "chat-api".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let response = ApiResponse {
        success: true,
        data: Some(health_data),
        message: "Service is running properly".to_string(),
    };

    Ok(Json(response))
}

// Basic info endpoint
async fn info() -> Result<Json<ApiResponse<HashMap<String, String>>>, StatusCode> {
    let mut info_data = HashMap::new();
    info_data.insert("name".to_string(), "Chat Bot API".to_string());
    info_data.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
    info_data.insert("framework".to_string(), "Axum".to_string());
    info_data.insert("language".to_string(), "Rust".to_string());

    let response = ApiResponse {
        success: true,
        data: Some(info_data),
        message: "API information retrieved successfully".to_string(),
    };

    Ok(Json(response))
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "chat_api=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Build our application with routes
    let app = Router::new()
        .route("/health", get(health))
        .route("/info", get(info))
        .layer(CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http());

    // Run the server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();
    
    tracing::info!("🚀 Chat API server starting on http://0.0.0.0:3000");
    tracing::info!("📊 Health endpoint available at: http://0.0.0.0:3000/health");
    tracing::info!("ℹ️  Info endpoint available at: http://0.0.0.0:3000/info");

    axum::serve(listener, app).await.unwrap();
} 