use std::error::Error;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    routing::post,
    Json,
    Router,
    debug_handler,
};

use tokio::net::TcpListener;

use tracing::info;

use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{
        params::LlamaModelParams,
        LlamaModel,
    },
};

mod config;
mod inference;
mod schemas;
mod prompts;
mod scheduler;

use scheduler::Node;
use config::AppConfig;
use prompts::*;
use schemas::{
    GenerateRequest,
    GenerationConfig,
    GenerationResult,
};


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {

    tracing_subscriber::fmt()
        .with_line_number(true)
        .with_thread_names(true)
        .init();

    info!("Initializing llama.cpp backend...");
 
    // Configuration 

    let config =
        AppConfig::from_env()?;

    info!(
        "Loading model: {}",
        config.model_path.display()
    );

    // Backend 

    let backend =
        Arc::new(
            LlamaBackend::init()?
        );

    info!("Initialized llama.cpp backend...");

    // Model 

    let model_params =
        LlamaModelParams::default();

    let model =
        Arc::new(
            LlamaModel::load_from_file(
                backend.as_ref(),
                &config.model_path,
                &model_params,
            )?
        );

    info!("Model loaded successfully.");

    // Inference node 

    let generation_config =
        GenerationConfig::default();

    let n_workers: usize = 1;
    
    info!("Creating inference node...");
    let node =
        Arc::new(
            Node::new(
                n_workers,
                Arc::clone(&backend),
                Arc::clone(&model),
                generation_config,
            )?
        );
    
    info!("Inference node created successfully.");

 
    // HTTP server 
    info!("Creating router...");
    let app =
        Router::new()
            .route(
                "/generate",
                post(generate_handler),
            )
            .with_state(node);

    info!("Binding TCP listener...");
    let listener = TcpListener::bind("127.0.0.1:3000").await?;

    info!("Listening on {}",listener.local_addr()?);

    axum::serve(listener, app).await?;

    Ok(())
}

fn get_prompt(data: String) -> String {
    SYSTEM_PROMPT.replace(
        "{}",
        &data,
    )
}

#[debug_handler]
async fn generate_handler(
    State(node): State<Arc<Node>>,
    Json(request): Json<GenerateRequest>,
) -> Result<
    Json<GenerationResult>,
    StatusCode
> {
 
    // Select worker 

    let worker = node.get_idle_worker().ok_or(
                StatusCode::SERVICE_UNAVAILABLE
            )?;
 
    // Build prompt 

    let prompt =
        get_prompt(request.prompt);

    // Send inference request to worker 

    let result =
        worker
            .generate(
                prompt,
                request.max_tokens,
            )
            .await
            .map_err(|error| {

                tracing::error!(
                    "Inference failed: {}",
                    error
                );

                StatusCode::INTERNAL_SERVER_ERROR
            })?;

    Ok(Json(result))
}