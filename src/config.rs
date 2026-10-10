use std::env;
use std::path::PathBuf;

pub struct AppConfig {
    pub model_path: PathBuf,
}

impl AppConfig {
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        dotenvy::dotenv().ok();

        let model_path = env::var("MODEL_PATH")?;

        Ok(Self {
            model_path: PathBuf::from(model_path),
        })
    }
}
