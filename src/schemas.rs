use serde::{Deserialize, Serialize};

#[derive(Clone, Copy)]
pub struct GenerationConfig {
    pub temp: f32,
    pub top_k: u32,
    pub top_p: (f32, f32),
    pub dist: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temp: 0.7,
            top_k: 20,
            top_p: (0.8, 1.0),
            dist: 1234,
            n_threads:4,
            n_threads_batch:4,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct GenerationResult {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_time: f64,
    pub ttft: f64,
    pub generation_time: f64,
}

impl GenerationResult {
    pub fn prompt_tokens_per_second(&self) -> f64 {
        self.prompt_tokens as f64
            / (self.prompt_time / 1000.0)
    }

    pub fn generation_tokens_per_second(&self) -> f64 {
        self.generated_tokens as f64
            / (self.generation_time / 1000.0)
    }

    pub fn print_benchmark(&self) {
        println!("----------------------------------------");
        println!("Benchmark");
        println!(
            "Prompt tokens:      {}",
            self.prompt_tokens
        );
        println!(
            "Generated tokens:   {}",
            self.generated_tokens
        );
        println!(
            "Prompt time:        {:.3} ms",
            self.prompt_time
        );
        println!(
            "Generation time:    {:.3} ms",
            self.generation_time
        );
        println!(
            "Generation speed:   {:.2} tokens/sec",
            self.generation_tokens_per_second()
        );
        println!(
            "Prompt speed:       {:.2} tokens/sec",
            self.prompt_tokens_per_second()
        );
        println!("----------------------------------------");
    }
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize
}