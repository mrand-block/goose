use crate::conversation::message::Message;
use anyhow::{anyhow, Result};
use mcp_core::tool::ToolCall;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::OnceCell;

use crate::security::model_downloader::{get_global_downloader, ModelInfo};

// ML inference backends
use ort::{session::builder::GraphOptimizationLevel, session::Session};
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct ScanResult {
    pub is_malicious: bool,
    pub confidence: f32,
    pub explanation: String,
}

/// Trait for different ML inference backends
#[async_trait::async_trait]
pub trait PromptInjectionModel: Send + Sync {
    async fn predict(&self, text: &str) -> Result<(f32, String)>;
    fn model_name(&self) -> &str;
    fn get_tokenizer(&self) -> &Arc<Tokenizer>;
}

/// ONNX Runtime implementation
pub struct OnnxPromptInjectionModel {
    session: Arc<std::sync::Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    model_name: String,
}

impl OnnxPromptInjectionModel {
    pub async fn new(
        model_path: PathBuf,
        tokenizer_path: PathBuf,
        model_name: String,
    ) -> Result<Self> {
        tracing::info!("🔒 Starting ONNX model initialization...");

        // Initialize ONNX Runtime session
        tracing::info!("🔒 Creating ONNX session from: {:?}", model_path);
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&model_path)?;

        tracing::info!("🔒 ONNX session created successfully");

        // Load tokenizer
        tracing::info!("🔒 Loading tokenizer from: {:?}", tokenizer_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        tracing::info!("🔒 Tokenizer loaded successfully");

        Ok(Self {
            session: Arc::new(std::sync::Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            model_name,
        })
    }

    /// Log ONNX prediction input to file for analysis
    async fn log_onnx_prediction_to_file(&self, text: &str) {
        use chrono::Utc;
        use std::io::Write;

        let log_path = "/tmp/ml-log.txt";

        // Create log entry with timestamp and full text
        let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S%.3f UTC");
        let separator = "=".repeat(80);
        let log_entry = format!(
            "\n{}\n[{}] ONNX_PREDICT_INPUT - Length: {} chars\n{}\n{}\n{}\n",
            separator,
            timestamp,
            text.len(),
            separator,
            text,
            separator
        );

        // Append to log file (create if doesn't exist)
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
        {
            if let Err(e) = file.write_all(log_entry.as_bytes()) {
                tracing::warn!(
                    "🔒 Failed to write ONNX input to security log file {}: {}",
                    log_path,
                    e
                );
            } else {
                tracing::debug!("🔒 Logged ONNX input to {}", log_path);
            }
        } else {
            tracing::warn!("🔒 Failed to open security log file {}", log_path);
        }
    }
}

#[async_trait::async_trait]
impl PromptInjectionModel for OnnxPromptInjectionModel {
    async fn predict(&self, text: &str) -> Result<(f32, String)> {
        tracing::info!("🔒 ONNX predict() called with text length: {}", text.len());
        tracing::info!(
            "🔒 ONNX predict() received text: '{}'",
            text.chars().take(200).collect::<String>()
        );

        // Log full text to file for analysis
        self.log_onnx_prediction_to_file(text).await;

        // Tokenize the input text
        tracing::debug!("🔒 Tokenizing input text...");
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        tracing::debug!(
            "🔒 Tokenization complete. Sequence length: {}",
            input_ids.len()
        );

        // Convert to the format expected by ONNX (batch_size=1)
        let input_ids: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = attention_mask.iter().map(|&mask| mask as i64).collect();

        let seq_len = input_ids.len();

        tracing::debug!("🔒 Creating ONNX tensors...");
        // Create ONNX tensors
        let input_ids_tensor =
            ort::value::Tensor::from_array(([1, seq_len], input_ids.into_boxed_slice()))?;
        let attention_mask_tensor =
            ort::value::Tensor::from_array(([1, seq_len], attention_mask.into_boxed_slice()))?;

        tracing::debug!("🔒 Running ONNX inference...");
        // Run inference and extract the logits immediately
        let (logit_0, logit_1) = {
            let mut session = self
                .session
                .lock()
                .map_err(|e| anyhow!("Failed to lock session: {}", e))?;
            tracing::debug!("🔒 Session locked, running inference...");
            let outputs = session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor
            ])?;

            tracing::debug!("🔒 Inference complete, extracting logits...");
            // Extract logits from output immediately while we have the lock
            let logits = outputs["logits"].try_extract_tensor::<f32>()?;
            let logits_slice = logits.1;

            // Extract the values we need
            let logit_0 = logits_slice[0]; // Non-injection class
            let logit_1 = logits_slice[1]; // Injection class

            tracing::debug!("🔒 Logits extracted: [{:.3}, {:.3}]", logit_0, logit_1);

            (logit_0, logit_1)
        };

        // Apply softmax to get probabilities
        let exp_0 = logit_0.exp();
        let exp_1 = logit_1.exp();
        let sum_exp = exp_0 + exp_1;

        let prob_injection = exp_1 / sum_exp;

        let explanation = format!(
            "ONNX model '{}': Injection probability = {:.3} (logits: [{:.3}, {:.3}])",
            self.model_name, prob_injection, logit_0, logit_1
        );

        tracing::info!(
            "🔒 ONNX prediction complete: confidence={:.3}, explanation={}",
            prob_injection,
            explanation
        );

        tracing::debug!(
            model = %self.model_name,
            text_length = text.len(),
            seq_length = seq_len,
            logit_0 = logit_0,
            logit_1 = logit_1,
            prob_injection = prob_injection,
            "ONNX inference completed"
        );

        Ok((prob_injection, explanation))
    }

    fn get_tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

type ModelCache = Arc<tokio::sync::RwLock<Option<Arc<dyn PromptInjectionModel>>>>;

/// Global model cache with reload capability
static MODEL_CACHE: OnceCell<ModelCache> = OnceCell::const_new();

/// Initialize the global model
async fn initialize_model() -> Result<Option<Arc<dyn PromptInjectionModel>>> {
    tracing::info!("🔒 Attempting to initialize ONNX security model...");

    // Try to load the ONNX model
    match get_global_downloader().await {
        Ok(downloader) => {
            let model_info = PromptInjectionScanner::get_model_info_from_config();
            match downloader.ensure_model_available(&model_info).await {
                Ok((model_path, tokenizer_path)) => {
                    tracing::info!("🔒 Loading ONNX model from: {:?}", model_path);
                    match OnnxPromptInjectionModel::new(
                        model_path,
                        tokenizer_path,
                        model_info.hf_model_name.clone(),
                    )
                    .await
                    {
                        Ok(model) => {
                            tracing::info!("🔒 ✅ ONNX security model loaded successfully");
                            return Ok(Some(Arc::new(model)));
                        }
                        Err(e) => {
                            tracing::warn!("🔒 Failed to initialize ONNX model: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("🔒 Failed to ensure model available: {}", e);
                }
            }
        }
        Err(e) => {
            tracing::warn!("🔒 Failed to get model downloader: {}", e);
        }
    }

    tracing::info!("🔒 ONNX model not available, will use pattern-based scanning");
    Ok(None)
}

/// Get or initialize the global model
async fn get_model() -> Option<Arc<dyn PromptInjectionModel>> {
    let cache = MODEL_CACHE
        .get_or_init(|| async { Arc::new(tokio::sync::RwLock::new(None)) })
        .await;

    // Check if model is already loaded in memory
    let read_guard = cache.read().await;
    if let Some(model) = read_guard.as_ref() {
        tracing::debug!("🔒 Model found in memory cache, using cached instance");
        return Some(model.clone());
    }
    drop(read_guard);

    // Model not loaded in memory, try to initialize from disk
    tracing::info!("🔒 Model not loaded in memory, loading from disk cache...");
    let mut write_guard = cache.write().await;

    // Double-check in case another task loaded it while we were waiting
    if let Some(model) = write_guard.as_ref() {
        tracing::debug!("🔒 Model was loaded by another task while waiting, using that instance");
        return Some(model.clone());
    }

    // Load the model from disk
    match initialize_model().await {
        Ok(Some(model)) => {
            tracing::info!("🔒 ✅ Model successfully loaded into memory cache");
            *write_guard = Some(model.clone());
            Some(model)
        }
        Ok(None) => {
            tracing::info!("🔒 No model available, using pattern-based fallback");
            None
        }
        Err(e) => {
            tracing::warn!("🔒 Failed to initialize model: {}", e);
            None
        }
    }
}

/// Check if model is available without triggering download/initialization
pub async fn get_model_if_available() -> Option<Arc<dyn PromptInjectionModel>> {
    let cache = MODEL_CACHE
        .get_or_init(|| async { Arc::new(tokio::sync::RwLock::new(None)) })
        .await;

    // Only check if model is already loaded in memory - don't trigger loading
    let read_guard = cache.read().await;
    read_guard.as_ref().cloned()
}

/// Simple prompt injection scanner
/// Uses the existing model_downloader infrastructure
pub struct PromptInjectionScanner {
    enabled: bool,
}

impl PromptInjectionScanner {
    pub fn new() -> Self {
        println!("🔒 PromptInjectionScanner::new() called");

        // Check if models are available, trigger download if needed
        let scanner = Self {
            enabled: Self::check_and_prepare_models(),
        };

        println!("🔒 Scanner enabled: {}", scanner.enabled);

        scanner
    }

    /// Check if models are available and trigger download if needed
    fn check_and_prepare_models() -> bool {
        // Check if models are already cached
        let model_info = Self::get_model_info_from_config();

        // Check if model files exist in cache
        if let Some(cache_dir) = dirs::cache_dir() {
            let security_models_dir = cache_dir.join("goose").join("security_models");
            let model_path = security_models_dir.join(&model_info.onnx_filename);
            let tokenizer_path = security_models_dir.join(&model_info.tokenizer_filename);

            if model_path.exists() && tokenizer_path.exists() {
                tracing::info!(
                    "🔒 Security model files found on disk - loading model into memory now..."
                );

                // Load model into memory immediately at startup
                tokio::spawn(async move {
                    tracing::info!("🔒 Pre-loading security model at startup...");
                    if let Some(_model) = get_model().await {
                        tracing::info!(
                            "🔒 ✅ Security model pre-loaded successfully - ready for scanning"
                        );
                    } else {
                        tracing::warn!("🔒 Failed to pre-load security model");
                    }
                });

                return true;
            }
        }

        // Models not cached - we need to download them
        tracing::info!("🔒 Security model files not found on disk");
        tracing::info!(
            "🔒 Models will be downloaded on first security scan (this may cause a delay)"
        );

        // For now, return true to enable security scanning
        // The models will be downloaded lazily on first scan
        // TODO: Consider blocking startup to download models synchronously
        true
    }

    /// Get model information from config file
    pub fn get_model_info_from_config() -> ModelInfo {
        use crate::config::Config;
        let config = Config::global();

        // Try to get model from config
        let security_config = config.get_param::<serde_json::Value>("security").ok();

        let model_name = security_config
            .as_ref()
            .and_then(|security| security.get("models")?.as_array()?.first())
            .and_then(|model| model.get("model")?.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                tracing::warn!(
                    "🔒 No security model configured, security scanning will be disabled"
                );
                // Return a placeholder that won't work, forcing pattern-only mode
                "no-model-configured".to_string()
            });

        tracing::info!("🔒 Using security model from config: {}", model_name);

        // Create ModelInfo from config
        ModelInfo::from_config_model(&model_name)
    }

    /// Two-step security analysis: scan tool call first, then analyze context if suspicious
    pub async fn analyze_tool_call_with_context(
        &self,
        tool_call: &ToolCall,
        messages: &[Message],
    ) -> Result<ScanResult> {
        // Step 1: Scan the tool call itself for suspicious patterns
        let tool_call_result = self.scan_tool_call_only(tool_call).await?;

        if !tool_call_result.is_malicious {
            // Tool call looks safe, no need for context analysis
            tracing::debug!(
                tool_name = %tool_call.name,
                confidence = tool_call_result.confidence,
                "✅ Tool call passed initial security scan"
            );
            return Ok(tool_call_result);
        }

        // Step 2: Tool call looks suspicious, analyze conversation context
        tracing::info!(
            tool_name = %tool_call.name,
            confidence = tool_call_result.confidence,
            "🔍 Tool call flagged as suspicious, analyzing conversation context"
        );

        let user_messages_result = self.scan_user_messages_only(messages).await?;

        // Decision logic: combine both results
        let final_result =
            self.make_final_security_decision(&tool_call_result, &user_messages_result, tool_call);

        tracing::info!(
            tool_name = %tool_call.name,
            tool_confidence = tool_call_result.confidence,
            conversation_confidence = user_messages_result.confidence,
            final_malicious = final_result.is_malicious,
            final_confidence = final_result.confidence,
            "🔒 Two-step security analysis complete"
        );

        Ok(final_result)
    }

    /// Step 1: Scan only the tool call for suspicious patterns
    async fn scan_tool_call_only(&self, tool_call: &ToolCall) -> Result<ScanResult> {
        // Debug: Log the raw tool call arguments first
        tracing::info!("🔒 Raw tool call arguments: {:?}", tool_call.arguments);

        // Create text representation of the tool call for analysis
        let arguments_json =
            serde_json::to_string_pretty(&tool_call.arguments).unwrap_or_else(|e| {
                tracing::warn!("🔒 Failed to serialize tool arguments: {}", e);
                format!("{{\"error\": \"Failed to serialize arguments: {}\"}}", e)
            });

        let tool_text = format!("Tool: {}\nArguments: {}", tool_call.name, arguments_json);

        tracing::info!(
            "🔒 Complete tool text being analyzed (length: {}): '{}'",
            tool_text.len(),
            tool_text
        );

        self.scan_with_prompt_injection_model(&tool_text).await
    }

    /// Step 2: Scan only the user messages (conversation history) for prompt injection
    async fn scan_user_messages_only(&self, messages: &[Message]) -> Result<ScanResult> {
        // Extract only user messages from recent conversation history
        let user_messages: Vec<String> = messages
            .iter()
            .rev()
            .take(5) // Take last 5 messages for context
            .rev()
            .filter_map(|msg| {
                // Only analyze user messages, not assistant responses
                if matches!(msg.role, rmcp::model::Role::User) {
                    msg.content.first()?.as_text().map(|text| text.to_string())
                } else {
                    None
                }
            })
            .collect();

        if user_messages.is_empty() {
            return Ok(ScanResult {
                is_malicious: false,
                confidence: 0.0,
                explanation: "No user messages found in conversation history".to_string(),
            });
        }

        let user_context = user_messages.join("\n\n");
        self.scan_with_prompt_injection_model(&user_context).await
    }

    /// Make final security decision based on both tool call and user message analysis
    fn make_final_security_decision(
        &self,
        tool_call_result: &ScanResult,
        user_messages_result: &ScanResult,
        _tool_call: &ToolCall,
    ) -> ScanResult {
        // Simple decision logic using config threshold:
        // 1. If user messages contain prompt injection, tool call is likely malicious
        // 2. Otherwise, use the tool call confidence as-is
        // 3. Let the config threshold determine if user should be asked

        let (confidence, explanation) = if user_messages_result.is_malicious {
            // User messages contain prompt injection - tool call is likely malicious
            let combined_confidence =
                (tool_call_result.confidence + user_messages_result.confidence) / 2.0;
            let explanation =
                "Tool appears to be the result of a prompt injection attack.".to_string();
            (combined_confidence, explanation)
        } else {
            // Use tool call confidence as-is, let config threshold decide
            let explanation = if tool_call_result.confidence > 0.0 {
                format!(
                    "Tool flagged with confidence: {:.2}",
                    tool_call_result.confidence
                )
            } else {
                "Tool appears safe".to_string()
            };
            (tool_call_result.confidence, explanation)
        };

        // Get threshold from config to determine if malicious
        let config_threshold = self.get_threshold_from_config();
        let is_malicious = confidence > config_threshold;

        ScanResult {
            is_malicious,
            confidence,
            explanation,
        }
    }

    /// Scan system prompt for persistent injection attacks
    pub async fn scan_system_prompt(&self, system_prompt: &str) -> Result<ScanResult> {
        tracing::info!(
            "🔒 Scanning system prompt for persistent injection attacks (length: {})",
            system_prompt.len()
        );

        // Use the ML model to scan the system prompt - this is what we have the model for!
        self.scan_with_prompt_injection_model(system_prompt).await
    }

    /// Model-agnostic prompt injection scanning with chunking for long texts
    pub async fn scan_with_prompt_injection_model(&self, text: &str) -> Result<ScanResult> {
        tracing::info!(
            "🔒 Starting scan_with_prompt_injection_model for text (length: {}): '{}'",
            text.len(),
            text
        );

        // Always run pattern-based scanning first
        let pattern_result = self.scan_with_patterns(text).await?;

        // Try to get the ML model for additional scanning
        tracing::info!("🔒 Attempting to get ML model...");
        if let Some(model) = get_model().await {
            tracing::info!("🔒 ML model retrieved successfully, calling predict...");

            // Use single chunk scanning for general text (not tool responses)
            let ml_result = self.scan_single_chunk(text, &model).await?;

            // Combine ML and pattern results
            let combined_result = self.combine_scan_results(
                &pattern_result,
                ml_result.confidence,
                &ml_result.explanation,
                ml_result.is_malicious,
            );

            tracing::info!(
                "🔒 Combined scan result: ML confidence={:.3}, Pattern confidence={:.3}, Final confidence={:.3}, Final malicious={}",
                ml_result.confidence, pattern_result.confidence, combined_result.confidence, combined_result.is_malicious
            );

            return Ok(combined_result);
        } else {
            tracing::info!("🔒 No ML model available, using pattern-based scanning only");
        }

        tracing::info!("🔒 Using pattern-based scan result only");
        Ok(pattern_result)
    }

    /// Scan tool response content with normalization and 512-token chunking
    pub async fn scan_tool_response_content(&self, text: &str) -> Result<ScanResult> {
        tracing::info!(
            "🔒 Starting tool response content scan for text (length: {} chars): '{}'",
            text.len(),
            text.chars().take(200).collect::<String>()
        );

        // Log full content to file for analysis
        self.log_security_scan_to_file("ORIGINAL", text).await;

        // Step 1: Normalize the text
        let normalized_text = self.normalize_text(text);

        tracing::info!(
            "🔒 Text normalization complete: original {} chars → normalized {} chars, preview: '{}'",
            text.len(),
            normalized_text.len(),
            normalized_text.chars().take(200).collect::<String>()
        );

        // Log normalized content to file for analysis
        self.log_security_scan_to_file("NORMALIZED", &normalized_text)
            .await;

        // Step 2: Check if normalized text is empty
        if normalized_text.trim().is_empty() {
            tracing::info!("🔒 Normalized text is empty, skipping security scan");
            return Ok(ScanResult {
                is_malicious: false,
                confidence: 0.0,
                explanation: "Skipped scan: normalized text is empty".to_string(),
            });
        }

        // Step 3: Always run pattern-based scanning first on original text
        let pattern_result = self.scan_with_patterns(text).await?;

        // Step 4: Try to get the ML model for additional scanning
        tracing::info!("🔒 Attempting to get ML model for normalized text scanning...");
        if let Some(model) = get_model().await {
            tracing::info!("🔒 ML model retrieved successfully, starting 512-token chunking...");

            // Use 512-token chunking on normalized text
            let ml_result = self
                .scan_with_normalization_and_chunking(&normalized_text, &model)
                .await?;

            // Combine ML and pattern results
            let combined_result = self.combine_scan_results(
                &pattern_result,
                ml_result.confidence,
                &ml_result.explanation,
                ml_result.is_malicious,
            );

            tracing::info!(
                "🔒 Tool response scan complete: ML confidence={:.3}, Pattern confidence={:.3}, Final confidence={:.3}, Final malicious={}",
                ml_result.confidence, pattern_result.confidence, combined_result.confidence, combined_result.is_malicious
            );

            return Ok(combined_result);
        } else {
            tracing::info!("🔒 No ML model available, using pattern-based scanning only");
        }

        tracing::info!("🔒 Using pattern-based scan result only for tool response");
        Ok(pattern_result)
    }

    /// Level 15: N-gram deduplication normalization
    /// Apply ultra-aggressive word deduplication, then remove repetitive 2-grams and 3-grams
    fn normalize_text(&self, text: &str) -> String {
        self.level_15_ngram_deduplication_normalization(text)
    }

    /// Level 15: N-gram deduplication normalization
    /// Apply Level 14's word deduplication, then remove repetitive N-grams (2-grams, 3-grams).
    /// This targets repetitive phrases like "kgoose service src" appearing multiple times.
    ///
    /// Strategy:
    /// 1. Replace special characters with spaces to properly separate tokens
    /// 2. Apply ultra-aggressive word deduplication: remove ANY word that appears more than once
    /// 3. Apply N-gram deduplication: remove repetitive 2-grams and 3-grams
    /// 4. Final cleanup and space collapse (no spaces in final output)
    fn level_15_ngram_deduplication_normalization(&self, text: &str) -> String {
        use regex::Regex;

        if text.trim().is_empty() {
            return text.to_string();
        }

        let mut normalized = text.to_string();

        // Step 1: Pre-normalization - replace special chars with spaces to separate tokens properly
        // Remove permission strings first
        let perm_regex = Regex::new(r"-[rwx-]{9}").unwrap();
        normalized = perm_regex.replace_all(&normalized, "").to_string();

        // Remove directory permissions
        let dir_perm_regex = Regex::new(r"drwx[rwx-]*").unwrap();
        normalized = dir_perm_regex.replace_all(&normalized, "").to_string();

        // Remove file sizes
        let size_regex = Regex::new(r"\b\d+[BKMG]?\b").unwrap();
        normalized = size_regex.replace_all(&normalized, "").to_string();

        // Remove timestamps
        let timestamp_regex = Regex::new(r"\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}\b").unwrap();
        normalized = timestamp_regex.replace_all(&normalized, "").to_string();

        // Remove dates
        let date_regex = Regex::new(r"\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{4}\b").unwrap();
        normalized = date_regex.replace_all(&normalized, "").to_string();

        // Remove "total 104" patterns
        let total_regex = Regex::new(r"\btotal\s+\d+\b").unwrap();
        normalized = total_regex.replace_all(&normalized, "").to_string();

        // Replace special characters and numbers with spaces (but keep letters)
        let special_char_regex = Regex::new(r"[^a-zA-Z\s]").unwrap();
        normalized = special_char_regex.replace_all(&normalized, " ").to_string();

        // Normalize whitespace
        let whitespace_regex = Regex::new(r"\s+").unwrap();
        normalized = whitespace_regex
            .replace_all(&normalized, " ")
            .trim()
            .to_string();

        // Step 2: Ultra-aggressive word deduplication - remove ANY word that appears more than once
        let deduplicated_text = self.ultra_aggressive_deduplication(&normalized);

        // Step 3: N-gram deduplication - remove repetitive phrases
        let ngram_deduplicated_text = self.ngram_deduplication(&deduplicated_text);

        // Step 4: Final cleanup - collapse spaces (back to no spaces like Level 14)
        let final_text = whitespace_regex
            .replace_all(&ngram_deduplicated_text, "")
            .to_string();

        if final_text.is_empty() {
            "EMPTY_CONTENT".to_string()
        } else {
            final_text
        }
    }

    /// Ultra-aggressive deduplication: Remove ANY word that appears more than once.
    /// Keep only the first occurrence of every word.
    fn ultra_aggressive_deduplication(&self, text: &str) -> String {
        use regex::Regex;
        use std::collections::HashSet;

        // Split into words for analysis
        let word_regex = Regex::new(r"\w+").unwrap();
        let words: Vec<&str> = word_regex.find_iter(text).map(|m| m.as_str()).collect();

        if words.len() < 2 {
            return text.to_string();
        }

        // Ultra-simple approach: remove ANY word that appears more than once
        let mut filtered_words = Vec::new();
        let mut seen_words = HashSet::new();
        let word_count = words.len();

        for word in words {
            if !seen_words.contains(word) {
                filtered_words.push(word);
                seen_words.insert(word);
            }
            // Skip all subsequent occurrences of this word
        }

        tracing::debug!(
            "🔒 Ultra-aggressive deduplication: {} words -> {} words",
            word_count,
            filtered_words.len()
        );

        filtered_words.join(" ")
    }

    /// Remove repetitive N-grams (2-grams and 3-grams) from text.
    /// This targets repetitive phrases like "kgoose service src" appearing multiple times.
    fn ngram_deduplication(&self, text: &str) -> String {
        use std::collections::{HashMap, HashSet};

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < 3 {
            return text.to_string();
        }

        // Step 1: Remove repetitive 2-grams (bigrams)
        let mut bigrams = Vec::new();
        for i in 0..words.len().saturating_sub(1) {
            let bigram = format!("{} {}", words[i], words[i + 1]);
            bigrams.push((bigram, i));
        }

        // Count bigram frequencies
        let mut bigram_counts = HashMap::new();
        for (bigram, _) in &bigrams {
            *bigram_counts.entry(bigram.clone()).or_insert(0) += 1;
        }

        // Mark positions to remove (keep only first occurrence of repetitive bigrams)
        let mut positions_to_remove = HashSet::new();
        let mut seen_repetitive_bigrams = HashSet::new();

        for (bigram, pos) in bigrams {
            if bigram_counts[&bigram] > 1 {
                // Repetitive bigram
                if seen_repetitive_bigrams.contains(&bigram) {
                    // Remove this occurrence (mark both positions)
                    positions_to_remove.insert(pos);
                    positions_to_remove.insert(pos + 1);
                } else {
                    // Keep first occurrence
                    seen_repetitive_bigrams.insert(bigram);
                }
            }
        }

        // Step 2: Remove repetitive 3-grams (trigrams) from remaining words
        let remaining_words: Vec<&str> = words
            .iter()
            .enumerate()
            .filter(|(i, _)| !positions_to_remove.contains(i))
            .map(|(_, word)| *word)
            .collect();

        let final_words = if remaining_words.len() >= 3 {
            let mut trigrams = Vec::new();
            for i in 0..remaining_words.len().saturating_sub(2) {
                let trigram = format!(
                    "{} {} {}",
                    remaining_words[i],
                    remaining_words[i + 1],
                    remaining_words[i + 2]
                );
                trigrams.push((trigram, i));
            }

            // Count trigram frequencies
            let mut trigram_counts = HashMap::new();
            for (trigram, _) in &trigrams {
                *trigram_counts.entry(trigram.clone()).or_insert(0) += 1;
            }

            // Mark additional positions to remove
            let mut additional_positions_to_remove = HashSet::new();
            let mut seen_repetitive_trigrams = HashSet::new();

            for (trigram, pos) in trigrams {
                if trigram_counts[&trigram] > 1 {
                    // Repetitive trigram
                    if seen_repetitive_trigrams.contains(&trigram) {
                        // Remove this occurrence (mark all three positions)
                        additional_positions_to_remove.insert(pos);
                        additional_positions_to_remove.insert(pos + 1);
                        additional_positions_to_remove.insert(pos + 2);
                    } else {
                        // Keep first occurrence
                        seen_repetitive_trigrams.insert(trigram);
                    }
                }
            }

            // Apply trigram removal
            remaining_words
                .iter()
                .enumerate()
                .filter(|(i, _)| !additional_positions_to_remove.contains(i))
                .map(|(_, word)| *word)
                .collect()
        } else {
            remaining_words
        };

        tracing::debug!(
            "🔒 N-gram deduplication: {} words -> {} words",
            words.len(),
            final_words.len()
        );

        final_words.join(" ")
    }

    /// Optimized context-aware repetitive pattern removal using statistical and linguistic
    /// patterns instead of hardcoded word lists. This approach is agnostic to content domain.
    fn optimized_repetitive_pattern_removal(&self, text: &str) -> String {
        use regex::Regex;
        use std::collections::{HashMap, HashSet};

        // Split into words for analysis
        let word_regex = Regex::new(r"\w+").unwrap();
        let words: Vec<&str> = word_regex.find_iter(text).map(|m| m.as_str()).collect();

        if words.len() < 3 {
            return text.to_string();
        }

        // Analyze word patterns
        let mut word_counts = HashMap::new();
        for word in &words {
            *word_counts.entry(*word).or_insert(0) += 1;
        }

        let total_words = words.len() as f64;
        let unique_words = word_counts.len() as f64;

        // Statistical indicators for content type detection
        let short_words = words.iter().filter(|w| w.len() <= 4).count() as f64;
        let long_words = words.iter().filter(|w| w.len() >= 8).count() as f64;
        let very_long_words = words.iter().filter(|w| w.len() >= 12).count() as f64;

        let short_word_ratio = if total_words > 0.0 {
            short_words / total_words
        } else {
            0.0
        };
        let long_word_ratio = if total_words > 0.0 {
            long_words / total_words
        } else {
            0.0
        };
        let very_long_word_ratio = if total_words > 0.0 {
            very_long_words / total_words
        } else {
            0.0
        };

        // Repetition analysis
        let _highly_repeated_words =
            word_counts.values().filter(|&&count| count >= 5).count() as f64;
        let _moderately_repeated_words =
            word_counts.values().filter(|&&count| count >= 3).count() as f64;

        // Diversity metrics
        let lexical_diversity = if total_words > 0.0 {
            unique_words / total_words
        } else {
            0.0
        };
        let repetition_density = if total_words > 0.0 {
            (total_words - unique_words) / total_words
        } else {
            0.0
        };

        // Content type classification based on statistical patterns
        // High repetition + low diversity + many short words = likely structured/filesystem data
        let structured_data_score = (repetition_density * 2.0)
            + (1.0 - lexical_diversity)
            + short_word_ratio
            + (very_long_word_ratio * 0.5);

        // High diversity + balanced word lengths = likely natural language
        let natural_language_score = lexical_diversity
            + (1.0 - repetition_density)
            + (1.0 - short_word_ratio)
            + (long_word_ratio * 0.5);

        tracing::debug!(
            "🔒 Content analysis: structured_score={:.3}, natural_score={:.3}, lexical_diversity={:.3}, repetition_density={:.3}",
            structured_data_score, natural_language_score, lexical_diversity, repetition_density
        );

        // Determine content type and apply appropriate strategy
        let (repetitive_threshold, keep_occurrences) =
            if structured_data_score > natural_language_score && structured_data_score > 1.5 {
                // Structured/filesystem-like content - aggressive deduplication
                tracing::debug!(
                    "🔒 Detected structured/filesystem content - applying aggressive deduplication"
                );
                (2, 1)
            } else if natural_language_score > structured_data_score && natural_language_score > 1.2
            {
                // Natural language content - conservative deduplication
                tracing::debug!(
                    "🔒 Detected natural language content - applying conservative deduplication"
                );
                (6, 3)
            } else {
                // Mixed or uncertain content - balanced approach
                tracing::debug!(
                    "🔒 Detected mixed/uncertain content - applying balanced deduplication"
                );
                (4, 2)
            };

        // Apply deduplication based on determined strategy
        let repetitive_words: HashSet<&str> = word_counts
            .iter()
            .filter(|(&word, &count)| count >= repetitive_threshold && word.len() >= 3)
            .map(|(&word, _)| word)
            .collect();

        tracing::debug!(
            "🔒 Found {} repetitive words with threshold {}, keeping {} occurrences each",
            repetitive_words.len(),
            repetitive_threshold,
            keep_occurrences
        );

        let mut filtered_words = Vec::new();
        let mut word_usage = HashMap::new();

        for word in &words {
            if repetitive_words.contains(word) {
                let usage = word_usage.entry(*word).or_insert(0);
                *usage += 1;
                if *usage <= keep_occurrences {
                    filtered_words.push(*word);
                }
                // Skip subsequent occurrences
            } else {
                filtered_words.push(*word);
            }
        }

        let mut result = filtered_words.join(" ");

        tracing::debug!(
            "🔒 After word deduplication: {} words -> {} words",
            words.len(),
            filtered_words.len()
        );

        // Character-level pattern removal (applies to all content types)
        let original_len = result.len();
        for pattern_len in 3..std::cmp::min(12, result.len() / 3) {
            let pattern = format!(r"(\w{{{}}})(\1){{2,}}", pattern_len);
            if let Ok(pattern_regex) = Regex::new(&pattern) {
                result = pattern_regex.replace_all(&result, "$1").to_string();
            }
        }

        tracing::debug!(
            "🔒 After character-level deduplication: {} chars -> {} chars",
            original_len,
            result.len()
        );

        result
    }

    /// Log security scan data to file for analysis
    async fn log_security_scan_to_file(&self, stage: &str, content: &str) {
        use chrono::Utc;
        use std::io::Write;

        let log_path = "/tmp/ml-log.txt";

        // Create log entry with timestamp and stage
        let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S%.3f UTC");
        let separator = "=".repeat(80);
        let log_entry = format!(
            "\n{}\n[{}] {} - Length: {} chars\n{}\n{}\n{}\n",
            separator,
            timestamp,
            stage,
            content.len(),
            separator,
            content,
            separator
        );

        // Append to log file (create if doesn't exist)
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
        {
            if let Err(e) = file.write_all(log_entry.as_bytes()) {
                tracing::warn!(
                    "🔒 Failed to write to security log file {}: {}",
                    log_path,
                    e
                );
            } else {
                tracing::debug!("🔒 Logged {} content to {}", stage, log_path);
            }
        } else {
            tracing::warn!("🔒 Failed to open security log file {}", log_path);
        }
    }

    /// Scan normalized text with 512-token chunking strategy
    async fn scan_with_normalization_and_chunking(
        &self,
        normalized_text: &str,
        model: &Arc<dyn PromptInjectionModel>,
    ) -> Result<ScanResult> {
        const MAX_TOKENS_PER_CHUNK: usize = 512;

        tracing::info!(
            "🔒 Starting 512-token chunking scan for normalized text (length: {} chars)",
            normalized_text.len()
        );

        // Use the model's tokenizer
        let tokenizer = model.get_tokenizer();

        // Tokenize the normalized text
        let encoding = tokenizer
            .encode(normalized_text, false)
            .map_err(|e| anyhow!("Failed to tokenize normalized text: {}", e))?;

        let tokens = encoding.get_ids();
        let total_tokens = tokens.len();

        tracing::info!(
            "🔒 Normalized text tokenized: {} total tokens, max {} tokens per chunk",
            total_tokens,
            MAX_TOKENS_PER_CHUNK
        );

        // If text fits in single chunk, scan directly
        if total_tokens <= MAX_TOKENS_PER_CHUNK {
            tracing::info!(
                "🔒 Normalized text fits in single chunk ({} tokens), scanning directly",
                total_tokens
            );
            return self.scan_single_chunk(normalized_text, model).await;
        }

        // Generate sequential chunks (no overlap)
        let chunks = self.generate_sequential_chunks(tokens, tokenizer, MAX_TOKENS_PER_CHUNK)?;

        tracing::info!(
            "🔒 Generated {} sequential chunks of max {} tokens each",
            chunks.len(),
            MAX_TOKENS_PER_CHUNK
        );

        // Scan each chunk and look for any malicious content
        let threshold = self.get_threshold_from_config();

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let chunk_num = chunk_idx + 1;

            tracing::info!(
                "🔒 Scanning chunk {} of {}: {} tokens, {} chars, preview: '{}'",
                chunk_num,
                chunks.len(),
                chunk.token_count,
                chunk.text.len(),
                chunk.text.chars().take(100).collect::<String>()
            );

            match self.scan_single_chunk(&chunk.text, model).await {
                Ok(chunk_result) => {
                    tracing::info!(
                        "🔒 Chunk {} result: confidence={:.3}, threshold={:.3}, malicious={}",
                        chunk_num,
                        chunk_result.confidence,
                        threshold,
                        chunk_result.is_malicious
                    );

                    // If ANY chunk is flagged as malicious, trigger security warning
                    if chunk_result.is_malicious {
                        tracing::warn!(
                            "🔒 Malicious content detected in chunk {} of {}: confidence={:.3}",
                            chunk_num,
                            chunks.len(),
                            chunk_result.confidence
                        );

                        return Ok(ScanResult {
                            is_malicious: true,
                            confidence: chunk_result.confidence,
                            explanation: format!(
                                "Normalized 512-token scan: Malicious content detected in chunk {} of {} (tokens {}-{}): {}",
                                chunk_num, chunks.len(), chunk.start_token, chunk.end_token, chunk_result.explanation
                            ),
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!("🔒 Failed to scan chunk {}: {}", chunk_num, e);
                    // Continue with other chunks even if one fails
                }
            }
        }

        // All chunks passed - content is safe
        tracing::info!(
            "🔒 All {} chunks passed security scan - content is safe",
            chunks.len()
        );

        Ok(ScanResult {
            is_malicious: false,
            confidence: 0.0,
            explanation: format!(
                "Normalized 512-token scan: Analyzed {} sequential chunks ({} tokens total), no threats detected",
                chunks.len(), total_tokens
            ),
        })
    }

    /// Generate sequential chunks with no overlap for 512-token scanning
    fn generate_sequential_chunks(
        &self,
        tokens: &[u32],
        tokenizer: &Arc<Tokenizer>,
        chunk_size: usize,
    ) -> Result<Vec<SequentialChunk>> {
        let mut chunks = Vec::new();
        let total_tokens = tokens.len();
        let mut start_token = 0;

        while start_token < total_tokens {
            let end_token = std::cmp::min(start_token + chunk_size, total_tokens);

            // Extract token slice and decode back to text
            let chunk_tokens = &tokens[start_token..end_token];
            let chunk_text = tokenizer
                .decode(chunk_tokens, true)
                .map_err(|e| anyhow!("Failed to decode chunk tokens: {}", e))?;

            chunks.push(SequentialChunk {
                start_token,
                end_token,
                token_count: chunk_tokens.len(),
                text: chunk_text,
            });

            // Move to next chunk (no overlap)
            start_token = end_token;
        }

        Ok(chunks)
    }

    // COMMENTED OUT: Sliding window approach - keeping for reference
    /*
    /// Scan text with ML model using mathematically guaranteed sliding window approach
    async fn scan_with_ml_model_chunked(
        &self,
        text: &str,
        model: &Arc<dyn PromptInjectionModel>,
    ) -> Result<ScanResult> {
        // Use the already-loaded tokenizer from the model
        let tokenizer = model.get_tokenizer();

        // Sliding window parameters for mathematical guarantee
        const WINDOW_SIZE: usize = 32; // Maximum model capacity
        const DEFAULT_STRIDE: usize = 16;
        const HIGH_SECURITY_STRIDE: usize = 8;

        tracing::info!(
            "🔒 Starting sliding window ML scanning for text length: {} characters",
            text.len()
        );

        // Tokenize the full text first
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("Failed to tokenize text: {}", e))?;

        let tokens = encoding.get_ids();
        let total_tokens = tokens.len();

        // Adaptive stride selection based on document size and security needs
        let stride = if total_tokens < 1000 {
            HIGH_SECURITY_STRIDE // High security for short documents
        } else {
            DEFAULT_STRIDE // Balanced approach for longer documents
        };

        let max_detectable_attack_length = WINDOW_SIZE - stride;

        tracing::info!(
            "🔒 Text tokenized: {} tokens total, window size: {} tokens, stride: {} tokens, max detectable attack: {} tokens",
            total_tokens, WINDOW_SIZE, stride, max_detectable_attack_length
        );

        // For texts that fit in a single window, scan directly
        if total_tokens <= WINDOW_SIZE {
            tracing::info!(
                "🔒 Text fits in single window ({} tokens), scanning directly",
                total_tokens
            );
            return self.scan_single_chunk(text, model).await;
        }

        // Generate sliding windows with mathematical guarantee
        let windows = self.generate_sliding_windows(tokens, tokenizer, WINDOW_SIZE, stride)?;

        tracing::info!(
            "🔒 Generated {} sliding windows with stride {} (guarantees detection of attacks up to {} tokens)",
            windows.len(), stride, max_detectable_attack_length
        );

        // Scan windows with early termination on high confidence threats
        let mut max_confidence = 0.0;
        let mut threat_chunks = Vec::new();
        let emergency_threshold = 0.9; // Stop immediately on very high confidence

        for (window_idx, window) in windows.iter().enumerate() {
            let window_num = window_idx + 1;

            tracing::debug!(
                "🔒 Processing window {} of {}: {} tokens, {} chars",
                window_num,
                windows.len(),
                window.token_count,
                window.text.len()
            );

            // Scan this window
            match self.scan_single_chunk(&window.text, model).await {
                Ok(window_result) => {
                    tracing::debug!(
                        "🔒 Window {} result: confidence={:.3}, malicious={}",
                        window_num,
                        window_result.confidence,
                        window_result.is_malicious
                    );

                    // Track maximum confidence across all windows
                    if window_result.confidence > max_confidence {
                        max_confidence = window_result.confidence;
                    }

                    // Early termination on emergency-level threats
                    if window_result.confidence > emergency_threshold {
                        tracing::warn!(
                            "🔒 Emergency threshold exceeded in window {}: confidence={:.3}, terminating scan early",
                            window_num, window_result.confidence
                        );
                        return Ok(ScanResult {
                            is_malicious: true,
                            confidence: window_result.confidence,
                            explanation: format!(
                                "Sliding window scan: Emergency threat detected in window {} of {} (tokens {}-{}): {}",
                                window_num, windows.len(), window.start_token, window.end_token, window_result.explanation
                            ),
                        });
                    }

                    // Collect information about threatening windows
                    if window_result.is_malicious {
                        threat_chunks.push(format!(
                            "Window {} (tokens {}-{}): confidence={:.3}",
                            window_num,
                            window.start_token,
                            window.end_token,
                            window_result.confidence
                        ));
                    }
                }
                Err(e) => {
                    tracing::warn!("🔒 Failed to scan window {}: {}", window_num, e);
                    // Continue with other windows even if one fails
                }
            }
        }

        // Aggregate results
        let threshold = self.get_threshold_from_config();
        let is_malicious = max_confidence > threshold;

        let explanation = if threat_chunks.is_empty() {
            format!(
                "Sliding window scan: Analyzed {} overlapping windows ({} tokens total, stride {}), max confidence {:.3}, no threats detected",
                windows.len(), total_tokens, stride, max_confidence
            )
        } else {
            format!(
                "Sliding window scan: Analyzed {} overlapping windows ({} tokens total, stride {}), threats detected in {} windows: {}",
                windows.len(), total_tokens, stride, threat_chunks.len(), threat_chunks.join("; ")
            )
        };

        tracing::info!(
            "🔒 Sliding window scan complete: {} windows with stride {}, max_confidence={:.3}, threshold={:.3}, malicious={}",
            windows.len(),
            stride,
            max_confidence,
            threshold,
            is_malicious
        );

        Ok(ScanResult {
            is_malicious,
            confidence: max_confidence,
            explanation,
        })
    }

    /// Generate sliding windows with mathematical guarantee for attack detection
    fn generate_sliding_windows(
        &self,
        tokens: &[u32],
        tokenizer: &Arc<Tokenizer>,
        window_size: usize,
        stride: usize,
    ) -> Result<Vec<SlidingWindow>> {
        let mut windows = Vec::new();
        let total_tokens = tokens.len();
        let mut start_token = 0;

        while start_token < total_tokens {
            let end_token = std::cmp::min(start_token + window_size, total_tokens);

            // Skip tiny windows at the end unless they're the only window
            if end_token - start_token < 50 && !windows.is_empty() {
                break;
            }

            // Extract token slice and decode back to text
            let window_tokens = &tokens[start_token..end_token];
            let window_text = tokenizer
                .decode(window_tokens, true)
                .map_err(|e| anyhow!("Failed to decode window tokens: {}", e))?;

            windows.push(SlidingWindow {
                start_token,
                end_token,
                token_count: window_tokens.len(),
                text: window_text,
            });

            // Move to next window
            start_token += stride;

            // If we've covered all tokens, break
            if end_token >= total_tokens {
                break;
            }
        }

        Ok(windows)
    }
    */

    /// Scan a single chunk of text with the ML model
    async fn scan_single_chunk(
        &self,
        text: &str,
        model: &Arc<dyn PromptInjectionModel>,
    ) -> Result<ScanResult> {
        tracing::debug!(
            "🔒 Scanning single chunk: length {} chars, preview: '{}'",
            text.len(),
            text.chars().take(100).collect::<String>()
        );

        match model.predict(text).await {
            Ok((ml_confidence, ml_explanation)) => {
                let threshold = self.get_threshold_from_config();
                let is_malicious = ml_confidence > threshold;

                tracing::debug!(
                    "🔒 Single chunk ML result: confidence={:.3}, threshold={:.3}, malicious={}",
                    ml_confidence,
                    threshold,
                    is_malicious
                );

                Ok(ScanResult {
                    is_malicious,
                    confidence: ml_confidence,
                    explanation: ml_explanation,
                })
            }
            Err(e) => {
                tracing::warn!("🔒 ML model prediction failed for chunk: {}", e);
                Err(e)
            }
        }
    }

    /// Combine ML model and pattern matching results
    fn combine_scan_results(
        &self,
        pattern_result: &ScanResult,
        ml_confidence: f32,
        _ml_explanation: &str,
        ml_is_malicious: bool,
    ) -> ScanResult {
        // Take the higher confidence score
        let final_confidence = pattern_result.confidence.max(ml_confidence);

        // Mark as malicious if either method detects it
        let final_is_malicious = pattern_result.is_malicious || ml_is_malicious;

        // Simplified explanation - just show what detected the threat
        let combined_explanation = if pattern_result.is_malicious && ml_is_malicious {
            "Detected by both pattern analysis and ML model".to_string()
        } else if pattern_result.is_malicious {
            format!(
                "Detected by pattern analysis: {}",
                pattern_result
                    .explanation
                    .replace("Pattern-based detection: ", "")
            )
        } else if ml_is_malicious {
            "Detected by machine learning model".to_string()
        } else {
            "No threats detected".to_string()
        };

        ScanResult {
            is_malicious: final_is_malicious,
            confidence: final_confidence,
            explanation: combined_explanation,
        }
    }

    /// Get threshold from config
    pub fn get_threshold_from_config(&self) -> f32 {
        use crate::config::Config;
        let config = Config::global();

        // Get security config and extract threshold
        if let Ok(security_value) = config.get_param::<serde_json::Value>("security") {
            if let Some(models_array) = security_value.get("models").and_then(|m| m.as_array()) {
                if let Some(first_model) = models_array.first() {
                    if let Some(threshold) = first_model.get("threshold").and_then(|t| t.as_f64()) {
                        return threshold as f32;
                    }
                }
            }
        }

        0.7 // Default threshold
    }

    /// Fallback pattern-based scanning
    async fn scan_with_patterns(&self, text: &str) -> Result<ScanResult> {
        let text_lower = text.to_lowercase();

        // Command injection patterns - detect potentially dangerous commands
        let dangerous_patterns = [
            // File system operations
            "rm -rf /",
            "rm -rf /*",
            "rm -rf ~",
            "rm -rf $home",
            "rmdir /",
            "del /s /q",
            "format c:",
            // System manipulation
            "shutdown",
            "reboot",
            "halt",
            "poweroff",
            "kill -9",
            "killall",
            // Network/data exfiltration and remote execution
            "bash <(curl",
            "sh <(curl",
            "bash <(wget",
            "sh <(wget",
            "curl http",
            "wget http",
            "nc -l",
            "netcat",
            "ssh ",
            "scp ",
            "rsync",
            // Process manipulation
            "sudo ",
            "su -",
            "chmod 777",
            "chown root",
            // Command chaining that could hide malicious intent
            "; rm ",
            "&& rm ",
            "| rm ",
            "; curl ",
            "&& curl ",
            "| curl ",
            "; wget ",
            "&& wget ",
            // Suspicious file operations
            "rm -f /",
            "rm -rf .",
            "rm -rf ..",
            "> /dev/",
            "dd if=",
            "mkfs",
            // Potential data theft
            "cat /etc/passwd",
            "cat /etc/shadow",
            "/etc/hosts",
            "~/.ssh/",
            "id_rsa",
            // Obfuscation attempts
            "base64 -d",
            "echo | sh",
            "eval ",
            "exec ",
        ];

        let mut detected_patterns = Vec::new();
        let mut max_risk_score: f32 = 0.0;

        for pattern in &dangerous_patterns {
            if text_lower.contains(pattern) {
                detected_patterns.push(pattern.to_string());

                // Assign risk scores based on severity
                let risk_score = match *pattern {
                    // Critical - system destruction
                    "rm -rf /" | "rm -rf /*" | "format c:" | "mkfs" => 0.95,
                    "rm -rf ~" | "rm -rf $home" => 0.90,

                    // Critical - remote code execution patterns
                    "bash <(curl" | "sh <(curl" | "bash <(wget" | "sh <(wget" => 0.95,

                    // High - system control
                    "shutdown" | "reboot" | "halt" | "poweroff" => 0.85,
                    "sudo " | "su -" | "chmod 777" | "chown root" => 0.80,

                    // Medium-High - network/data access
                    "curl http" | "wget http" | "ssh " | "scp " => 0.75,
                    "cat /etc/passwd" | "cat /etc/shadow" | "~/.ssh/" => 0.85,

                    // Medium - suspicious operations
                    "; rm " | "&& rm " | "| rm " => 0.70,
                    "kill -9" | "killall" => 0.65,

                    // Lower - potentially legitimate but suspicious
                    "base64 -d" | "eval " | "exec " => 0.60,

                    _ => 0.50,
                };

                max_risk_score = max_risk_score.max(risk_score);
            }
        }

        if !detected_patterns.is_empty() {
            // Use config threshold for pattern-based detection too
            let config_threshold = self.get_threshold_from_config();
            let is_malicious = max_risk_score > config_threshold;
            let explanation = format!(
                "Pattern-based detection: Found {} suspicious command pattern(s): [{}]. Risk score: {:.2}",
                detected_patterns.len(),
                detected_patterns.join(", "),
                max_risk_score
            );

            tracing::info!(
                "🔒 Pattern-based scan detected {} suspicious patterns with max risk score: {:.2}",
                detected_patterns.len(),
                max_risk_score
            );

            Ok(ScanResult {
                is_malicious,
                confidence: max_risk_score,
                explanation,
            })
        } else {
            Ok(ScanResult {
                is_malicious: false,
                confidence: 0.0,
                explanation: "Pattern-based scan: No suspicious command patterns detected"
                    .to_string(),
            })
        }
    }
}

impl Default for PromptInjectionScanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a sliding window for ML scanning
#[derive(Debug, Clone)]
struct SlidingWindow {
    start_token: usize,
    end_token: usize,
    token_count: usize,
    text: String,
}

/// Represents a sequential chunk for 512-token scanning
#[derive(Debug, Clone)]
struct SequentialChunk {
    start_token: usize,
    end_token: usize,
    token_count: usize,
    text: String,
}
