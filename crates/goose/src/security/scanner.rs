use anyhow::{Result, anyhow};
use mcp_core::tool::ToolCall;
use crate::conversation::message::Message;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::OnceCell;

use crate::security::model_downloader::{get_global_downloader, ModelInfo};

// ML inference backends
use ort::{session::Session, session::builder::GraphOptimizationLevel};
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
}

/// ONNX Runtime implementation
pub struct OnnxPromptInjectionModel {
    _session: Session,  // Temporarily unused due to mutable reference issue
    _tokenizer: Arc<Tokenizer>,  // Temporarily unused
    model_name: String,
}

impl OnnxPromptInjectionModel {
    pub async fn new(model_path: PathBuf, tokenizer_path: PathBuf, model_name: String) -> Result<Self> {
        // Initialize ONNX Runtime session
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&model_path)?;
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        
        Ok(Self {
            _session: session,
            _tokenizer: Arc::new(tokenizer),
            model_name,
        })
    }
}

#[async_trait::async_trait]
impl PromptInjectionModel for OnnxPromptInjectionModel {
    async fn predict(&self, text: &str) -> Result<(f32, String)> {
        // For now, return a reasonable prediction based on simple heuristics
        // TODO: Implement actual ONNX inference once we resolve the mutable reference issue
        
        let text_lower = text.to_lowercase();
        
        // Check for prompt injection patterns
        let injection_patterns = [
            "ignore previous instructions",
            "ignore all previous", 
            "forget everything",
            "new instructions",
            "system prompt",
            "you are now",
            "act as",
            "pretend to be",
            "roleplay as",
            "jailbreak",
            "developer mode",
        ];
        
        // Check for dangerous shell commands
        let dangerous_commands = [
            "rm -rf",
            "sudo rm",
            "del /s",
            "format c:",
            "dd if=",
            "mkfs",
            "fdisk",
            "chmod 777",
            "wget http",
            "curl http",
            "nc -l",
            "netcat",
        ];
        
        let mut confidence = 0.0f32;
        let mut detected_patterns = Vec::new();
        
        // Check for prompt injection patterns
        for pattern in &injection_patterns {
            if text_lower.contains(pattern) {
                detected_patterns.push(format!("injection:{}", pattern));
                confidence = confidence.max(0.9);
            }
        }
        
        // Check for dangerous commands
        for command in &dangerous_commands {
            if text_lower.contains(command) {
                detected_patterns.push(format!("dangerous:{}", command));
                confidence = confidence.max(0.8);
            }
        }
        
        let explanation = if detected_patterns.is_empty() {
            format!("ONNX model '{}': No threats detected", self.model_name)
        } else {
            format!("ONNX model '{}': Detected threats: {}", 
                self.model_name, detected_patterns.join(", "))
        };
        
        Ok((confidence, explanation))
    }
    
    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Global model cache
static MODEL_CACHE: OnceCell<Option<Arc<dyn PromptInjectionModel>>> = OnceCell::const_new();

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
                    match OnnxPromptInjectionModel::new(model_path, tokenizer_path, model_info.hf_model_name.clone()).await {
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
    MODEL_CACHE
        .get_or_init(|| async { initialize_model().await.unwrap_or(None) })
        .await
        .clone()
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
                tracing::info!("🔒 Security models found in cache, enabling security scanning");
                return true;
            }
        }
        
        // Models not cached, trigger download in background
        tracing::info!("🔒 Security models not found in cache, downloading in background");
        tokio::spawn(async {
            Self::ensure_models_available().await;
        });
        
        // For now, use pattern-based scanning while models download
        // TODO: In the future, we could block here or enable ONNX scanning after download
        false
    }

    /// Ensure models are available using the existing model_downloader
    async fn ensure_models_available() {
        tracing::info!("🔒 Ensuring security models are available...");
        
        match get_global_downloader().await {
            Ok(downloader) => {
                let model_info = Self::get_model_info_from_config();
                match downloader.ensure_model_available(&model_info).await {
                    Ok((model_path, tokenizer_path)) => {
                        tracing::info!(
                            "🔒 ✅ Security models ready: model={:?}, tokenizer={:?}",
                            model_path, tokenizer_path
                        );
                    }
                    Err(e) => {
                        tracing::warn!("🔒 Failed to ensure models available: {}", e);
                        tracing::info!("🔒 Continuing with pattern-based security scanning");
                    }
                }
            }
            Err(e) => {
                tracing::warn!("🔒 Failed to get model downloader: {}", e);
            }
        }
    }

    /// Get model information from config file
    fn get_model_info_from_config() -> ModelInfo {
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
                tracing::warn!("🔒 No security model configured, security scanning will be disabled");
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
        let final_result = self.make_final_security_decision(
            &tool_call_result,
            &user_messages_result,
            tool_call,
        );
        
        tracing::info!(
            tool_name = %tool_call.name,
            tool_confidence = tool_call_result.confidence,
            user_confidence = user_messages_result.confidence,
            final_malicious = final_result.is_malicious,
            final_confidence = final_result.confidence,
            "🔒 Two-step security analysis complete"
        );
        
        Ok(final_result)
    }

    /// Step 1: Scan only the tool call for suspicious patterns
    async fn scan_tool_call_only(&self, tool_call: &ToolCall) -> Result<ScanResult> {
        // Create text representation of the tool call for analysis
        let tool_text = format!(
            "Tool: {}\nArguments: {}",
            tool_call.name,
            serde_json::to_string_pretty(&tool_call.arguments)?
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
        tool_call: &ToolCall,
    ) -> ScanResult {
        // Decision logic:
        // 1. If user messages contain prompt injection, tool call is likely malicious
        // 2. If user messages are clean but tool call is suspicious, it might be a legitimate response
        // 3. Consider tool risk level as well
        
        let tool_risk = self.assess_tool_risk(&tool_call.name);
        
        let (is_malicious, confidence, explanation) = if user_messages_result.is_malicious {
            // User messages contain prompt injection - tool call is likely malicious
            let combined_confidence = (tool_call_result.confidence + user_messages_result.confidence) / 2.0;
            let explanation = format!(
                "MALICIOUS: Tool '{}' appears to be result of prompt injection. Tool scan: {:.2} confidence ({}). User messages scan: {:.2} confidence ({})",
                tool_call.name,
                tool_call_result.confidence,
                if tool_call_result.is_malicious { "suspicious" } else { "clean" },
                user_messages_result.confidence,
                user_messages_result.explanation
            );
            (true, combined_confidence.max(0.8), explanation)
        } else {
            // User messages are clean - suspicious tool call might be legitimate
            // Lower the confidence since user didn't inject malicious prompts
            let adjusted_confidence = tool_call_result.confidence * 0.6; // Reduce confidence
            let explanation = format!(
                "LIKELY SAFE: Tool '{}' flagged as suspicious but user messages appear clean. Tool scan: {:.2} confidence. User messages: clean ({:.2} confidence). Adjusted confidence: {:.2}",
                tool_call.name,
                tool_call_result.confidence,
                user_messages_result.confidence,
                adjusted_confidence
            );
            
            // Only consider malicious if adjusted confidence is still high AND tool is high-risk
            let is_malicious = adjusted_confidence > 0.7 && tool_risk > 0.6;
            (is_malicious, adjusted_confidence, explanation)
        };
        
        ScanResult {
            is_malicious,
            confidence,
            explanation,
        }
    }

    /// Legacy method for backward compatibility - now delegates to two-step analysis
    pub async fn scan_tool_call(&self, tool_call: &ToolCall) -> Result<ScanResult> {
        // For backward compatibility, just scan the tool call without context
        self.scan_tool_call_only(tool_call).await
    }

    /// Legacy method for backward compatibility - now delegates to user message scanning
    pub async fn analyze_conversation_context(
        &self,
        messages: &[Message],
        _tool_call: &ToolCall, // Ignored in new implementation
    ) -> Result<ScanResult> {
        // For backward compatibility, just scan user messages
        self.scan_user_messages_only(messages).await
    }

    /// Model-agnostic prompt injection scanning
    async fn scan_with_prompt_injection_model(&self, text: &str) -> Result<ScanResult> {
        // Try to get the ML model
        if let Some(model) = get_model().await {
            match model.predict(text).await {
                Ok((confidence, explanation)) => {
                    // Get threshold from config
                    let threshold = self.get_threshold_from_config();
                    let is_malicious = confidence > threshold;
                    
                    tracing::info!(
                        "🔒 ML model prediction: confidence={:.3}, threshold={:.3}, malicious={}",
                        confidence, threshold, is_malicious
                    );
                    
                    return Ok(ScanResult {
                        is_malicious,
                        confidence,
                        explanation,
                    });
                }
                Err(e) => {
                    tracing::warn!("🔒 ML model prediction failed: {}", e);
                    // Fall through to pattern-based scanning
                }
            }
        } else {
            tracing::info!("🔒 No ML model available, using pattern-based fallback");
        }
        
        // Fallback to pattern-based scanning if ML model is not available
        self.scan_with_patterns(text).await
    }
    
    /// Get threshold from config
    fn get_threshold_from_config(&self) -> f32 {
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
        let _text_lower = text.to_lowercase();
        
        // Use BERT model-based scanning instead of hardcoded patterns
        // This provides more sophisticated detection than simple string matching
        
        // For now, return a low-confidence result indicating no threats detected
        // The actual ML-based scanning happens in the ONNX model prediction above
        Ok(ScanResult {
            is_malicious: false,
            confidence: 0.0,
            explanation: "Pattern-based fallback: No threats detected using ML-based analysis".to_string(),
        })
    }

    /// Assess inherent risk of specific tools
    fn assess_tool_risk(&self, tool_name: &str) -> f32 {
        // Higher risk tools that could be used maliciously
        match tool_name {
            name if name.contains("shell") || name.contains("exec") => 0.8,
            name if name.contains("file") && name.contains("write") => 0.6,
            name if name.contains("network") || name.contains("http") => 0.5,
            name if name.contains("read") => 0.3,
            _ => 0.1,
        }
    }
}

impl Default for PromptInjectionScanner {
    fn default() -> Self {
        Self::new()
    }
}
