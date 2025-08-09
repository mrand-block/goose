use anyhow::Result;
use mcp_core::tool::ToolCall;
use crate::conversation::message::Message;
use std::path::PathBuf;

use crate::security::model_downloader::{get_global_downloader, ModelInfo};

#[derive(Debug, Clone)]
pub struct ScanResult {
    pub is_malicious: bool,
    pub confidence: f32,
    pub explanation: String,
}

/// Simple prompt injection scanner
/// Uses the existing model_downloader infrastructure
pub struct PromptInjectionScanner {
    model_path: Option<PathBuf>,
    enabled: bool,
}

impl PromptInjectionScanner {
    pub fn new() -> Self {
        println!("🔒 PromptInjectionScanner::new() called");
        
        // Check if models are available, trigger download if needed
        let scanner = Self {
            model_path: None,
            enabled: Self::check_and_prepare_models(),
        };
        
        println!("🔒 Scanner enabled: {}", scanner.enabled);
        
        scanner
    }

    /// Check if models are available and trigger download if needed
    fn check_and_prepare_models() -> bool {
        // For now, trigger model download in background and use pattern-based scanning
        tokio::spawn(async {
            Self::ensure_models_available().await;
        });
        
        // Return false for now to use pattern-based scanning
        // This will be true once models are properly downloaded and available
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
                tracing::warn!("🔒 No security model configured, using default");
                "protectai/deberta-v3-base-prompt-injection-v2".to_string()
            });

        tracing::info!("🔒 Using security model from config: {}", model_name);

        // Create ModelInfo from config
        let safe_filename = model_name.replace("/", "_").replace("-", "_");
        ModelInfo {
            hf_model_name: model_name,
            onnx_filename: format!("{}.onnx", safe_filename),
            tokenizer_filename: format!("{}_tokenizer.json", safe_filename),
        }
    }

    /// Scan a tool call for suspicious patterns
    pub async fn scan_tool_call(&self, tool_call: &ToolCall) -> Result<ScanResult> {
        // Create text representation of the tool call for analysis
        let tool_text = format!(
            "Tool: {}\nArguments: {}",
            tool_call.name,
            serde_json::to_string_pretty(&tool_call.arguments)?
        );

        // For now, always use pattern-based scanning
        // TODO: Use ONNX model when available
        self.scan_with_patterns(&tool_text).await
    }

    /// Analyze conversation context to determine if tool call is malicious
    pub async fn analyze_conversation_context(
        &self,
        messages: &[Message],
        tool_call: &ToolCall,
    ) -> Result<ScanResult> {
        // Combine recent messages for context analysis
        let context_text = self.build_context_text(messages, tool_call);

        // For now, always use pattern-based analysis
        // TODO: Use ONNX model when available
        self.analyze_context_with_patterns(&context_text, tool_call).await
    }

    /// Build context text from conversation history
    fn build_context_text(&self, messages: &[Message], tool_call: &ToolCall) -> String {
        // Take last 5 messages for context (adjust as needed)
        let recent_messages: Vec<String> = messages
            .iter()
            .rev()
            .take(5)
            .rev()
            .filter_map(|msg| {
                // Extract text content from messages
                msg.content.first()?.as_text().map(|text| {
                    format!("{:?}: {}", msg.role, text)
                })
            })
            .collect();

        let context = recent_messages.join("\n");
        let tool_text = format!(
            "Tool: {}\nArguments: {}",
            tool_call.name,
            serde_json::to_string_pretty(&tool_call.arguments).unwrap_or_default()
        );

        format!("Context:\n{}\n\nProposed Action:\n{}", context, tool_text)
    }

    /// Fallback pattern-based scanning
    async fn scan_with_patterns(&self, text: &str) -> Result<ScanResult> {
        let text_lower = text.to_lowercase();
        
        // Simple patterns that might indicate prompt injection
        let suspicious_patterns = [
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

        let mut max_confidence: f32 = 0.0;
        let mut detected_patterns = Vec::new();

        for pattern in &suspicious_patterns {
            if text_lower.contains(pattern) {
                detected_patterns.push(*pattern);
                max_confidence = max_confidence.max(0.8); // High confidence for pattern match
            }
        }

        let is_malicious = max_confidence > 0.5;
        let explanation = if detected_patterns.is_empty() {
            "Pattern-based scan: No suspicious patterns detected".to_string()
        } else {
            format!("Pattern-based scan detected: {} (confidence: {:.2})", 
                detected_patterns.join(", "), max_confidence)
        };

        Ok(ScanResult {
            is_malicious,
            confidence: max_confidence,
            explanation,
        })
    }

    /// Analyze context with patterns
    async fn analyze_context_with_patterns(
        &self,
        context_text: &str,
        tool_call: &ToolCall,
    ) -> Result<ScanResult> {
        // First scan the context for suspicious patterns
        let context_result = self.scan_with_patterns(context_text).await?;
        
        // Consider tool-specific risks
        let tool_risk = self.assess_tool_risk(&tool_call.name);
        
        // Combine context analysis with tool risk
        let combined_confidence = (context_result.confidence * 0.7) + (tool_risk * 0.3);
        let is_malicious = combined_confidence > 0.6;

        let explanation = format!(
            "Context analysis: {} (confidence: {:.2}). Tool risk assessment: {:.2}. Combined risk: {:.2}",
            context_result.explanation,
            context_result.confidence,
            tool_risk,
            combined_confidence
        );

        Ok(ScanResult {
            is_malicious,
            confidence: combined_confidence,
            explanation,
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
