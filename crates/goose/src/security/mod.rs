pub mod scanner;
pub mod model_downloader;

use anyhow::Result;
use crate::conversation::message::Message;
use crate::permission::permission_judge::PermissionCheckResult;
use scanner::PromptInjectionScanner;

/// Simple security manager for the POC
/// Focuses on tool call analysis with conversation context
pub struct SecurityManager {
    scanner: Option<PromptInjectionScanner>,
}

#[derive(Debug, Clone)]
pub struct SecurityResult {
    pub is_malicious: bool,
    pub confidence: f32,
    pub explanation: String,
    pub should_ask_user: bool,
}

impl SecurityManager {
    pub fn new() -> Self {
        println!("🔒 SecurityManager::new() called - checking if security should be enabled");
        
        // Initialize scanner based on config
        let should_enable = Self::should_enable_security();
        println!("🔒 Security enabled check result: {}", should_enable);
        
        let scanner = match should_enable {
            true => {
                println!("🔒 Initializing security scanner");
                tracing::info!("🔒 Initializing security scanner");
                Some(PromptInjectionScanner::new())
            }
            false => {
                println!("🔓 Security scanning disabled");
                tracing::info!("🔓 Security scanning disabled");
                None
            }
        };

        Self { scanner }
    }

    /// Check if security should be enabled based on config
    fn should_enable_security() -> bool {
        // Check config file for security settings
        use crate::config::Config;
        let config = Config::global();
        
        // Try to get security.enabled from config
        let result = config.get_param::<serde_json::Value>("security")
            .ok()
            .and_then(|security_config| security_config.get("enabled")?.as_bool())
            .unwrap_or(false);
        
        println!("🔒 Config check - security config result: {:?}", 
                 config.get_param::<serde_json::Value>("security"));
        println!("🔒 Final security enabled result: {}", result);
        
        result
    }

    /// Main security check function - called from reply_internal
    pub async fn filter_evil_tool_calls(
        &self,
        messages: &[Message],
        permission_check_result: &PermissionCheckResult,
    ) -> Result<Vec<SecurityResult>> {
        let Some(scanner) = &self.scanner else {
            // Security disabled, return empty results
            return Ok(vec![]);
        };

        let mut results = Vec::new();

        // Check tools that need approval for potential security issues
        for tool_request in &permission_check_result.needs_approval {
            if let Ok(tool_call) = &tool_request.tool_call {
                tracing::info!(
                    tool_name = %tool_call.name,
                    "🔍 Analyzing tool call for security threats"
                );

                // First, check if the tool call itself looks suspicious
                let tool_suspicious = scanner.scan_tool_call(tool_call).await?;
                
                if tool_suspicious.is_malicious {
                    // Tool call looks suspicious, analyze conversation context
                    tracing::warn!(
                        tool_name = %tool_call.name,
                        confidence = tool_suspicious.confidence,
                        "🚨 Suspicious tool call detected, analyzing conversation context"
                    );

                    let context_result = scanner.analyze_conversation_context(
                        messages,
                        tool_call,
                    ).await?;

                    results.push(SecurityResult {
                        is_malicious: context_result.is_malicious,
                        confidence: context_result.confidence,
                        explanation: format!(
                            "Tool '{}' flagged as suspicious (confidence: {:.2}). Context analysis: {}",
                            tool_call.name,
                            tool_suspicious.confidence,
                            context_result.explanation
                        ),
                        should_ask_user: context_result.is_malicious && context_result.confidence > 0.7,
                    });
                } else {
                    tracing::debug!(
                        tool_name = %tool_call.name,
                        "✅ Tool call passed security check"
                    );
                }
            }
        }

        Ok(results)
    }
}

impl Default for SecurityManager {
    fn default() -> Self {
        Self::new()
    }
}
