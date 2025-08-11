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
    /// Uses the proper two-step security analysis process
    /// Scans ALL tools (approved + needs_approval) for security threats
    pub async fn filter_malicious_tool_calls(
        &self,
        messages: &[Message],
        permission_check_result: &PermissionCheckResult,
    ) -> Result<Vec<SecurityResult>> {
        let Some(scanner) = &self.scanner else {
            // Security disabled, return empty results
            return Ok(vec![]);
        };

        let mut results = Vec::new();

        // Collect ALL tool requests (approved + needs_approval) for security scanning
        let mut all_tool_requests = Vec::new();
        all_tool_requests.extend(&permission_check_result.approved);
        all_tool_requests.extend(&permission_check_result.needs_approval);

        // Check ALL tools for potential security issues
        for tool_request in &all_tool_requests {
            if let Ok(tool_call) = &tool_request.tool_call {
                tracing::info!(
                    tool_name = %tool_call.name,
                    "🔍 Starting two-step security analysis for tool call"
                );

                // Use the new two-step analysis method
                let analysis_result = scanner.analyze_tool_call_with_context(
                    tool_call,
                    messages,
                ).await?;

                if analysis_result.is_malicious {
                    tracing::warn!(
                        tool_name = %tool_call.name,
                        confidence = analysis_result.confidence,
                        explanation = %analysis_result.explanation,
                        "🚨 Tool call flagged as malicious after two-step analysis"
                    );

                    results.push(SecurityResult {
                        is_malicious: analysis_result.is_malicious,
                        confidence: analysis_result.confidence,
                        explanation: analysis_result.explanation,
                        should_ask_user: analysis_result.confidence > 0.7,
                    });
                } else {
                    tracing::debug!(
                        tool_name = %tool_call.name,
                        confidence = analysis_result.confidence,
                        explanation = %analysis_result.explanation,
                        "✅ Tool call passed two-step security analysis"
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
