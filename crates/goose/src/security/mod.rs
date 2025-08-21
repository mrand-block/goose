pub mod model_downloader;
pub mod scanner;

use crate::conversation::message::Message;
use crate::permission::permission_judge::PermissionCheckResult;
use anyhow::Result;
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
    pub finding_id: String,
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
        let result = config
            .get_param::<serde_json::Value>("security")
            .ok()
            .and_then(|security_config| security_config.get("enabled")?.as_bool())
            .unwrap_or(false);

        println!(
            "🔒 Config check - security config result: {:?}",
            config.get_param::<serde_json::Value>("security")
        );
        println!("🔒 Final security enabled result: {}", result);

        result
    }

    /// Scan tool response content for malicious content (prompt injection, etc.)
    pub async fn scan_response_content(
        &self,
        response_content: &str,
        _messages: &[Message],
    ) -> Result<SecurityResult> {
        let Some(scanner) = &self.scanner else {
            // Security disabled, return safe result
            return Ok(SecurityResult {
                is_malicious: false,
                confidence: 0.0,
                explanation: "Security scanning disabled".to_string(),
                should_ask_user: false,
                finding_id: "DISABLED".to_string(),
            });
        };

        // Use new tool response scanning with normalization and 512-token chunking
        let scan_result = scanner.scan_tool_response_content(response_content).await?;

        let finding_id = format!(
            "RESP-{}",
            &uuid::Uuid::new_v4().simple().to_string().to_uppercase()[..8]
        );

        Ok(SecurityResult {
            is_malicious: scan_result.is_malicious,
            confidence: scan_result.confidence,
            explanation: format!("Tool response scan: {}", scan_result.explanation),
            should_ask_user: scan_result.confidence > scanner.get_threshold_from_config(),
            finding_id,
        })
    }
    /// Uses the proper two-step security analysis process
    /// Scans ALL tools (approved + needs_approval) for security threats
    /// Also scans system prompt if provided for persistent injection attacks
    pub async fn filter_malicious_tool_calls(
        &self,
        messages: &[Message],
        permission_check_result: &PermissionCheckResult,
        system_prompt: Option<&str>,
    ) -> Result<Vec<SecurityResult>> {
        let Some(scanner) = &self.scanner else {
            // Security disabled, return empty results
            return Ok(vec![]);
        };

        let mut results = Vec::new();

        // First, scan system prompt if provided for persistent injection attacks
        if let Some(system_prompt) = system_prompt {
            tracing::info!("🔍 Scanning system prompt for persistent injection attacks");

            let system_prompt_result = scanner.scan_system_prompt(system_prompt).await?;

            if system_prompt_result.is_malicious {
                let finding_id = format!(
                    "SYS-{}",
                    &uuid::Uuid::new_v4().simple().to_string().to_uppercase()[..8]
                );

                tracing::warn!(
                    confidence = system_prompt_result.confidence,
                    explanation = %system_prompt_result.explanation,
                    finding_id = %finding_id,
                    "🔒 System prompt contains persistent injection attack"
                );

                let config_threshold = scanner.get_threshold_from_config();

                results.push(SecurityResult {
                    is_malicious: system_prompt_result.is_malicious,
                    confidence: system_prompt_result.confidence,
                    explanation: format!(
                        "System prompt injection: {}",
                        system_prompt_result.explanation
                    ),
                    should_ask_user: system_prompt_result.confidence > config_threshold,
                    finding_id,
                });
            } else {
                tracing::debug!("✅ System prompt passed security analysis");
            }
        }

        // Check ALL tools (approved + needs_approval) for potential security issues
        for tool_request in permission_check_result
            .approved
            .iter()
            .chain(permission_check_result.needs_approval.iter())
        {
            if let Ok(tool_call) = &tool_request.tool_call {
                tracing::info!(
                    tool_name = %tool_call.name,
                    "🔍 Starting two-step security analysis for tool call"
                );

                // Use the new two-step analysis method
                let analysis_result = scanner
                    .analyze_tool_call_with_context(tool_call, messages)
                    .await?;

                if analysis_result.is_malicious {
                    // Generate a unique finding ID for this security detection
                    let finding_id = format!(
                        "SEC-{}",
                        &uuid::Uuid::new_v4().simple().to_string().to_uppercase()[..8]
                    );

                    tracing::warn!(
                        tool_name = %tool_call.name,
                        confidence = analysis_result.confidence,
                        explanation = %analysis_result.explanation,
                        finding_id = %finding_id,
                        "🔒 Tool call flagged as malicious after two-step analysis"
                    );

                    // Get threshold from config - if confidence > threshold, ask user
                    let config_threshold = scanner.get_threshold_from_config();

                    results.push(SecurityResult {
                        is_malicious: analysis_result.is_malicious,
                        confidence: analysis_result.confidence,
                        explanation: analysis_result.explanation,
                        should_ask_user: analysis_result.confidence > config_threshold,
                        finding_id,
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

    /// Check if models need to be downloaded and return appropriate user message
    pub async fn check_model_download_status(&self) -> Option<String> {
        let Some(_scanner) = &self.scanner else {
            return None;
        };

        // Check if models are already available in memory
        if let Some(_model) = scanner::get_model_if_available().await {
            return None; // Models ready, no message needed
        }

        // Check if models exist on disk but aren't loaded
        if Self::models_exist_on_disk() {
            return Some("🔒 Loading security models...".to_string());
        }

        // Models need to be downloaded
        Some(
            "🔒 Setting up security scanning for first time use - this could take a minute..."
                .to_string(),
        )
    }

    /// Scan recipe components for security threats
    /// This should be called when loading/applying recipes
    pub async fn scan_recipe_components(
        &self,
        recipe: &crate::recipe::Recipe,
    ) -> Result<Vec<SecurityResult>> {
        let Some(scanner) = &self.scanner else {
            // Security disabled, return empty results
            return Ok(vec![]);
        };

        let mut results = Vec::new();

        // Scan recipe prompt (becomes initial user message)
        if let Some(prompt) = &recipe.prompt {
            if !prompt.trim().is_empty() {
                tracing::info!("🔍 Scanning recipe prompt for injection attacks");

                let prompt_result = scanner.scan_with_prompt_injection_model(prompt).await?;

                if prompt_result.is_malicious {
                    let finding_id = format!(
                        "RCP-{}",
                        &uuid::Uuid::new_v4().simple().to_string().to_uppercase()[..8]
                    );

                    tracing::warn!(
                        confidence = prompt_result.confidence,
                        explanation = %prompt_result.explanation,
                        finding_id = %finding_id,
                        "🔒 Recipe prompt contains malicious content"
                    );

                    let config_threshold = scanner.get_threshold_from_config();

                    results.push(SecurityResult {
                        is_malicious: prompt_result.is_malicious,
                        confidence: prompt_result.confidence,
                        explanation: format!(
                            "Recipe prompt injection: {}",
                            prompt_result.explanation
                        ),
                        should_ask_user: prompt_result.confidence > config_threshold,
                        finding_id,
                    });
                }
            }
        }

        // Scan recipe context (additional context data)
        if let Some(context_items) = &recipe.context {
            for (i, context_item) in context_items.iter().enumerate() {
                if !context_item.trim().is_empty() {
                    tracing::info!(
                        "🔍 Scanning recipe context item {} for injection attacks",
                        i
                    );

                    let context_result = scanner
                        .scan_with_prompt_injection_model(context_item)
                        .await?;

                    if context_result.is_malicious {
                        let finding_id = format!(
                            "RCC-{}",
                            &uuid::Uuid::new_v4().simple().to_string().to_uppercase()[..8]
                        );

                        tracing::warn!(
                            context_index = i,
                            confidence = context_result.confidence,
                            explanation = %context_result.explanation,
                            finding_id = %finding_id,
                            "🔒 Recipe context contains malicious content"
                        );

                        let config_threshold = scanner.get_threshold_from_config();

                        results.push(SecurityResult {
                            is_malicious: context_result.is_malicious,
                            confidence: context_result.confidence,
                            explanation: format!(
                                "Recipe context[{}] injection: {}",
                                i, context_result.explanation
                            ),
                            should_ask_user: context_result.confidence > config_threshold,
                            finding_id,
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    /// Check if model files exist on disk
    fn models_exist_on_disk() -> bool {
        use crate::security::scanner::PromptInjectionScanner;

        let model_info = PromptInjectionScanner::get_model_info_from_config();

        if let Some(cache_dir) = dirs::cache_dir() {
            let security_models_dir = cache_dir.join("goose").join("security_models");
            let model_path = security_models_dir.join(&model_info.onnx_filename);
            let tokenizer_path = security_models_dir.join(&model_info.tokenizer_filename);

            return model_path.exists() && tokenizer_path.exists();
        }

        false
    }
}

impl Default for SecurityManager {
    fn default() -> Self {
        Self::new()
    }
}
