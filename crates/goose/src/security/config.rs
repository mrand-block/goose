use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enabled: bool,
    pub scanner_type: ScannerType,
    pub ollama_endpoint: String,
    pub action_policy: ActionPolicy,
    pub scan_threshold: ThreatThreshold,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScannerType {
    None,
    MistralNemo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActionPolicy {
    Block,      // Block content above threshold
    Sanitize,   // Use sanitized version if available
    Warn,       // Just warn but allow content
    LogOnly,    // Only log, no intervention
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ThreatThreshold {
    Any,        // Detect any threat level
    Low,        // Low and above
    Medium,     // Medium and above
    High,       // High and above
    Critical,   // Only critical threats
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            scanner_type: ScannerType::MistralNemo,
            ollama_endpoint: "http://localhost:11434".to_string(),
            action_policy: ActionPolicy::Block,
            scan_threshold: ThreatThreshold::Medium,
        }
    }
}