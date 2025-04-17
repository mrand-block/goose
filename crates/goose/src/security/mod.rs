pub mod content_scanner;
pub mod threat_detection;
pub mod config;

use anyhow::Result;
use content_scanner::{ContentScanner, ScanResult, ThreatLevel};
use config::{SecurityConfig, ScannerType, ActionPolicy, ThreatThreshold};
use mcp_core::Content;
use serde_json::Value;
use std::sync::Arc;
use threat_detection::MistralNemoScanner;

pub struct SecurityManager {
    config: SecurityConfig,
    scanner: Option<Arc<dyn ContentScanner>>,
}

impl SecurityManager {
    pub fn new(config: SecurityConfig) -> Self {
        let scanner = if config.enabled {
            match config.scanner_type {
                ScannerType::MistralNemo => {
                    tracing::info!(
                        enabled = true,
                        scanner = ?config.scanner_type,
                        endpoint = %config.ollama_endpoint,
                        action_policy = ?config.action_policy,
                        threshold = ?config.scan_threshold,
                        "Initializing security scanner"
                    );
                    Some(Arc::new(MistralNemoScanner::new(config.ollama_endpoint.clone())) as Arc<dyn ContentScanner>)
                }
                ScannerType::None => {
                    tracing::info!("Security scanner type is None, scanner will be disabled");
                    None
                }
            }
        } else {
            tracing::info!("Security scanner is disabled in configuration");
            None
        };
        
        Self { config, scanner }
    }
    
    pub fn is_enabled(&self) -> bool {
        self.config.enabled && self.scanner.is_some()
    }
    
    pub async fn scan_content(&self, content: &[Content]) -> Result<Option<ScanResult>> {
        if !self.is_enabled() {
            tracing::info!("Security scanner is disabled, skipping content scan");
            return Ok(None);
        }
        
        tracing::info!("Starting security scan of content");
        let scanner = self.scanner.as_ref().unwrap();
        let scan_result = scanner.scan_content(content).await?;
        
        // Log the scan result
        match scan_result.threat_level {
            ThreatLevel::Safe => {
                tracing::info!("Content scan result: Safe");
            }
            ThreatLevel::Low => {
                tracing::info!(
                    threat = "low",
                    explanation = %scan_result.explanation,
                    "Content scan detected low threat"
                );
            }
            ThreatLevel::Medium => {
                tracing::info!(
                    threat = "medium", 
                    explanation = %scan_result.explanation,
                    "Content scan detected medium threat"
                );
            }
            ThreatLevel::High | ThreatLevel::Critical => {
                tracing::info!(
                    threat = ?scan_result.threat_level,
                    explanation = %scan_result.explanation,
                    "Content scan detected high/critical threat"
                );
            }
        }
        
        Ok(Some(scan_result))
    }
    
    pub async fn scan_tool_result(&self, tool_name: &str, arguments: &Value, result: &[Content]) -> Result<Option<ScanResult>> {
        if !self.is_enabled() {
            tracing::info!("Security scanner is disabled, skipping tool result scan");
            return Ok(None);
        }
        
        tracing::info!(tool = tool_name, "Starting security scan of tool result");
        let scanner = self.scanner.as_ref().unwrap();
        let scan_result = scanner.scan_tool_result(tool_name, arguments, result).await?;
        
        // Log the scan result
        match scan_result.threat_level {
            ThreatLevel::Safe => {
                tracing::info!(tool = tool_name, "Tool result scan: Safe");
            }
            ThreatLevel::Low => {
                tracing::info!(
                    tool = tool_name,
                    threat = "low",
                    explanation = %scan_result.explanation,
                    "Tool result scan detected low threat"
                );
            }
            ThreatLevel::Medium => {
                tracing::info!(
                    tool = tool_name,
                    threat = "medium", 
                    explanation = %scan_result.explanation,
                    "Tool result scan detected medium threat"
                );
            }
            ThreatLevel::High | ThreatLevel::Critical => {
                tracing::info!(
                    tool = tool_name,
                    threat = ?scan_result.threat_level,
                    explanation = %scan_result.explanation,
                    "Tool result scan detected high/critical threat"
                );
            }
        }
        
        Ok(Some(scan_result))
    }
    
    pub fn should_block(&self, scan_result: &ScanResult) -> bool {
        if !self.config.enabled || self.config.action_policy != ActionPolicy::Block {
            return false;
        }
        
        // Check if threat level is at or above threshold
        match self.config.scan_threshold {
            ThreatThreshold::Any => scan_result.threat_level != ThreatLevel::Safe,
            ThreatThreshold::Low => {
                matches!(
                    scan_result.threat_level,
                    ThreatLevel::Low | ThreatLevel::Medium | ThreatLevel::High | ThreatLevel::Critical
                )
            }
            ThreatThreshold::Medium => {
                matches!(
                    scan_result.threat_level,
                    ThreatLevel::Medium | ThreatLevel::High | ThreatLevel::Critical
                )
            }
            ThreatThreshold::High => {
                matches!(
                    scan_result.threat_level,
                    ThreatLevel::High | ThreatLevel::Critical
                )
            }
            ThreatThreshold::Critical => scan_result.threat_level == ThreatLevel::Critical,
        }
    }
    
    pub fn get_safe_content(&self, original: &[Content], scan_result: &ScanResult) -> Vec<Content> {
        if !self.is_enabled() || self.config.action_policy == ActionPolicy::LogOnly {
            tracing::info!("Security scanner: passing through original content (policy: {})", 
                if !self.is_enabled() { "disabled" } else { "LogOnly" });
            return original.to_vec();
        }
        
        if self.config.action_policy == ActionPolicy::Sanitize && scan_result.sanitized_content.is_some() {
            tracing::info!(
                threat = ?scan_result.threat_level,
                policy = "Sanitize",
                "Security scanner: sanitizing content due to detected threat: {}", 
                scan_result.explanation
            );
            return scan_result.sanitized_content.clone().unwrap();
        }
        
        if self.should_block(scan_result) {
            // Replace with warning message
            tracing::info!(
                threat = ?scan_result.threat_level,
                policy = "Block",
                "Security scanner: BLOCKING content due to detected threat: {}", 
                scan_result.explanation
            );
            return vec![Content::text(format!(
                "[SECURITY WARNING] Content blocked due to detected threat: {}",
                scan_result.explanation
            ))];
        }
        
        // Default to original content
        tracing::info!(
            threat = ?scan_result.threat_level,
            policy = ?self.config.action_policy,
            "Security scanner: allowing content through (threat below threshold)"
        );
        original.to_vec()
    }
}