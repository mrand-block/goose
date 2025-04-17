use async_trait::async_trait;
use anyhow::Result;
use serde_json::Value;
use mcp_core::Content;

#[derive(Debug, Clone, PartialEq)]
pub enum ThreatLevel {
    Safe,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug)]
pub struct ScanResult {
    pub threat_level: ThreatLevel,
    pub explanation: String,
    pub sanitized_content: Option<Vec<Content>>, // Optional sanitized version
}

#[async_trait]
pub trait ContentScanner: Send + Sync {
    async fn scan_content(&self, content: &[Content]) -> Result<ScanResult>;
    async fn scan_tool_result(&self, tool_name: &str, arguments: &Value, result: &[Content]) -> Result<ScanResult>;
}