use super::content_scanner::{ContentScanner, ScanResult, ThreatLevel};
use async_trait::async_trait;
use anyhow::{Result, Context};
use mcp_core::Content;
use serde_json::Value;

pub struct MistralNemoScanner {
    ollama_endpoint: String,
    model_name: String,
    detection_prompt_template: String,
}

impl MistralNemoScanner {
    pub fn new(ollama_endpoint: String) -> Self {
        Self {
            ollama_endpoint,
            model_name: "mistral-nemo".to_string(),
            detection_prompt_template: include_str!("../prompts/threat_detection.md").to_string(),
        }
    }
    
    async fn analyze_content(&self, content: &str, tool_context: Option<(&str, &Value)>) -> Result<ScanResult> {
        // Prepare the prompt with content and optional tool context
        let prompt = self.prepare_detection_prompt(content, tool_context);
        
        // Log the prompt being sent to Ollama
        tracing::info!(
            "Security scanner sending prompt to Ollama ({}): {}",
            self.model_name,
            prompt.chars().take(200).collect::<String>() + "..."
        );
        
        // Call the Ollama API using the chat endpoint
        let client = reqwest::Client::new();
        
        let request_body = serde_json::json!({
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a security expert analyzing content for prompt injection and other security threats."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "format": "json",
            "stream": false  // Disable streaming to get a complete response
        });
        
        tracing::info!("Sending request to Ollama chat API: {}", serde_json::to_string(&request_body).unwrap_or_default());
        
        let response = client.post(&format!("{}/api/chat", self.ollama_endpoint))
            .json(&request_body)
            .send()
            .await
            .context("Failed to connect to Ollama server")?;
        
        // Parse the response
        let response_text = response.text().await?;
        tracing::info!("Security scanner received raw response: {}", response_text);
        
        // Check if we got a single JSON object or a stream of JSON objects
        let response_json: Value = if response_text.contains("\n") {
            // We got a stream of JSON objects, process line by line
            tracing::info!("Detected streaming response despite stream:false, processing line by line");
            
            let mut full_content = String::new();
            for line in response_text.lines() {
                if let Ok(json) = serde_json::from_str::<Value>(line) {
                    if let Some(content) = json.get("message")
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_str()) {
                        full_content.push_str(content);
                    }
                }
            }
            
            tracing::info!("Assembled content from stream: {}", full_content);
            
            // Create a synthetic response object
            serde_json::json!({
                "message": {
                    "role": "assistant",
                    "content": full_content
                }
            })
        } else {
            // We got a single JSON object
            match serde_json::from_str(&response_text) {
                Ok(json) => json,
                Err(e) => {
                    tracing::error!("Failed to parse Ollama response as JSON: {}", e);
                    return Ok(ScanResult {
                        threat_level: ThreatLevel::Medium,
                        explanation: format!("Failed to parse Ollama response: {}", e),
                        sanitized_content: None,
                    });
                }
            }
        };
        
        // Extract the message content from the response
        let message_content = response_json
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or_default();
        
        tracing::info!("Security scanner extracted message content: {}", message_content);
        
        // Try to parse the message content as JSON
        let analysis: Value = match serde_json::from_str(message_content) {
            Ok(json) => {
                tracing::info!("Successfully parsed message content as JSON");
                json
            },
            Err(e) => {
                // If parsing fails, try to extract JSON from the response
                // Sometimes the model might add text before or after the JSON
                tracing::warn!("Failed to parse message content as JSON: {}", e);
                
                // Try to extract JSON using regex
                let re = regex::Regex::new(r"(?s)\{.*\}").unwrap();
                if let Some(captures) = re.find(message_content) {
                    let json_str = captures.as_str();
                    tracing::info!("Extracted JSON from message content: {}", json_str);
                    match serde_json::from_str(json_str) {
                        Ok(json) => json,
                        Err(e) => {
                            tracing::error!("Failed to parse extracted JSON: {}", e);
                            
                            // If we can't parse the JSON, return a default medium threat level
                            // This is a conservative approach - if we can't analyze it, treat it as potentially risky
                            return Ok(ScanResult {
                                threat_level: ThreatLevel::Medium,
                                explanation: format!("Failed to parse security analysis: {}", e),
                                sanitized_content: None,
                            });
                        }
                    }
                } else {
                    tracing::error!("Could not extract JSON from message content");
                    
                    // If we can't find any JSON, assume a conservative medium threat level
                    return Ok(ScanResult {
                        threat_level: ThreatLevel::Low,
                        explanation: "Unable to analyze content - treating as low risk by default".to_string(),
                        sanitized_content: None,
                    });
                }
            }
        };
        
        // Extract threat assessment
        self.parse_threat_assessment(analysis, content)
    }
    
    fn prepare_detection_prompt(&self, content: &str, tool_context: Option<(&str, &Value)>) -> String {
        let mut prompt = self.detection_prompt_template.clone();
        
        // Add tool context if available
        if let Some((tool_name, arguments)) = tool_context {
            prompt = prompt.replace("{{TOOL_CONTEXT}}", &format!(
                "Tool name: {}\nTool arguments: {}", 
                tool_name, 
                serde_json::to_string_pretty(arguments).unwrap_or_default()
            ));
        } else {
            prompt = prompt.replace("{{TOOL_CONTEXT}}", "No specific tool context available.");
        }
        
        // Add the content to analyze
        prompt.replace("{{CONTENT}}", content)
    }
    
    fn parse_threat_assessment(&self, analysis: Value, _original_content: &str) -> Result<ScanResult> {
        // Extract fields from the JSON response
        let threat_level_str = analysis.get("threat_level")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
            
        let explanation = analysis.get("explanation")
            .and_then(|v| v.as_str())
            .unwrap_or("No explanation provided")
            .to_string();
            
        // Convert string threat level to enum
        let threat_level = match threat_level_str.to_lowercase().as_str() {
            "safe" => ThreatLevel::Safe,
            "low" => ThreatLevel::Low,
            "medium" => ThreatLevel::Medium,
            "high" => ThreatLevel::High,
            "critical" => ThreatLevel::Critical,
            _ => ThreatLevel::Medium, // Default to Medium if unknown
        };
        
        // Get sanitized content if available
        let sanitized_content = if threat_level != ThreatLevel::Safe {
            analysis.get("sanitized_content")
                .and_then(|v| v.as_str())
                .map(|s| vec![Content::text(s.to_string())])
        } else {
            None
        };
        
        Ok(ScanResult {
            threat_level,
            explanation,
            sanitized_content,
        })
    }
}

#[async_trait]
impl ContentScanner for MistralNemoScanner {
    async fn scan_content(&self, content: &[Content]) -> Result<ScanResult> {
        // Combine all content into a single string for analysis
        let combined_content = content.iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join("\n");
            
        self.analyze_content(&combined_content, None).await
    }
    
    async fn scan_tool_result(&self, tool_name: &str, arguments: &Value, result: &[Content]) -> Result<ScanResult> {
        // Combine all content into a single string for analysis
        let combined_content = result.iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join("\n");
            
        self.analyze_content(&combined_content, Some((tool_name, arguments))).await
    }
}