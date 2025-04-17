use anyhow::Result;
use chrono::{DateTime, TimeZone, Utc};
use futures::future;
use futures::stream::{FuturesUnordered, StreamExt};
use mcp_client::McpService;
use mcp_core::protocol::GetPromptResult;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::task;
use tracing::info;

use super::extension::{ExtensionConfig, ExtensionError, ExtensionInfo, ExtensionResult, ToolInfo};
use crate::config::ExtensionConfigManager;
use crate::prompt_template;
use crate::security::SecurityManager;
use crate::security::config::SecurityConfig;
use mcp_client::client::{ClientCapabilities, ClientInfo, McpClient, McpClientTrait};
use mcp_client::transport::{SseTransport, StdioTransport, Transport};
use mcp_core::{prompt::Prompt, Content, Tool, ToolCall, ToolError, ToolResult};
use serde_json::Value;

// By default, we set it to Jan 1, 2020 if the resource does not have a timestamp
// This is to ensure that the resource is considered less important than resources with a more recent timestamp
static DEFAULT_TIMESTAMP: LazyLock<DateTime<Utc>> =
    LazyLock::new(|| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap());

type McpClientBox = Arc<Mutex<Box<dyn McpClientTrait>>>;

/// Manages Goose extensions / MCP clients and their interactions
pub struct ExtensionManager {
    clients: HashMap<String, McpClientBox>,
    instructions: HashMap<String, String>,
    resource_capable_extensions: HashSet<String>,
    security_manager: SecurityManager,
}

/// A flattened representation of a resource used by the agent to prepare inference
#[derive(Debug, Clone)]
pub struct ResourceItem {
    pub client_name: String,      // The name of the client that owns the resource
    pub uri: String,              // The URI of the resource
    pub name: String,             // The name of the resource
    pub content: String,          // The content of the resource
    pub timestamp: DateTime<Utc>, // The timestamp of the resource
    pub priority: f32,            // The priority of the resource
    pub token_count: Option<u32>, // The token count of the resource (filled in by the agent)
}

impl ResourceItem {
    pub fn new(
        client_name: String,
        uri: String,
        name: String,
        content: String,
        timestamp: DateTime<Utc>,
        priority: f32,
    ) -> Self {
        Self {
            client_name,
            uri,
            name,
            content,
            timestamp,
            priority,
            token_count: None,
        }
    }
}

/// Sanitizes a string by replacing invalid characters with underscores.
/// Valid characters match [a-zA-Z0-9_-]
fn normalize(input: String) -> String {
    let mut result = String::with_capacity(input.len());
    for c in input.chars() {
        result.push(match c {
            c if c.is_ascii_alphanumeric() || c == '_' || c == '-' => c,
            c if c.is_whitespace() => continue, // effectively "strip" whitespace
            _ => '_',                           // Replace any other non-ASCII character with '_'
        });
    }
    result.to_lowercase()
}

pub fn get_parameter_names(tool: &Tool) -> Vec<String> {
    tool.input_schema
        .get("properties")
        .and_then(|props| props.as_object())
        .map(|props| props.keys().cloned().collect())
        .unwrap_or_default()
}

impl Default for ExtensionManager {
    fn default() -> Self {
        Self::new(None)
    }
}

impl ExtensionManager {
    /// Create a new ExtensionManager instance
    pub fn new(security_config: Option<SecurityConfig>) -> Self {
        Self {
            clients: HashMap::new(),
            instructions: HashMap::new(),
            resource_capable_extensions: HashSet::new(),
            security_manager: SecurityManager::new(security_config.unwrap_or_default()),
        }
    }

    pub fn supports_resources(&self) -> bool {
        !self.resource_capable_extensions.is_empty()
    }

    /// Add a new MCP extension based on the provided client type
    // TODO IMPORTANT need to ensure this times out if the extension command is broken!
    pub async fn add_extension(&mut self, config: ExtensionConfig) -> ExtensionResult<()> {
        let sanitized_name = normalize(config.key().to_string());
        let mut client: Box<dyn McpClientTrait> = match &config {
            ExtensionConfig::Sse {
                uri, envs, timeout, ..
            } => {
                let transport = SseTransport::new(uri, envs.get_env());
                let handle = transport.start().await?;
                let service = McpService::with_timeout(
                    handle,
                    Duration::from_secs(
                        timeout.unwrap_or(crate::config::DEFAULT_EXTENSION_TIMEOUT),
                    ),
                );
                Box::new(McpClient::new(service))
            }
            ExtensionConfig::Stdio {
                cmd,
                args,
                envs,
                timeout,
                ..
            } => {
                let transport = StdioTransport::new(cmd, args.to_vec(), envs.get_env());
                let handle = transport.start().await?;
                let service = McpService::with_timeout(
                    handle,
                    Duration::from_secs(
                        timeout.unwrap_or(crate::config::DEFAULT_EXTENSION_TIMEOUT),
                    ),
                );
                Box::new(McpClient::new(service))
            }
            ExtensionConfig::Builtin {
                name,
                display_name: _,
                timeout,
                bundled: _,
            } => {
                // For builtin extensions, we run the current executable with mcp and extension name
                let cmd = std::env::current_exe()
                    .expect("should find the current executable")
                    .to_str()
                    .expect("should resolve executable to string path")
                    .to_string();
                let transport = StdioTransport::new(
                    &cmd,
                    vec!["mcp".to_string(), name.clone()],
                    HashMap::new(),
                );
                let handle = transport.start().await?;
                let service = McpService::with_timeout(
                    handle,
                    Duration::from_secs(
                        timeout.unwrap_or(crate::config::DEFAULT_EXTENSION_TIMEOUT),
                    ),
                );
                Box::new(McpClient::new(service))
            }
            _ => unreachable!(),
        };

        // Initialize the client with default capabilities
        let info = ClientInfo {
            name: "goose".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        let capabilities = ClientCapabilities::default();

        let init_result = client
            .initialize(info, capabilities)
            .await
            .map_err(|e| ExtensionError::Initialization(config.clone(), e))?;

        // Store instructions if provided
        if let Some(instructions) = init_result.instructions {
            self.instructions
                .insert(sanitized_name.clone(), instructions);
        }

        // if the server is capable if resources we track it
        if init_result.capabilities.resources.is_some() {
            self.resource_capable_extensions
                .insert(sanitized_name.clone());
        }

        // Store the client using the provided name
        self.clients
            .insert(sanitized_name.clone(), Arc::new(Mutex::new(client)));

        Ok(())
    }

    /// Get extensions info
    pub async fn get_extensions_info(&self) -> Vec<ExtensionInfo> {
        self.clients
            .keys()
            .map(|name| {
                let instructions = self.instructions.get(name).cloned().unwrap_or_default();
                let has_resources = self.resource_capable_extensions.contains(name);
                ExtensionInfo::new(name, &instructions, has_resources)
            })
            .collect()
    }

    /// Get aggregated usage statistics
    pub async fn remove_extension(&mut self, name: &str) -> ExtensionResult<()> {
        let sanitized_name = normalize(name.to_string());

        self.clients.remove(&sanitized_name);
        self.instructions.remove(&sanitized_name);
        self.resource_capable_extensions.remove(&sanitized_name);
        Ok(())
    }

    pub async fn list_extensions(&self) -> ExtensionResult<Vec<String>> {
        Ok(self.clients.keys().cloned().collect())
    }

    /// Get all tools from all clients with proper prefixing
    pub async fn get_prefixed_tools(
        &self,
        extension_name: Option<String>,
    ) -> ExtensionResult<Vec<Tool>> {
        // Filter clients based on the provided extension_name or include all if None
        let filtered_clients = self.clients.iter().filter(|(name, _)| {
            if let Some(ref name_filter) = extension_name {
                *name == name_filter
            } else {
                true
            }
        });

        let client_futures = filtered_clients.map(|(name, client)| {
            let name = name.clone();
            let client = client.clone();

            task::spawn(async move {
                let mut tools = Vec::new();
                let client_guard = client.lock().await;
                let mut client_tools = client_guard.list_tools(None).await?;

                loop {
                    for tool in client_tools.tools {
                        tools.push(Tool::new(
                            format!("{}__{}", name, tool.name),
                            &tool.description,
                            tool.input_schema,
                            tool.annotations,
                        ));
                    }

                    // Exit loop when there are no more pages
                    if client_tools.next_cursor.is_none() {
                        break;
                    }

                    client_tools = client_guard.list_tools(client_tools.next_cursor).await?;
                }

                Ok::<Vec<Tool>, ExtensionError>(tools)
            })
        });

        // Collect all results concurrently
        let results = future::join_all(client_futures).await;

        // Aggregate tools and handle errors
        let mut tools = Vec::new();
        for result in results {
            match result {
                Ok(Ok(client_tools)) => tools.extend(client_tools),
                Ok(Err(err)) => return Err(err),
                Err(join_err) => return Err(ExtensionError::from(join_err)),
            }
        }

        Ok(tools)
    }

    /// Get client resources and their contents
    pub async fn get_resources(&self) -> ExtensionResult<Vec<ResourceItem>> {
        let mut result: Vec<ResourceItem> = Vec::new();

        for (name, client) in &self.clients {
            let client_guard = client.lock().await;
            let resources = client_guard.list_resources(None).await?;

            for resource in resources.resources {
                // Skip reading the resource if it's not marked active
                // This avoids blowing up the context with inactive resources
                if !resource.is_active() {
                    continue;
                }

                if let Ok(contents) = client_guard.read_resource(&resource.uri).await {
                    for content in contents.contents {
                        let (uri, content_str) = match content {
                            mcp_core::resource::ResourceContents::TextResourceContents {
                                uri,
                                text,
                                ..
                            } => (uri, text),
                            mcp_core::resource::ResourceContents::BlobResourceContents {
                                uri,
                                blob,
                                ..
                            } => (uri, blob),
                        };

                        result.push(ResourceItem::new(
                            name.clone(),
                            uri,
                            resource.name.clone(),
                            content_str,
                            resource.timestamp().unwrap_or(*DEFAULT_TIMESTAMP),
                            resource.priority().unwrap_or(0.0),
                        ));
                    }
                }
            }
        }
        Ok(result)
    }

    /// Get the extension prompt including client instructions
    pub async fn get_planning_prompt(&self, tools_info: Vec<ToolInfo>) -> String {
        let mut context: HashMap<&str, Value> = HashMap::new();
        context.insert("tools", serde_json::to_value(tools_info).unwrap());

        prompt_template::render_global_file("plan.md", &context).expect("Prompt should render")
    }

    /// Find and return a reference to the appropriate client for a tool call
    fn get_client_for_tool(&self, prefixed_name: &str) -> Option<(&str, McpClientBox)> {
        self.clients
            .iter()
            .find(|(key, _)| prefixed_name.starts_with(*key))
            .map(|(name, client)| (name.as_str(), Arc::clone(client)))
    }

    // Function that gets executed for read_resource tool
    pub async fn read_resource(&self, params: Value) -> Result<Vec<Content>, ToolError> {
        let uri = params
            .get("uri")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParameters("Missing 'uri' parameter".to_string()))?;

        let extension_name = params.get("extension_name").and_then(|v| v.as_str());

        // If extension name is provided, we can just look it up
        if extension_name.is_some() {
            let result = self
                .read_resource_from_extension(uri, extension_name.unwrap())
                .await?;
            return Ok(result);
        }

        // If extension name is not provided, we need to search for the resource across all extensions
        // Loop through each extension and try to read the resource, don't raise an error if the resource is not found
        // TODO: do we want to find if a provided uri is in multiple extensions?
        // currently it will return the first match and skip any others
        for extension_name in self.resource_capable_extensions.iter() {
            let result = self.read_resource_from_extension(uri, extension_name).await;
            match result {
                Ok(result) => return Ok(result),
                Err(_) => continue,
            }
        }

        // None of the extensions had the resource so we raise an error
        let available_extensions = self
            .clients
            .keys()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>()
            .join(", ");
        let error_msg = format!(
            "Resource with uri '{}' not found. Here are the available extensions: {}",
            uri, available_extensions
        );

        Err(ToolError::InvalidParameters(error_msg))
    }

    async fn read_resource_from_extension(
        &self,
        uri: &str,
        extension_name: &str,
    ) -> Result<Vec<Content>, ToolError> {
        let available_extensions = self
            .clients
            .keys()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>()
            .join(", ");
        let error_msg = format!(
            "Extension '{}' not found. Here are the available extensions: {}",
            extension_name, available_extensions
        );

        let client = self
            .clients
            .get(extension_name)
            .ok_or(ToolError::InvalidParameters(error_msg))?;

        let client_guard = client.lock().await;
        let read_result = client_guard.read_resource(uri).await.map_err(|_| {
            ToolError::ExecutionError(format!("Could not read resource with uri: {}", uri))
        })?;

        let mut result = Vec::new();
        for content in read_result.contents {
            // Only reading the text resource content; skipping the blob content cause it's too long
            if let mcp_core::resource::ResourceContents::TextResourceContents { text, .. } = content
            {
                let content_str = format!("{}\n\n{}", uri, text);
                result.push(Content::text(content_str));
            }
        }
        
        // Scan the resource content for security threats
        if let Ok(Some(scan_result)) = self.security_manager.scan_content(&result).await {
            // Get safe content based on security policy
            let safe_content = self.security_manager.get_safe_content(&result, &scan_result);
            
            // If content was modified, log it
            if safe_content != result {
                tracing::info!(
                    uri = uri,
                    extension = extension_name,
                    "Resource content was modified by security scanner: {}",
                    scan_result.explanation
                );
            }
            
            return Ok(safe_content);
        } else {
            tracing::info!(
                uri = uri,
                extension = extension_name,
                "Security scanning skipped or failed for resource content"
            );
        }

        Ok(result)
    }

    async fn list_resources_from_extension(
        &self,
        extension_name: &str,
    ) -> Result<Vec<Content>, ToolError> {
        let client = self.clients.get(extension_name).ok_or_else(|| {
            ToolError::InvalidParameters(format!("Extension {} is not valid", extension_name))
        })?;

        let client_guard = client.lock().await;
        client_guard
            .list_resources(None)
            .await
            .map_err(|e| {
                ToolError::ExecutionError(format!(
                    "Unable to list resources for {}, {:?}",
                    extension_name, e
                ))
            })
            .map(|lr| {
                let resource_list = lr
                    .resources
                    .into_iter()
                    .map(|r| format!("{} - {}, uri: ({})", extension_name, r.name, r.uri))
                    .collect::<Vec<String>>()
                    .join("\n");

                vec![Content::text(resource_list)]
            })
    }

    pub async fn list_resources(&self, params: Value) -> Result<Vec<Content>, ToolError> {
        let extension = params.get("extension").and_then(|v| v.as_str());

        match extension {
            Some(extension_name) => {
                // Handle single extension case
                self.list_resources_from_extension(extension_name).await
            }
            None => {
                // Handle all extensions case using FuturesUnordered
                let mut futures = FuturesUnordered::new();

                // Create futures for each resource_capable_extension
                for extension_name in &self.resource_capable_extensions {
                    futures.push(async move {
                        self.list_resources_from_extension(extension_name).await
                    });
                }

                let mut all_resources = Vec::new();
                let mut errors = Vec::new();

                // Process results as they complete
                while let Some(result) = futures.next().await {
                    match result {
                        Ok(content) => {
                            all_resources.extend(content);
                        }
                        Err(tool_error) => {
                            errors.push(tool_error);
                        }
                    }
                }

                // Log any errors that occurred
                if !errors.is_empty() {
                    tracing::error!(
                        errors = ?errors
                            .into_iter()
                            .map(|e| format!("{:?}", e))
                            .collect::<Vec<_>>(),
                        "errors from listing resources"
                    );
                }

                Ok(all_resources)
            }
        }
    }

    pub async fn dispatch_tool_call(&self, tool_call: ToolCall) -> ToolResult<Vec<Content>> {
        // Dispatch tool call based on the prefix naming convention
        let (client_name, client) = self
            .get_client_for_tool(&tool_call.name)
            .ok_or_else(|| ToolError::NotFound(tool_call.name.clone()))?;

        // rsplit returns the iterator in reverse, tool_name is then at 0
        let tool_name = tool_call
            .name
            .strip_prefix(client_name)
            .and_then(|s| s.strip_prefix("__"))
            .ok_or_else(|| ToolError::NotFound(tool_call.name.clone()))?;

        let client_guard = client.lock().await;

        let result = client_guard
            .call_tool(tool_name, tool_call.clone().arguments)
            .await
            .map(|result| result.content)
            .map_err(|e| ToolError::ExecutionError(e.to_string()));
            
        info!(
            "input" = serde_json::to_string(&tool_call).unwrap(),
            "output" = serde_json::to_string(&result).unwrap(),
        );
        
        // If result is an error, just return it
        if result.is_err() {
            return result;
        }
        
        let content = result.as_ref().unwrap();
        
        // Scan the tool result for security threats
        if let Ok(Some(scan_result)) = self.security_manager.scan_tool_result(
            &tool_call.name, 
            &tool_call.arguments, 
            content
        ).await {
            // Get safe content based on security policy
            let safe_content = self.security_manager.get_safe_content(content, &scan_result);
            
            // If content was modified, log it
            if safe_content != *content {
                tracing::info!(
                    tool = tool_call.name,
                    "Tool result was modified by security scanner: {}",
                    scan_result.explanation
                );
            }
            
            return Ok(safe_content);
        } else {
            tracing::info!(
                tool = tool_call.name,
                "Security scanning skipped or failed for tool result"
            );
        }
        
        // Return original result if scanning is disabled or failed
        result
    }

    pub async fn list_prompts_from_extension(
        &self,
        extension_name: &str,
    ) -> Result<Vec<Prompt>, ToolError> {
        let client = self.clients.get(extension_name).ok_or_else(|| {
            ToolError::InvalidParameters(format!("Extension {} is not valid", extension_name))
        })?;

        let client_guard = client.lock().await;
        client_guard
            .list_prompts(None)
            .await
            .map_err(|e| {
                ToolError::ExecutionError(format!(
                    "Unable to list prompts for {}, {:?}",
                    extension_name, e
                ))
            })
            .map(|lp| lp.prompts)
    }

    pub async fn list_prompts(&self) -> Result<HashMap<String, Vec<Prompt>>, ToolError> {
        let mut futures = FuturesUnordered::new();

        for extension_name in self.clients.keys() {
            futures.push(async move {
                (
                    extension_name,
                    self.list_prompts_from_extension(extension_name).await,
                )
            });
        }

        let mut all_prompts = HashMap::new();
        let mut errors = Vec::new();

        // Process results as they complete
        while let Some(result) = futures.next().await {
            let (name, prompts) = result;
            match prompts {
                Ok(content) => {
                    all_prompts.insert(name.to_string(), content);
                }
                Err(tool_error) => {
                    errors.push(tool_error);
                }
            }
        }

        // Log any errors that occurred
        if !errors.is_empty() {
            tracing::debug!(
                errors = ?errors
                    .into_iter()
                    .map(|e| format!("{:?}", e))
                    .collect::<Vec<_>>(),
                "errors from listing prompts"
            );
        }

        Ok(all_prompts)
    }

    pub async fn get_prompt(
        &self,
        extension_name: &str,
        name: &str,
        arguments: Value,
    ) -> Result<GetPromptResult> {
        let client = self
            .clients
            .get(extension_name)
            .ok_or_else(|| anyhow::anyhow!("Extension {} not found", extension_name))?;

        let client_guard = client.lock().await;
        client_guard
            .get_prompt(name, arguments)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get prompt: {}", e))
    }

    pub async fn search_available_extensions(&self) -> Result<Vec<Content>, ToolError> {
        let mut output_parts = vec![];

        // First get disabled extensions from current config
        let mut disabled_extensions: Vec<String> = vec![];
        for extension in ExtensionConfigManager::get_all().expect("should load extensions") {
            if !extension.enabled {
                let config = extension.config.clone();
                let description = match &config {
                    ExtensionConfig::Builtin {
                        name, display_name, ..
                    } => {
                        // For builtin extensions, use display name if available
                        display_name
                            .as_ref()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| name.clone())
                    }
                    ExtensionConfig::Sse {
                        description, name, ..
                    }
                    | ExtensionConfig::Stdio {
                        description, name, ..
                    } => {
                        // For SSE/Stdio, use description if available
                        description
                            .as_ref()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("Extension '{}'", name))
                    }
                    ExtensionConfig::Frontend { name, .. } => {
                        format!("Frontend extension '{}'", name)
                    }
                };
                disabled_extensions.push(format!("- {} - {}", config.name(), description));
            }
        }

        if !disabled_extensions.is_empty() {
            output_parts.push(format!(
                "Currently available extensions user can enable:\n{}\n",
                disabled_extensions.join("\n")
            ));
        } else {
            output_parts
                .push("No available extensions found in current configuration.\n".to_string());
        }

        Ok(vec![Content::text(output_parts.join("\n"))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcp_client::client::Error;
    use mcp_client::client::McpClientTrait;
    use mcp_core::protocol::{
        CallToolResult, GetPromptResult, InitializeResult, ListPromptsResult, ListResourcesResult,
        ListToolsResult, ReadResourceResult,
    };
    use serde_json::json;

    struct MockClient {}

    #[async_trait::async_trait]
    impl McpClientTrait for MockClient {
        async fn initialize(
            &mut self,
            _info: ClientInfo,
            _capabilities: ClientCapabilities,
        ) -> Result<InitializeResult, Error> {
            Err(Error::NotInitialized)
        }

        async fn list_resources(
            &self,
            _next_cursor: Option<String>,
        ) -> Result<ListResourcesResult, Error> {
            Err(Error::NotInitialized)
        }

        async fn read_resource(&self, _uri: &str) -> Result<ReadResourceResult, Error> {
            Err(Error::NotInitialized)
        }

        async fn list_tools(&self, _next_cursor: Option<String>) -> Result<ListToolsResult, Error> {
            Err(Error::NotInitialized)
        }

        async fn call_tool(&self, name: &str, _arguments: Value) -> Result<CallToolResult, Error> {
            match name {
                "tool" | "test__tool" => Ok(CallToolResult {
                    content: vec![],
                    is_error: None,
                }),
                _ => Err(Error::NotInitialized),
            }
        }

        async fn list_prompts(
            &self,
            _next_cursor: Option<String>,
        ) -> Result<ListPromptsResult, Error> {
            Err(Error::NotInitialized)
        }

        async fn get_prompt(
            &self,
            _name: &str,
            _arguments: Value,
        ) -> Result<GetPromptResult, Error> {
            Err(Error::NotInitialized)
        }
    }

    #[test]
    fn test_get_client_for_tool() {
        let mut extension_manager = ExtensionManager::new();

        // Add some mock clients
        extension_manager.clients.insert(
            normalize("test_client".to_string()),
            Arc::new(Mutex::new(Box::new(MockClient {}))),
        );

        extension_manager.clients.insert(
            normalize("__client".to_string()),
            Arc::new(Mutex::new(Box::new(MockClient {}))),
        );

        extension_manager.clients.insert(
            normalize("__cli__ent__".to_string()),
            Arc::new(Mutex::new(Box::new(MockClient {}))),
        );

        extension_manager.clients.insert(
            normalize("client 🚀".to_string()),
            Arc::new(Mutex::new(Box::new(MockClient {}))),
        );

        // Test basic case
        assert!(extension_manager
            .get_client_for_tool("test_client__tool")
            .is_some());

        // Test leading underscores
        assert!(extension_manager
            .get_client_for_tool("__client__tool")
            .is_some());

        // Test multiple underscores in client name, and ending with __
        assert!(extension_manager
            .get_client_for_tool("__cli__ent____tool")
            .is_some());

        // Test unicode in tool name, "client 🚀" should become "client_"
        assert!(extension_manager
            .get_client_for_tool("client___tool")
            .is_some());
    }

    #[tokio::test]
    async fn test_dispatch_tool_call() {
        // test that dispatch_tool_call parses out the sanitized name correctly, and extracts
        // tool_names
        let mut extension_manager = ExtensionManager::new();

        // Add some mock clients
        extension_manager.clients.insert(
            normalize("test_client".to_string()),
            Arc::new(Mutex::new(Box::new(MockClient {}))),
        );

        extension_manager.clients.insert(
            normalize("__cli__ent__".to_string()),
            Arc::new(Mutex::new(Box::new(MockClient {}))),
        );

        extension_manager.clients.insert(
            normalize("client 🚀".to_string()),
            Arc::new(Mutex::new(Box::new(MockClient {}))),
        );

        // verify a normal tool call
        let tool_call = ToolCall {
            name: "test_client__tool".to_string(),
            arguments: json!({}),
        };

        let result = extension_manager.dispatch_tool_call(tool_call).await;
        assert!(result.is_ok());

        let tool_call = ToolCall {
            name: "test_client__test__tool".to_string(),
            arguments: json!({}),
        };

        let result = extension_manager.dispatch_tool_call(tool_call).await;
        assert!(result.is_ok());

        // verify a multiple underscores dispatch
        let tool_call = ToolCall {
            name: "__cli__ent____tool".to_string(),
            arguments: json!({}),
        };

        let result = extension_manager.dispatch_tool_call(tool_call).await;
        assert!(result.is_ok());

        // Test unicode in tool name, "client 🚀" should become "client_"
        let tool_call = ToolCall {
            name: "client___tool".to_string(),
            arguments: json!({}),
        };

        let result = extension_manager.dispatch_tool_call(tool_call).await;
        assert!(result.is_ok());

        let tool_call = ToolCall {
            name: "client___test__tool".to_string(),
            arguments: json!({}),
        };

        let result = extension_manager.dispatch_tool_call(tool_call).await;
        assert!(result.is_ok());

        // this should error out, specifically for an ToolError::ExecutionError
        let invalid_tool_call = ToolCall {
            name: "client___tools".to_string(),
            arguments: json!({}),
        };

        let result = extension_manager
            .dispatch_tool_call(invalid_tool_call)
            .await;
        assert!(matches!(
            result.err().unwrap(),
            ToolError::ExecutionError(_)
        ));

        // this should error out, specifically with an ToolError::NotFound
        // this client doesn't exist
        let invalid_tool_call = ToolCall {
            name: "_client__tools".to_string(),
            arguments: json!({}),
        };

        let result = extension_manager
            .dispatch_tool_call(invalid_tool_call)
            .await;
        assert!(matches!(result.err().unwrap(), ToolError::NotFound(_)));
    }
}
