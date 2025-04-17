pub mod base;
mod experiments;
pub mod extensions;
pub mod permission;

pub use crate::agents::ExtensionConfig;
pub use base::{Config, ConfigError, APP_STRATEGY};
pub use experiments::ExperimentManager;
pub use extensions::{ExtensionConfigManager, ExtensionEntry};
pub use permission::PermissionManager;

pub use extensions::DEFAULT_DISPLAY_NAME;
pub use extensions::DEFAULT_EXTENSION;
pub use extensions::DEFAULT_EXTENSION_DESCRIPTION;
pub use extensions::DEFAULT_EXTENSION_TIMEOUT;

// Re-export security config
pub use crate::security::config::SecurityConfig;
