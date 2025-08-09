use anyhow::{anyhow};
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;
use tokio::sync::OnceCell;

pub struct ModelDownloader {
    cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new() -> anyhow::Result<Self> {
        // Use platform-appropriate cache directory
        let cache_dir = if let Some(cache_dir) = dirs::cache_dir() {
            cache_dir.join("goose").join("security_models")
        } else {
            // Fallback to home directory
            dirs::home_dir()
                .ok_or_else(|| anyhow!("Could not determine home directory"))?
                .join(".cache")
                .join("goose")
                .join("security_models")
        };

        Ok(Self { cache_dir })
    }

    pub async fn ensure_model_available(&self, model_info: &ModelInfo) -> anyhow::Result<(PathBuf, PathBuf)> {
        let model_path = self.cache_dir.join(&model_info.onnx_filename);
        let tokenizer_path = self.cache_dir.join(&model_info.tokenizer_filename);

        // Check if both model and tokenizer exist
        if model_path.exists() && tokenizer_path.exists() {
            tracing::info!(
                model = %model_info.hf_model_name,
                path = ?model_path,
                "Using cached ONNX model"
            );
            return Ok((model_path, tokenizer_path));
        }

        tracing::info!(
            model = %model_info.hf_model_name,
            "🔒 Goose is being set up, this could take up to a minute…"
        );

        // Create cache directory if it doesn't exist
        fs::create_dir_all(&self.cache_dir).await?;

        // Download and convert the model - this blocks until complete
        self.download_and_convert_model(model_info).await?;

        // Verify the files were created
        if !model_path.exists() || !tokenizer_path.exists() {
            return Err(anyhow!(
                "Model conversion completed but files not found at expected paths. Model: {:?}, Tokenizer: {:?}",
                model_path, tokenizer_path
            ));
        }

        tracing::info!(
            model = %model_info.hf_model_name,
            model_path = ?model_path,
            tokenizer_path = ?tokenizer_path,
            "✅ Successfully downloaded and converted model"
        );

        Ok((model_path, tokenizer_path))
    }

    async fn download_and_convert_model(&self, model_info: &ModelInfo) -> anyhow::Result<()> {
        // Set up Python virtual environment with required dependencies
        let venv_dir = self.cache_dir.join("python_venv");
        self.ensure_python_venv(&venv_dir).await?;
        
        let python_script = self.create_conversion_script(model_info).await?;
        
        tracing::info!("Running model conversion script in virtual environment...");
        
        // Use the virtual environment's Python
        let python_exe = if cfg!(windows) {
            venv_dir.join("Scripts").join("python.exe")
        } else {
            venv_dir.join("bin").join("python")
        };
        
        let output = Command::new(&python_exe)
            .arg(&python_script)
            .env("CACHE_DIR", &self.cache_dir)
            .env("MODEL_NAME", &model_info.hf_model_name)
            .output()
            .map_err(|e| anyhow!("Failed to execute Python conversion script: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(anyhow!(
                "Model conversion failed:\nStdout: {}\nStderr: {}",
                stdout,
                stderr
            ));
        }

        // Clean up the temporary script
        let _ = fs::remove_file(&python_script).await;

        Ok(())
    }

    async fn ensure_python_venv(&self, venv_dir: &std::path::Path) -> anyhow::Result<()> {
        // Check if virtual environment already exists and has required packages
        let python_exe = if cfg!(windows) {
            venv_dir.join("Scripts").join("python.exe")
        } else {
            venv_dir.join("bin").join("python")
        };

        if python_exe.exists() {
            // Check if required packages are installed
            let output = Command::new(&python_exe)
                .args(&["-c", "import torch, transformers, onnx, tokenizers; print('OK')"])
                .output();
            
            if let Ok(output) = output {
                if output.status.success() && String::from_utf8_lossy(&output.stdout).trim() == "OK" {
                    tracing::info!("Python virtual environment already set up with required packages");
                    return Ok(());
                }
            }
        }

        tracing::info!("Setting up Python virtual environment...");

        // Create virtual environment
        fs::create_dir_all(venv_dir).await?;
        
        let output = Command::new("python3")
            .args(&["-m", "venv", venv_dir.to_str()
                .ok_or_else(|| anyhow!("Invalid venv directory path"))?])
            .output()
            .map_err(|e| anyhow!("Failed to create Python virtual environment: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Failed to create virtual environment: {}", stderr));
        }

        tracing::info!("Installing required Python packages...");

        // Install required packages
        let pip_exe = if cfg!(windows) {
            venv_dir.join("Scripts").join("pip.exe")
        } else {
            venv_dir.join("bin").join("pip")
        };

        let packages = [
            "torch",
            "transformers", 
            "onnx",
            "tokenizers",
        ];

        for package in &packages {
            tracing::info!("Installing {}...", package);
            let output = Command::new(&pip_exe)
                .args(&["install", package])
                .output()
                .map_err(|e| anyhow!("Failed to install {}: {}", package, e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(anyhow!("Failed to install {}: {}", package, stderr));
            }
        }

        tracing::info!("Python virtual environment setup complete");
        Ok(())
    }

    async fn create_conversion_script(&self, model_info: &ModelInfo) -> anyhow::Result<PathBuf> {
        let script_content = format!(
            r#"#!/usr/bin/env python3
"""
Runtime model conversion script for Goose security models
"""

import os
import sys

def install_packages():
    """Install required packages"""
    import subprocess
    packages = ["torch", "transformers", "onnx", "tokenizers"]
    for package in packages:
        print(f"📦 Installing {{package}}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {{package}}: {{e}}")
            return False
    return True

def check_and_install_packages():
    """Check if packages are available, install if needed"""
    try:
        import torch
        import transformers
        import onnx
        import tokenizers
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing packages: {{e}}")
        print("📦 Installing required packages...")
        if install_packages():
            # Try importing again after installation
            try:
                import torch
                import transformers
                import onnx
                import tokenizers
                print("✅ Successfully installed and imported all packages")
                return True
            except ImportError as e2:
                print(f"❌ Still missing packages after installation: {{e2}}")
                return False
        else:
            return False

# Check and install packages first
if not check_and_install_packages():
    print("❌ Failed to install required packages")
    sys.exit(1)

# Now import everything we need
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def convert_model_to_onnx(model_name: str, output_dir: str):
    """Convert a Hugging Face model to ONNX format"""
    print(f"Converting {{model_name}} to ONNX...")

    # Create output directory
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Handle authentication for gated models
        hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
        auth_kwargs = {{}}
        if hf_token:
            auth_kwargs['token'] = hf_token
            print(f"   Using HF token for authentication")

        # Load model and tokenizer
        print(f"   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_kwargs)

        print(f"   Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **auth_kwargs)
        model.eval()

        # Create dummy input
        dummy_text = "This is a test input for ONNX conversion"
        inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Export to ONNX
        model_filename = model_name.replace("/", "_") + ".onnx"
        model_path = os.path.join(output_dir, model_filename)

        print(f"   Exporting to ONNX...")
        torch.onnx.export(
            model,
            (inputs['input_ids'], inputs['attention_mask']),
            model_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={{
                'input_ids': {{0: 'batch_size', 1: 'sequence'}},
                'attention_mask': {{0: 'batch_size', 1: 'sequence'}},
                'logits': {{0: 'batch_size'}}
            }}
        )

        # Save tokenizer with model-specific filename
        tokenizer_filename = model_name.replace("/", "_") + "_tokenizer.json"
        tokenizer_path = os.path.join(output_dir, tokenizer_filename)
        
        # First save to temp directory to get the tokenizer.json file
        temp_dir = os.path.join(output_dir, "temp_tokenizer")
        tokenizer.save_pretrained(temp_dir, legacy_format=False)
        
        # Copy the tokenizer.json file to the expected location with model-specific name
        import shutil
        temp_tokenizer_json = os.path.join(temp_dir, "tokenizer.json")
        if os.path.exists(temp_tokenizer_json):
            shutil.copy2(temp_tokenizer_json, tokenizer_path)
            # Clean up temp directory
            shutil.rmtree(temp_dir)
        else:
            print(f"   Warning: tokenizer.json not found in {{temp_dir}}")
            # Fallback: save the entire tokenizer directory
            tokenizer_dir = os.path.join(output_dir, model_name.replace("/", "_") + "_tokenizer")
            tokenizer.save_pretrained(tokenizer_dir, legacy_format=False)
            print(f"   Saved tokenizer to directory: {{tokenizer_dir}}")

        print(f"✅ Successfully converted {{model_name}}")
        print(f"   Model: {{model_path}}")
        print(f"   Tokenizer: {{tokenizer_path}}")
        return True

    except Exception as e:
        print(f"❌ Failed to convert {{model_name}}: {{e}}")
        if "gated repo" in str(e).lower() or "access" in str(e).lower():
            print(f"   This might be a gated model. Make sure you:")
            print(f"   1. Have access to {{model_name}} on Hugging Face")
            print(f"   2. Set your HF token: export HUGGINGFACE_TOKEN='your_token'")
            print(f"   3. Get a token from: https://huggingface.co/settings/tokens")
        import traceback
        traceback.print_exc()
        return False

def main():
    model_name = os.getenv('MODEL_NAME')
    cache_dir = os.getenv('CACHE_DIR')
    
    if not model_name or not cache_dir:
        print("Error: MODEL_NAME and CACHE_DIR environment variables must be set")
        sys.exit(1)
    
    success = convert_model_to_onnx(model_name, cache_dir)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
"#
        );

        let script_path = self.cache_dir.join(format!("convert_model_{}.py", 
            model_info.hf_model_name.replace("/", "_").replace("-", "_")));
        fs::write(&script_path, script_content).await?;
        
        // Make the script executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script_path).await?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script_path, perms).await?;
        }

        Ok(script_path)
    }

    pub fn get_cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub hf_model_name: String,
    pub onnx_filename: String,
    pub tokenizer_filename: String,
}

impl ModelInfo {
    pub fn deepset_deberta() -> Self {
        Self {
            hf_model_name: "deepset/deberta-v3-base-injection".to_string(),
            onnx_filename: "deepset_deberta-v3-base-injection.onnx".to_string(),
            tokenizer_filename: "deepset_deberta-v3-base-injection_tokenizer.json".to_string(),
        }
    }

    pub fn protectai_deberta() -> Self {
        Self {
            hf_model_name: "protectai/deberta-v3-base-prompt-injection-v2".to_string(),
            onnx_filename: "protectai_deberta-v3-base-prompt-injection-v2.onnx".to_string(),
            tokenizer_filename: "protectai_deberta-v3-base-prompt-injection-v2_tokenizer.json".to_string(),
        }
    }
}

// Global downloader instance
static GLOBAL_DOWNLOADER: OnceCell<ModelDownloader> = OnceCell::const_new();

pub async fn get_global_downloader() -> anyhow::Result<&'static ModelDownloader> {
    GLOBAL_DOWNLOADER
        .get_or_try_init(|| async { ModelDownloader::new() })
        .await
}
