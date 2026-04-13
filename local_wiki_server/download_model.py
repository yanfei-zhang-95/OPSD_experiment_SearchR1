#!/usr/bin/env python3
"""
Download Hugging Face model to local directory.
"""
import argparse
from huggingface_hub import snapshot_download
import os

def download_model(model_id: str, local_dir: str):
    """
    Download a Hugging Face model to a local directory.
    
    Args:
        model_id: Hugging Face model ID (e.g., "intfloat/e5-base-v2")
        local_dir: Local directory to save the model
    """
    print(f"Downloading model {model_id} to {local_dir}...")
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Download the model
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"Model downloaded successfully to {local_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hugging Face model to local directory.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="intfloat/e5-base-v2",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./wiki-18-e5-index/intfloat/e5-base-v2",
        help="Local directory to save the model"
    )
    
    args = parser.parse_args()
    
    download_model(args.model_id, args.local_dir)
