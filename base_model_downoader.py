from huggingface_hub import snapshot_download

# Define model repository name
model_name = "stabilityai/stable-diffusion-xl-base-1.0"

# Download the model (this may take some time)
model_path = snapshot_download(repo_id=model_name, local_dir="sdxl_base_1_0")

print(f"Model downloaded to: {model_path}")
