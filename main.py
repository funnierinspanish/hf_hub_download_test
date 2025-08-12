import os
from huggingface_hub import hf_hub_download, snapshot_download

repo_id = os.getenv("HUB_REPO_ID", "jer030/nanogpt")
repo_type = os.getenv("HUB_REPO_TYPE", "dataset")
download_type = os.getenv("HUB_DOWNLOAD_TYPE", "snapshot_download")
hf_home = os.getenv("HF_HOME")
cache_dir_env_var_value = os.getenv("HF_HUB_CACHE")
filename = os.getenv("HUB_FILE_NAME", "config.json")
default_cache_dir = "/output/.cache"

if cache_dir_env_var_value and not cache_dir_env_var_value.startswith("/output/"):
  print(f"Warning: The cache directory {cache_dir_env_var_value} does not start with '/output/'. If you manually set it, then it might have been overwritten. Using default cache directory instead: {default_cache_dir}")
  cache_dir_env_var_value = None

cache_dir = cache_dir_env_var_value if cache_dir_env_var_value else default_cache_dir

print("\n")
print("=======================================================")
print("--------------------------------------------------")
print(f"HF_HOME: {hf_home}")
print("--------------------------------------------------")
print(f"Download type: {download_type}")
print(f"Repository ID: {repo_id}")
print(f"Repository type: {repo_type}")
print(f"Target cache directory: {cache_dir}")
print("=======================================================")
print("🚀\n\n")

if download_type == "snapshot_download":
  path = snapshot_download(repo_id=repo_id, repo_type=repo_type, cache_dir=cache_dir)
elif download_type == "hf_hub_download":
  path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename, cache_dir=cache_dir)
else:
  raise ValueError(f"Unknown download type: {download_type}")

print(f"\n\nDownloaded file path:\n  {path}\n\n")
