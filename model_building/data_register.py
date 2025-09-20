from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
import sys

# Hugging Face repo info based on your setup
repo_id = "Ansh91/tourism"   # you created this name
repo_type = "dataset"        # data repo (separate from your Space)

# Get token from environment (set in GitHub Actions as HF_TOKEN)
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError(
        "HF_TOKEN environment variable is not set. "
        "Make sure the GitHub Actions secret is configured, or set it in your environment before running."
    )

# Initialize API client
api = HfApi(token=hf_token)

# Step 1: Check if the dataset repo exists; create if missing
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=hf_token)
    print(f"Dataset '{repo_id}' created.")
except HfHubHTTPError as e:
    print(f"[HF Hub Error] {e}")
    sys.exit(1)

# Local folder containing your uploaded CSV in Colab
folder_path = "/content/tourism"  # you loaded /content/tourism/tourism.csv
csv_path = os.path.join(folder_path, "tourism.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"Expected CSV at {csv_path}. Please upload your dataset there before running this script."
    )

# Step 2: Upload the folder (includes tourism.csv) to the dataset repo
print(f"Uploading folder '{folder_path}' to '{repo_id}' ({repo_type})...")
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Upload complete.")
