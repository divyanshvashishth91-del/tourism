from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Resolve repo root and deployment dir robustly
ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT / "deployment"

if not DEPLOY_DIR.is_dir():
    raise SystemExit(f"❌ Deploy folder not found: {DEPLOY_DIR}")

api = HfApi(token=os.getenv("HF_TOKEN"))

# Ensure Space exists (SDK Docker/Streamlit already created per your prereqs)
space_id = "Ansh91/tourism"
try:
    api.repo_info(repo_id=space_id, repo_type="space")
    print(f"✅ Space '{space_id}' exists.")
except RepositoryNotFoundError:
    print(f"ℹ️ Space '{space_id}' not found. Creating it (public)...")
    create_repo(repo_id=space_id, repo_type="space", private=False)

# Upload contents of ./deployment to the Space root
print(f"⬆️ Uploading folder: {DEPLOY_DIR}")
api.upload_folder(
    folder_path=str(DEPLOY_DIR),
    repo_id=space_id,
    repo_type="space",
    path_in_repo="",  # put files at Space root
    commit_message="Update tourism Docker Streamlit app",
)
print("✅ Uploaded deployment assets to Space root.")

