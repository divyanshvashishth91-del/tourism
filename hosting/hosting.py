from huggingface_hub import HfApi
import os

# Uses your HF token from env (set this earlier in Colab or via GitHub Actions secret)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the Dockerized Streamlit app (Dockerfile, requirements.txt, app.py) to your Space
api.upload_folder(
    folder_path="tourism/deployment",         # local folder with deployment assets
    repo_id="Ansh91/tourism",                 # your Space
    repo_type="space",                        # we're deploying to a Space
    path_in_repo="",                          # upload at repo root
    commit_message="Update tourism deployment assets"
)

print("✅ Uploaded deployment assets to Space: Ansh91/tourism")
