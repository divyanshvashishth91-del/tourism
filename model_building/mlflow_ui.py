from pyngrok import ngrok
import subprocess, os
import mlflow

# Use the token you entered via the hidden prompt earlier
ngrok.set_auth_token(os.environ["NGROK_TOKEN"])

# Start MLflow UI on port 5000 (reads/writes ./mlruns by default)
process = subprocess.Popen(["mlflow", "ui", "--port", "5000"])

# Create public tunnel
public_url = ngrok.connect(5000).public_url
print("MLflow UI is available at:", public_url)
