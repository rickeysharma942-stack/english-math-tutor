# Run these as separate Colab/Kaggle cells (GPU runtime).

# Cell A - install
!pip install -q streamlit transformers accelerate peft bitsandbytes sympy requests
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

# Cell B - get the app code
!git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO.git
%cd YOUR_REPO

# Cell C - set the base model id (not stored in the repo), start the app and the public tunnel
import os, subprocess, time
os.environ["BASE_MODEL"] = "PUT_BASE_MODEL_ID_HERE"
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])
time.sleep(5)
subprocess.Popen("../cloudflared tunnel --url http://localhost:8501 > tunnel.log 2>&1", shell=True)
time.sleep(8)
print([l for l in open("tunnel.log") if "trycloudflare.com" in l])   # open this link
