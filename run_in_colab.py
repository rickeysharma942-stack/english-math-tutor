# Run these as separate Colab/Kaggle cells (GPU runtime).

# Cell A - install (-U makes sure Streamlit is recent enough for every feature)
!pip install -q -U "streamlit>=1.40" transformers accelerate peft bitsandbytes sympy requests
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

# Cell B - get the app code
!git clone https://github.com/rickeysharma942-stack/english-math-tutor.git
%cd english-math-tutor

# Cell C - start the app and the public tunnel
import os, subprocess, time
os.environ["BASE_MODEL"] = "Qwen/Qwen3-8B"
# Optional: keep chats after the session ends (mount Google Drive first on Colab)
# os.environ["CHAT_DIR"] = "/content/drive/MyDrive/tutor_chats"
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])
time.sleep(5)
subprocess.Popen("../cloudflared tunnel --url http://localhost:8501 > tunnel.log 2>&1", shell=True)
time.sleep(8)
print([l for l in open("tunnel.log") if "trycloudflare.com" in l])   # open this link
