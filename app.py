import os
from pathlib import Path
from contextlib import nullcontext

import requests
import torch
import streamlit as st
import sympy as sp
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------- EDIT THESE ----------------
GITHUB_REPO = "YOUR_GITHUB_USERNAME/YOUR_REPO"   # repo that has the release
RELEASE_TAG = "v1.0"                             # the release tag that holds the adapter files

# File names exactly as you uploaded them to the GitHub Release
ASSETS = {
    "english_adapter": {
        "adapter_config.json": "english_adapter_config.json",
        "adapter_model.safetensors": "english_adapter_model.safetensors",
    },
    "math_adapter": {
        "adapter_config.json": "math_adapter_config.json",
        "adapter_model.safetensors": "math_adapter_model.safetensors",
    },
}
# --------------------------------------------

ADAPTER_DIR = Path("adapters")
x = sp.Symbol("x")

st.set_page_config(page_title="English + Math Tutor", page_icon="🧠")
st.title("🧠 English + Math Tutor")


def get_base_model_id():
    """Base model id comes from an environment variable or a Streamlit secret (not hard-coded)."""
    value = os.environ.get("BASE_MODEL")
    if not value:
        try:
            value = st.secrets.get("BASE_MODEL")
        except Exception:
            value = None
    return value


BASE_MODEL = get_base_model_id()
if not BASE_MODEL:
    st.error("BASE_MODEL is not set. Set it as an environment variable or a Streamlit secret.")
    st.stop()

if not torch.cuda.is_available():
    st.error("No GPU found. This model needs an NVIDIA GPU (for example a Colab or Kaggle T4).")
    st.stop()


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    tmp.replace(dest)


def ensure_adapters():
    """Download adapter files from the GitHub Release if they are not already on disk."""
    base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"
    for adapter_name, files in ASSETS.items():
        for local_name, asset_name in files.items():
            dest = ADAPTER_DIR / adapter_name / local_name
            if not dest.exists():
                download_file(f"{base_url}/{asset_name}", dest)


@st.cache_resource(show_spinner="Loading model (first run downloads files and takes a few minutes)...")
def load_model():
    ensure_adapters()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map={"": 0}, dtype=torch.float16
    )
    model = PeftModel.from_pretrained(
        base, str(ADAPTER_DIR / "english_adapter"), adapter_name="english_adapter"
    )
    model.load_adapter(str(ADAPTER_DIR / "math_adapter"), adapter_name="math_adapter")
    model.eval()
    return tok, model


try:
    tokenizer, model = load_model()
except requests.HTTPError as e:
    st.error(f"Could not download adapter files from the GitHub Release: {e}. "
             "Check GITHUB_REPO, RELEASE_TAG and the asset names, and make sure the repo is public.")
    st.stop()


def generate_response(prompt, mode, math_expression=None):
    ctx = nullcontext()
    user_msg = prompt
    if mode == "English":
        model.set_adapter("english_adapter")
    elif mode == "Math":
        model.set_adapter("math_adapter")
        if math_expression:
            try:
                result = sp.simplify(sp.sympify(math_expression, locals={"x": x}))
                user_msg = f"{prompt}\n[SymPy result: {result}]"
            except Exception as e:
                user_msg = f"{prompt}\n[SymPy parsing error: {e}]"
    else:
        ctx = model.disable_adapter()

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad(), ctx:
        out = model.generate(
            **inputs, max_new_tokens=250, temperature=0.7, top_p=0.8,
            do_sample=True, repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


mode = st.radio("Mode", ["General", "English", "Math"], horizontal=True)
prompt = st.text_area("Your question", height=100)
expr = None
if mode == "Math":
    expr = st.text_input("Math expression (SymPy syntax)", value="diff(x**3 + 5*x, x)")

if st.button("Generate") and prompt.strip():
    with st.spinner("Thinking..."):
        st.write(generate_response(prompt, mode, expr))
