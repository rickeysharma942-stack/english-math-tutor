import os
import io
import re
import json
import base64
import zipfile
import threading
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

import requests
import torch
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

# ----------------------------- CONFIG -----------------------------

GITHUB_REPO = "rickeysharma942-stack/english-math-tutor"
RELEASE_TAG = "chatbot"

ASSETS = {
    "english_adapter": {
        "adapter_config.json": "english_adapter_config.json",
        "adapter_model.safetensors": "english_adapter_model.safetensors",
    },
    "math_adapter": {
        "adapter_config.json": "math_adapter_config.json",
        "adapter_model.safetensors": "math_adapter_model.safetensors",
    },
    "general_adapter": {
        "adapter_config.json": "general_adapter_config.json",
        "adapter_model.safetensors": "general_adapter_model.safetensors",
    },
}

ADAPTER_DIR = Path("adapters") / RELEASE_TAG
CHAT_DIR = Path(os.environ.get("CHAT_DIR", "chats"))
CHAT_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_CHARS = 16000
MAX_HISTORY_CHARS = 24000

MODES = {
    "General": {
        "icon": "💬",
        "adapter": "general_adapter",
        "blurb": "Ask anything, summarize documents, brainstorm, or analyze files.",
        "placeholder": "Ask anything, or upload PDFs, Word files, spreadsheets, code...",
        "examples": [
            "Summarize this document in simple English.",
            "Explain how HTTPS works to a beginner.",
            "Create a 7-day study plan for Python cybersecurity.",
        ],
        "file_prompt": "Summarize and extract the key insights from this attached file:",
    },
    "English": {
        "icon": "✍️",
        "adapter": "english_adapter",
        "blurb": "Grammar correction, writing improvement, and detailed explanations.",
        "placeholder": "Paste text to correct, or upload a document...",
        "examples": [
            "Correct the grammar and explain every correction: He don't know nothing about it.",
            "Improve this email professionally: Me and her goes to school yesterday.",
            "Explain the difference between affect and effect.",
        ],
        "file_prompt": "Proofread this attached document. Correct grammar, clarity, tone, and explain each major correction:",
    },
    "Math": {
        "icon": "🧮",
        "adapter": "math_adapter",
        "blurb": "Step-by-step math help with optional SymPy verification.",
        "placeholder": "Ask a math question, or upload a worksheet/dataset...",
        "examples": [
            "Explain step by step: differentiate x^3 + 5x",
            "Solve step by step: 2x + 7 = 19",
            "A train travels 180 km in 2.5 hours. Find its average speed.",
        ],
        "file_prompt": "Analyze the problems in this attached file and solve them step by step:",
    },
}

# ----------------------------- UI SETUP -----------------------------

LOGO_PATH = Path(__file__).parent / "logo.png"

if LOGO_PATH.exists():
    logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    PAGE_ICON = str(LOGO_PATH)
    LOGO_HTML = f"""
    <div class="brand">
      <img src="data:image/png;base64,{logo_b64}" class="logo">
      <div>
        <div class="title">Tutor Studio</div>
        <div class="subtitle">English · Math · General AI workspace</div>
      </div>
    </div>
    """
else:
    PAGE_ICON = "✨"
    LOGO_HTML = """
    <div class="brand">
      <div class="logo emoji">✨</div>
      <div>
        <div class="title">Tutor Studio</div>
        <div class="subtitle">English · Math · General AI workspace</div>
      </div>
    </div>
    """

st.set_page_config(
    page_title="English + Math Tutor Studio",
    page_icon=PAGE_ICON,
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1250px; }
      .brand { display:flex; align-items:center; gap:14px; margin-bottom:14px; }
      .brand img, .brand .emoji { width:54px; height:54px; border-radius:16px; object-fit:cover; }
      .brand .emoji { display:flex; align-items:center; justify-content:center; font-size:30px;
                     background:linear-gradient(135deg,#6366f1,#22d3ee); color:white; }
      .brand .title { font-size:1.55rem; font-weight:800; line-height:1.1; }
      .brand .subtitle { color:#64748b; font-size:.88rem; }
      .mode-card { border:1px solid rgba(148,163,184,.25); border-radius:18px; padding:16px;
                   background:rgba(148,163,184,.07); height:100%; }
      .mode-title { font-weight:750; font-size:1.05rem; margin-bottom:4px; }
      .mode-blurb { color:#94a3b8; font-size:.86rem; }
      .file-chip { display:inline-block; background:rgba(99,102,241,.15); color:#a5b4fc;
                   border:1px solid rgba(99,102,241,.35); padding:3px 10px; border-radius:999px;
                   font-size:.76rem; margin:3px 4px 3px 0; }
      .chat-user { background:linear-gradient(135deg,rgba(99,102,241,.18),rgba(34,211,238,.12));
                   border:1px solid rgba(99,102,241,.28); border-radius:16px; padding:14px 16px; }
      .chat-bot { background:rgba(148,163,184,.08); border:1px solid rgba(148,163,184,.20);
                border-radius:16px; padding:14px 16px; }
      .small-muted { color:#94a3b8; font-size:.82rem; }
      div[data-testid="stFileUploaderDropzone"] { border-radius:16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(LOGO_HTML, unsafe_allow_html=True)

# ----------------------------- MODEL -----------------------------

def get_base_model_id():
    value = os.environ.get("BASE_MODEL")
    if not value:
        try:
            value = st.secrets.get("BASE_MODEL")
        except Exception:
            value = None
    return value


BASE_MODEL = get_base_model_id()

if not BASE_MODEL:
    st.error("BASE_MODEL is not set. Add it as an environment variable or Streamlit secret.")
    st.stop()

if not torch.cuda.is_available():
    st.error("No CUDA GPU detected. This app requires an NVIDIA GPU, such as a Kaggle/Colab T4.")
    st.stop()


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp.replace(dest)


def ensure_adapters():
    base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

    for adapter_name, files in ASSETS.items():
        for local_name, asset_name in files.items():
            dest = ADAPTER_DIR / adapter_name / local_name
            if not dest.exists():
                download_file(f"{base_url}/{asset_name}", dest)


@st.cache_resource(show_spinner="Loading model and adapters...")
def load_model():
    ensure_adapters()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER_DIR / "english_adapter"),
        adapter_name="english_adapter",
    )
    model.load_adapter(str(ADAPTER_DIR / "math_adapter"), adapter_name="math_adapter")
    model.load_adapter(str(ADAPTER_DIR / "general_adapter"), adapter_name="general_adapter")
    model.eval()

    return tokenizer, model, threading.Lock(), threading.Event()


try:
    tokenizer, model, generation_lock, stop_event = load_model()
except requests.HTTPError as e:
    st.error(f"Could not download adapter files from GitHub Releases: {e}")
    st.stop()
except Exception as e:
    st.error(f"Model loading failed: `{e}`")
    st.stop()

# ----------------------------- FILE PARSING -----------------------------

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".jsx", ".tsx", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh",
    ".bash", ".zsh", ".sql", ".html", ".htm", ".css", ".scss", ".xml", ".json",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".log", ".csv", ".tsv",
    ".tex", ".latex", ".r", ".m", ".ipynb", ".env", ".gitignore", ".dockerfile",
}

ZIP_LIKE_EXTENSIONS = {
    ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".epub", ".zip",
    ".pages", ".numbers", ".key",
}

LIBREOFFICE_EXTENSIONS = {
    ".doc", ".xls", ".ppt", ".rtf", ".ps", ".wpd", ".pub", ".xps",
    ".mht", ".mhtml", ".indd", ".fb2",
}


def clean_xml_text(raw: str) -> str:
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = raw.replace("&nbsp;", " ").replace("&amp;", "&")
    raw = raw.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return re.sub(r"\s+", " ", raw).strip()


def extract_text_from_zip_bytes(data: bytes, original_name: str) -> str:
    chunks = []

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            useful = []

            for name in z.namelist():
                lower = name.lower()

                if lower.endswith("__MACOSX") or name.startswith("__MACOSX/"):
                    continue

                if lower.endswith((".xml", ".txt", ".html", ".htm", ".xhtml", ".md", ".json", ".csv")):
                    useful.append(name)

            priority = [
                "word/document.xml",
                "xl/sharedstrings.xml",
                "xl/worksheets/",
                "ppt/slides/",
                "content.xml",
                "content.opf",
                "index_split",
            ]

            useful.sort(
                key=lambda n: (
                    0 if any(p in n.lower() for p in priority) else 1,
                    n.lower(),
                )
            )

            for name in useful[:80]:
                try:
                    with z.open(name) as f:
                        raw = f.read(2_000_000).decode("utf-8", errors="replace")

                    if name.lower().endswith(".json"):
                        try:
                            parsed = json.loads(raw)
                            text = json.dumps(parsed, indent=2, ensure_ascii=False)
                        except Exception:
                            text = raw
                    else:
                        text = clean_xml_text(raw)

                    if text:
                        chunks.append(f"--- {name} ---\n{text[:6000]}")

                except Exception:
                    continue

    except Exception as e:
        return f"[Could not read archive structure in {original_name}: {e}]"

    if not chunks:
        return f"[No readable text found inside {original_name}.]"

    return "\n\n".join(chunks)


def extract_pdf_with_pypdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"--- Page {i + 1} ---\n{text.strip()}")

        return "\n\n".join(pages)

    except Exception as e:
        return f"[PDF extraction error: {e}]"


def convert_with_libreoffice(data: bytes, original_name: str) -> str:
    suffix = Path(original_name).suffix or ".bin"

    try:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / f"input{suffix}"
            src.write_bytes(data)

            result = subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "txt:Text",
                    "--outdir",
                    td,
                    str(src),
                ],
                capture_output=True,
                timeout=90,
            )

            out = Path(td) / "input.txt"

            if out.exists():
                return out.read_text(encoding="utf-8", errors="replace")

            return f"[LibreOffice conversion failed for {original_name}.]{result.stderr.decode(errors='replace')[:500]}"

    except FileNotFoundError:
        return ""
    except Exception as e:
        return f"[LibreOffice conversion error for {original_name}: {e}]"


def extract_docx(data: bytes) -> str:
    try:
        from docx import Document

        doc = Document(io.BytesIO(data))
        parts = []

        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)

        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        return "\n".join(parts)

    except Exception as e:
        return f"[DOCX extraction error: {e}]"


def extract_xlsx(data: bytes) -> str:
    try:
        from openpyxl import load_workbook

        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        parts = []

        for sheet in wb.worksheets:
            parts.append(f"--- Sheet: {sheet.title} ---")

            for row in sheet.iter_rows(values_only=True):
                values = ["" if v is None else str(v) for v in row]

                if any(v.strip() for v in values):
                    parts.append(" | ".join(values))

        return "\n".join(parts)

    except Exception as e:
        return f"[XLSX extraction error: {e}]"


def extract_pptx(data: bytes) -> str:
    try:
        from pptx import Presentation

        prs = Presentation(io.BytesIO(data))
        parts = []

        for i, slide in enumerate(prs.slides, start=1):
            parts.append(f"--- Slide {i} ---")

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    parts.append(shape.text.strip())

        return "\n".join(parts)

    except Exception as e:
        return f"[PPTX extraction error: {e}]"


def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name
    ext = Path(name).suffix.lower()
    data = uploaded_file.getvalue()

    if ext in TEXT_EXTENSIONS:
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return data.decode("latin-1", errors="replace")

    if ext == ".pdf":
        text = extract_pdf_with_pypdf(data)

        if text and not text.startswith("[PDF extraction error"):
            return text

        fallback = extract_text_from_zip_bytes(data, name)

        if fallback and not fallback.startswith("[Could not read"):
            return fallback

        return text or f"[Unable to extract text from PDF: {name}]"

    if ext == ".docx":
        return extract_docx(data) or extract_text_from_zip_bytes(data, name)

    if ext == ".xlsx":
        return extract_xlsx(data) or extract_text_from_zip_bytes(data, name)

    if ext == ".pptx":
        return extract_pptx(data) or extract_text_from_zip_bytes(data, name)

    if ext in ZIP_LIKE_EXTENSIONS:
        return extract_text_from_zip_bytes(data, name)

    if ext in LIBREOFFICE_EXTENSIONS:
        converted = convert_with_libreoffice(data, name)

        if converted:
            return converted

        return (
            f"[Legacy format {ext} could not be converted. Install LibreOffice for support: "
            "`sudo apt install libreoffice`.]"
        )

    try:
        decoded = data.decode("utf-8")

        if decoded.isprintable() or "\n" in decoded:
            return decoded

    except Exception:
        pass

    return f"[Binary or unsupported file: {name} ({ext})]"


def make_file_context(files) -> tuple[str, list[str]]:
    contexts = []
    chips = []

    for uploaded in files[:8]:
        raw_text = extract_text_from_upload(uploaded)
        text = raw_text[:MAX_FILE_CHARS]
        contexts.append(
            f"### Uploaded file: {uploaded.name}\n"
            f"Type: {uploaded.type or 'unknown'}\n\n"
            f"{text}"
        )
        chips.append(uploaded.name)

    joined = "\n\n".join(contexts)

    if len(joined) > MAX_HISTORY_CHARS:
        joined = joined[:MAX_HISTORY_CHARS] + "\n\n[File context truncated.]"

    return joined, chips

# ----------------------------- MATH -----------------------------

SYMPY_NAMES = [
    "diff", "integrate", "limit", "solve", "simplify", "expand", "factor",
    "cancel", "apart", "trigsimp", "series", "summation", "sin", "cos", "tan",
    "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log", "sqrt",
    "Abs", "factorial", "binomial", "gcd", "lcm", "Rational", "oo", "pi",
    "E", "Integer", "Float", "Symbol", "Function",
]

SAFE_SYMPY = {n: getattr(sp, n) for n in SYMPY_NAMES if hasattr(sp, n)}
SAFE_SYMPY.update({letter: sp.Symbol(letter) for letter in "xyztnabc"})

SYMPY_TRANSFORMS = standard_transformations + (convert_xor,)
ALLOWED_MATH = re.compile(r"[A-Za-z0-9_+\-*/^().,\s]*")


def run_sympy(expression: str):
    expression = expression.strip()

    if not expression:
        return None, "Enter an expression first."

    if len(expression) > 250 or "__" in expression:
        return None, "Expression is too long or contains forbidden characters."

    if not ALLOWED_MATH.fullmatch(expression):
        return None, "Use only numbers, letters, + - * / ^ ( ) , and supported SymPy functions."

    try:
        result = sp.simplify(
            parse_expr(
                expression,
                local_dict=SAFE_SYMPY,
                global_dict={"__builtins__": {}},
                transformations=SYMPY_TRANSFORMS,
            )
        )
        return result, None

    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:180]}"

# ----------------------------- GENERATION -----------------------------

class GenerationStopped(Exception):
    pass


class StoppableStreamer(TextIteratorStreamer):
    def __init__(self, tok, event, **kwargs):
        super().__init__(tok, **kwargs)
        self.stop_event = event

    def put(self, value):
        if self.stop_event.is_set():
            raise GenerationStopped()

        super().put(value)


def stream_reply(messages, adapter, temperature, max_new_tokens, error_box):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    streamer = StoppableStreamer(
        tokenizer,
        stop_event,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=180,
    )

    def worker():
        with generation_lock:
            stop_event.clear()

            try:
                model.set_adapter(adapter)

                with torch.no_grad():
                    model.generate(
                        **inputs,
                        streamer=streamer,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.8,
                        do_sample=True,
                        repetition_penalty=1.05,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    )

            except GenerationStopped:
                streamer.on_finalized_text("", stream_end=True)

            except Exception as e:
                error_box["error"] = e
                streamer.on_finalized_text("", stream_end=True)

    threading.Thread(target=worker, daemon=True).start()

    for piece in streamer:
        yield piece

# ----------------------------- HELPERS -----------------------------

def safe_markdown(text: str) -> str:
    if text.count("```") % 2:
        text += "\n```"

    if text.count("$$") \% 2: text += "$$"

    return text


def render_stream(slot, text: str, final: bool = False):
    if final:
        slot.markdown(safe_markdown(text))
        return

    cut = max(text.rfind(" "), text.rfind("\n")) + 1
    held = text[:cut] if cut else text

    slot.markdown(safe_markdown(held) + " ▌")


def trim_history(messages, limit=MAX_HISTORY_CHARS):
    total = 0
    kept = []

    for msg in reversed(messages):
        cost = len(msg.get("content", ""))

        if total + cost > limit and kept:
            break

        kept.append(msg)
        total += cost

    return list(reversed(kept))


def new_chat():
    st.session_state.messages = []
    st.session_state.current_id = None
    st.session_state.title = None
    st.session_state.uploaded_files = []
    st.session_state.file_context = ""


def save_chat():
    if not st.session_state.messages:
        return

    if not st.session_state.current_id:
        st.session_state.current_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    first_user = next(
        (m["content"] for m in st.session_state.messages if m["role"] == "user"),
        "Chat",
    )

    title = re.sub(r"\s+", " ", first_user)[:60] or "Chat"

    record = {
        "id": st.session_state.current_id,
        "title": title,
        "mode": st.session_state.mode,
        "updated": datetime.now().isoformat(timespec="seconds"),
        "messages": st.session_state.messages,
    }

    path = CHAT_DIR / f"{st.session_state.current_id}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chat(path: Path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        st.session_state.messages = record.get("messages", [])
        st.session_state.current_id = record.get("id")
        st.session_state.title = record.get("title")
        st.session_state.mode = record.get("mode", "General")
        st.session_state.uploaded_files = []
        st.session_state.file_context = ""
    except Exception as e:
        st.error(f"Could not open chat: {e}")

# ----------------------------- SESSION -----------------------------

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("current_id", None)
ss.setdefault("title", None)
ss.setdefault("mode", "General")
ss.setdefault("uploaded_files", [])
ss.setdefault("file_context", "")
ss.setdefault("expr", "diff(x**3 + 5*x, x)")

# ----------------------------- SIDEBAR -----------------------------

with st.sidebar:
    st.subheader("Workspace")

    mode = st.radio(
        "Assistant mode",
        list(MODES.keys()),
        format_func=lambda m: f"{MODES[m]['icon']} {m}",
        key="mode",
    )

    st.caption(MODES[mode]["blurb"])

    st.divider()
    st.subheader("Generation")

    temperature = st.slider("Creativity", 0.0, 1.0, 0.35, 0.05)
    max_new_tokens = st.slider("Maximum response length", 256, 4096, 1600, 128)

    st.divider()
    st.subheader("Chats")

    if st.button("➕ New chat", use_container_width=True):
        new_chat()
        st.rerun()

    chat_files = sorted(CHAT_DIR.glob("*.json"), reverse=True)

    if chat_files:
        selected = st.selectbox(
            "Open saved chat",
            options=[""] + [p.stem for p in chat_files[:30]],
            format_func=lambda x: "Select a chat..." if not x else x,
        )

        if selected and selected != st.session_state.current_id:
            load_chat(CHAT_DIR / f"{selected}.json")
            st.rerun()
    else:
        st.caption("No saved chats yet.")

    st.divider()
    st.caption(f"Base model: `{BASE_MODEL}`")
    st.caption(f"GPU: `{torch.cuda.get_device_name(0)}`")

# ----------------------------- MAIN UI -----------------------------

cols = st.columns(3)

for i, (name, cfg) in enumerate(MODES.items()):
    with cols[i]:
        selected = name == mode
        border = "2px solid #6366f1" if selected else "1px solid rgba(148,163,184,.25)"
        st.markdown(
            f"""
            <div class="mode-card" style="border:{border}">
              <div class="mode-title">{cfg['icon']} {name}</div>
              <div class="mode-blurb">{cfg['blurb']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

left, right = st.columns([2.35, 1])

with right:
    st.subheader("📎 Attach files")

    uploaded_files = st.file_uploader(
        "Upload documents, spreadsheets, presentations, code, archives, or text files",
        type=[
            "pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt", "txt", "csv",
            "tsv", "md", "markdown", "html", "htm", "json", "xml", "yaml", "yml",
            "tex", "rtf", "odt", "ods", "odp", "epub", "fb2", "zip", "py", "ipynb",
            "js", "ts", "java", "c", "cpp", "h", "cs", "go", "rs", "rb", "php",
            "sh", "sql", "css", "log", "pages", "numbers", "key", "ps", "wpd",
            "pub", "xps", "mht", "mhtml",
        ],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        file_context, chips = make_file_context(uploaded_files)
        st.session_state.file_context = file_context

        st.markdown(
            "".join(f'<span class="file-chip">{c}</span>' for c in chips),
            unsafe_allow_html=True,
        )
    else:
        st.session_state.uploaded_files = []
        st.session_state.file_context = ""

    st.divider()
    st.subheader("🧮 Math checker")

    expression = st.text_area(
        "SymPy expression",
        value=ss.expr,
        height=90,
        help="Examples: diff(x**3 + 5*x, x), integrate(x**2, x), solve(x**2 - 4, x)",
    )

    ss.expr = expression

    if st.button("Check with SymPy", use_container_width=True):
        result, error = run_sympy(expression)

        if error:
            st.error(error)
        else:
            st.success("SymPy result")
            st.code(str(result), language="text")

with left:
    if not ss.messages:
        st.markdown("### Start a conversation")
        st.caption("Choose a mode, attach files if needed, then ask a question.")

        for example in MODES[mode]["examples"]:
            if st.button(example, use_container_width=True):
                ss.pending_prompt = example
                st.rerun()

    for message in ss.messages:
        role = message["role"]
        avatar = "🧑‍💻" if role == "user" else "✨"

        with st.chat_message(role, avatar=avatar):
            if message.get("files"):
                st.markdown(
                    "".join(
                        f'<span class="file-chip">{f}</span>'
                        for f in message["files"]
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown(message["content"])

    prompt = st.chat_input(
        MODES[mode]["placeholder"],
        key="main_chat_input",
    )

    if not prompt and "pending_prompt" in ss:
        prompt = ss.pending_prompt
        del ss.pending_prompt

    if prompt:
        user_message = {"role": "user", "content": prompt}
        file_names = [f.name for f in ss.uploaded_files]

        if file_names:
            user_message["files"] = file_names

        ss.messages.append(user_message)

        if ss.file_context:
            user_content = (
                f"{prompt}\n\n"
                "Attached file content follows. Use it when relevant and cite section/page names if available.\n\n"
                f"{ss.file_context}"
            )
        else:
            user_content = prompt

        history = trim_history(
            [
                {"role": m["role"], "content": m["content"]}
                for m in ss.messages[:-1]
            ]
        )

        messages = history + [{"role": "user", "content": user_content}]

        with st.chat_message("user", avatar="🧑‍💻"):
            if file_names:
                st.markdown(
                    "".join(
                        f'<span class="file-chip">{f}</span>' for f in file_names
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown(prompt)

        with st.chat_message("assistant", avatar="✨"):
            slot = st.empty()
            error_box = {}
            response = ""

            try:
                for piece in stream_reply(
                    messages,
                    MODES[mode]["adapter"],
                    temperature,
                    max_new_tokens,
                    error_box,
                ):
                    response += piece
                    render_stream(slot, response)

                render_stream(slot, response, final=True)

            except Exception as e:
                response += f"\n\n**Generation error:** `{e}`"
                slot.markdown(safe_markdown(response))

            if error_box.get("error"):
                response += f"\n\n**Generation error:** `{error_box['error']}`"
                slot.markdown(safe_markdown(response))

            if not response.strip():
                response = "I could not generate a response. Please try again."
                slot.markdown(response)

        ss.messages.append({"role": "assistant", "content": response})
        save_chat()
        st.rerun()

# ----------------------------- QUICK ACTIONS -----------------------------

if ss.messages:
    st.divider()

    action_cols = st.columns(4)

    with action_cols[0]:
        if st.button("🧹 Clear chat", use_container_width=True):
            new_chat()
            st.rerun()

    with action_cols[1]:
        if st.button("💾 Save chat", use_container_width=True):
            save_chat()
            st.success("Chat saved.")

    with action_cols[2]:
        if st.button("⬇️ Export Markdown", use_container_width=True):
            md_lines = [f"# Tutor Studio Chat\n"]

            for m in ss.messages:
                label = "You" if m["role"] == "user" else "Assistant"
                md_lines.append(f"## {label}\n\n{m['content']}\n")

            st.download_button(
                "Download chat",
                data="\n".join(md_lines),
                file_name="tutor-studio-chat.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with action_cols[3]:
        if st.button("⛔ Stop generation", use_container_width=True):
            stop_event.set()
            st.info("Stop requested. The current generation will halt at the next token.")
