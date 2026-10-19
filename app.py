import os
import re
import json
import time
import threading
from datetime import datetime, date, timedelta
from pathlib import Path

import requests
import torch
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TextIteratorStreamer)
from peft import PeftModel

# ---------------- EDIT THESE ----------------
GITHUB_REPO = "rickeysharma942-stack/english-math-tutor"   # repo that has the release
RELEASE_TAG = os.environ.get("ADAPTER_RELEASE_TAG", "model-v2")  # override with ADAPTER_RELEASE_TAG

# File names exactly as uploaded to the GitHub Release
ASSETS = {
    name: {
        "adapter_config.json": f"{short}_adapter_config.json",
        "adapter_model.safetensors": f"{short}_adapter_model.safetensors",
    }
    for name, short in [("english_adapter", "english"), ("math_adapter", "math"), ("general_adapter", "general")]
}
# --------------------------------------------

ADAPTER_DIR = Path("adapters")
# Chats are saved here. Point CHAT_DIR at a Google Drive folder to keep them after the Colab session ends.
CHAT_DIR = Path(os.environ.get("CHAT_DIR", "chats"))
CHAT_DIR.mkdir(parents=True, exist_ok=True)

MODES = {
    "General": {
        "adapter": "general_adapter", "icon": "💬",
        "blurb": "Ask anything. Long, detailed answers.",
        "placeholder": "Ask me anything...",
        "examples": ["What are some good habits for daily productivity?",
                     "Explain how the internet works to a beginner.",
                     "Give me a 7-day plan to learn public speaking."],
    },
    "English": {
        "adapter": "english_adapter", "icon": "✍️",
        "blurb": "Grammar fixes with a detailed explanation of every rule.",
        "placeholder": "Paste a sentence to correct, or ask about a grammar rule...",
        "examples": ["Correct the grammar and explain in detail: He don't know nothing about it.",
                     "Correct the grammar and explain in detail: Me and her goes to school yesterday.",
                     "Explain the difference between 'affect' and 'effect'."],
    },
    "Math": {
        "adapter": "math_adapter", "icon": "🧮",
        "blurb": "Step-by-step solutions. SymPy checks the algebra for you.",
        "placeholder": "Ask a math question...",
        "examples": [("Explain step by step the derivative of: diff(x**3 + 5*x, x)", "diff(x**3 + 5*x, x)"),
                     ("Explain step by step the integral of: integrate(x**2 + 3*x, x)", "integrate(x**2 + 3*x, x)"),
                     ("Solve step by step: a train travels 180 km in 2.5 hours. What is its average speed?", "")],
    },
}

st.set_page_config(page_title="English + Math Tutor", page_icon="🧠", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 900px; padding: 1.5rem 1.25rem 6rem;}
[data-testid="stAppViewContainer"] {scroll-behavior: smooth;}
.hero {padding: 1.15rem 1.35rem; border-radius: 20px; margin-bottom: 1rem;
       background: linear-gradient(135deg, rgba(99,102,241,.19), rgba(236,72,153,.10) 58%, rgba(14,165,233,.10));
       border: 1px solid rgba(128,128,128,.28); box-shadow: 0 8px 26px rgba(0,0,0,.045);}
.hero h1 {margin: 0; font-size: clamp(1.45rem, 3vw, 1.9rem); line-height: 1.2; letter-spacing: -.025em;}
.hero p {margin: .45rem 0 0; opacity: .82; font-size: .94rem; line-height: 1.5;}
.pill {display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .76rem;
       border: 1px solid rgba(128,128,128,.38); background: rgba(128,128,128,.07); margin-bottom: .3rem;}
.mode-card {height: 100%; min-height: 116px; padding: .9rem .85rem; border-radius: 15px;
       border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.035);
       transition: border-color .18s ease, background .18s ease, transform .18s ease;}
.mode-card-active {border-color: rgba(99,102,241,.85);
       background: linear-gradient(145deg, rgba(99,102,241,.13), rgba(236,72,153,.06));
       box-shadow: 0 5px 18px rgba(99,102,241,.08);}
.mode-card-title {font-weight: 700; font-size: 1rem; line-height: 1.35; margin-bottom: .45rem;}
.mode-card-description {font-size: .82rem; line-height: 1.45; opacity: .82;}

div[data-testid="stChatMessage"] {border: 1px solid rgba(128,128,128,.13); border-radius: 16px;
       padding: .85rem 1rem; margin: .65rem 0; transition: border-color .18s ease;}
div.stButton > button, div[data-testid="stDownloadButton"] button {border-radius: 12px; min-height: 2.55rem;
       transition: transform .15s ease, border-color .15s ease, background .15s ease;}
div.stButton > button:hover {transform: translateY(-1px);}
div[data-testid="stChatInput"] {border-radius: 18px;}
div[data-testid="stChatInput"] textarea {line-height: 1.5;}
div[data-testid="stExpander"] {border-radius: 14px; overflow: hidden;}
section[data-testid="stSidebar"] div.stButton > button {justify-content: flex-start; text-align: left; padding: .35rem .7rem;}
section[data-testid="stSidebar"] div.stButton > button p {white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.hist-label {font-size: .74rem; letter-spacing: .06em; text-transform: uppercase; opacity: .8; margin: .8rem 0 .2rem .2rem;}
@media (max-width: 640px) {
  .block-container {padding: 1rem .75rem 5.5rem;}
  .hero {padding: 1rem; border-radius: 16px;}
  div[data-testid="stChatMessage"] {padding: .65rem .75rem; border-radius: 13px;}
}
/* accessibility: strong visible keyboard focus, screen-reader-only text, reduced motion */
button:focus-visible, textarea:focus-visible, input:focus-visible, [role="radio"]:focus-visible, [role="tab"]:focus-visible
  {outline: 3px solid #6366f1 !important; outline-offset: 2px !important;}
.sr-only {position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap;}
@media (prefers-reduced-motion: reduce) {* {animation: none !important; transition: none !important;}}
.skeleton {height: 3.4rem; border-radius: 10px;
  background: linear-gradient(90deg, rgba(128,128,128,.18) 25%, rgba(128,128,128,.34) 37%, rgba(128,128,128,.18) 63%);
  background-size: 400% 100%; animation: shimmer 1.4s ease infinite;}
@keyframes shimmer {0% {background-position: 100% 0} 100% {background-position: 0 0}}
</style>
""", unsafe_allow_html=True)


# ---------------- model loading ----------------
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


@st.cache_resource(show_spinner="Loading model... (first run downloads files and takes a few minutes)")
def load_model():
    ensure_adapters()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16
    )
    mdl = PeftModel.from_pretrained(base, str(ADAPTER_DIR / "english_adapter"), adapter_name="english_adapter")
    mdl.load_adapter(str(ADAPTER_DIR / "math_adapter"), adapter_name="math_adapter")
    mdl.load_adapter(str(ADAPTER_DIR / "general_adapter"), adapter_name="general_adapter")
    mdl.eval()
    return tok, mdl, threading.Lock(), threading.Event()   # lock: one generation at a time; event: the Stop button


try:
    tokenizer, model, gen_lock, stop_event = load_model()
except requests.HTTPError as e:
    st.error(f"Could not download adapter files from the GitHub Release: {e}. "
             "Check GITHUB_REPO, ADAPTER_RELEASE_TAG and the exact release asset names; the repository/release must be public.")
    st.stop()
except Exception as e:
    st.error(f"Model setup failed: {type(e).__name__}: {e}")
    st.info("Check BASE_MODEL, GPU availability, package versions, and that each adapter matches the base model.")
    st.stop()


# ---------------- SymPy (restricted parser) ----------------
_NAMES = ["diff", "integrate", "limit", "solve", "simplify", "expand", "factor", "cancel", "apart",
          "trigsimp", "series", "summation", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh",
          "tanh", "exp", "log", "sqrt", "Abs", "factorial", "binomial", "gcd", "lcm", "Rational", "oo",
          "pi", "E", "Integer", "Float", "Symbol", "Function"]
SAFE = {n: getattr(sp, n) for n in _NAMES if hasattr(sp, n)}
SAFE.update({s: sp.Symbol(s) for s in "xyztnabc"})
_TRANSFORMS = standard_transformations + (convert_xor,)
_ALLOWED = re.compile(r"[A-Za-z0-9_+\-*/^().,\s]*")


def run_sympy(expr: str):
    """Return (result, error). Only a small whitelist of SymPy names is available to the expression."""
    if len(expr) > 200 or "__" in expr or not _ALLOWED.fullmatch(expr):
        return None, "Use only numbers, letters, + - * / ^ ( ) , and SymPy functions like diff(...)."
    try:
        return sp.simplify(parse_expr(expr, local_dict=SAFE, global_dict={"__builtins__": {}},
                                      transformations=_TRANSFORMS)), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


# ---------------- generation (streamed, stoppable) ----------------
class _Stopped(Exception):
    pass


class StoppableStreamer(TextIteratorStreamer):
    """Normal token streamer that aborts generation as soon as the Stop button sets the event."""
    def __init__(self, tok, ev, **kw):
        super().__init__(tok, **kw)
        self._ev = ev

    def put(self, value):
        if self._ev.is_set():
            raise _Stopped()
        super().put(value)


def stream_reply(messages, adapter, temperature, max_new_tokens, err, raw_prompt=None):
    if raw_prompt is not None:
        text = raw_prompt
    else:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    streamer = StoppableStreamer(tokenizer, stop_event, skip_prompt=True, skip_special_tokens=True, timeout=180)

    stop_event.clear()
    def work():
        with gen_lock:
            try:
                model.set_adapter(adapter)
                with torch.no_grad():
                    model.generate(
                        **inputs, streamer=streamer, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_p=0.8, do_sample=True, repetition_penalty=1.05,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    )
            except _Stopped:
                streamer.on_finalized_text("", stream_end=True)
            except Exception as e:                       # make sure the UI never hangs
                err["e"] = e
                streamer.on_finalized_text("", stream_end=True)

    threading.Thread(target=work, daemon=True).start()
    for piece in streamer:
        yield piece


# ---------------- text helpers ----------------
def safe_markdown(t: str) -> str:
    """Close unfinished code fences / math blocks so half-streamed markdown never breaks the layout."""
    if t.count("```") % 2:
        t += "\n```"
    if t.count("$$") % 2:
        t += "$$"
    return t


def render_stream(slot, text: str, final: bool = False):
    if final:
        slot.markdown(text)
        return
    held = text[: max(text.rfind(" "), text.rfind("\n")) + 1] or text    # hold back the unfinished last word
    slot.markdown(safe_markdown(held) + " ▌")


def normalize_math(s: str) -> str:
    for a, b in {"**": "^", "²": "^2", "³": "^3", "⁴": "^4", "⁵": "^5",
                 "\\cdot": "", "\\left": "", "\\right": ""}.items():
        s = s.replace(a, b)
    return re.sub(r"[\s*${}\\]", "", s)


def popover(label):
    """st.popover needs Streamlit >= 1.32; fall back to an expander on older versions."""
    return st.popover(label) if hasattr(st, "popover") else st.expander(label)


def columns(spec, **kw):
    """Vertically centred columns (Streamlit >= 1.36); plain columns on older versions."""
    try:
        return st.columns(spec, vertical_alignment="center", **kw)
    except TypeError:
        return st.columns(spec, **kw)


def count_tokens(t: str) -> int:
    if not t:
        return 0
    return len(tokenizer(t, add_special_tokens=False)["input_ids"])


# ---------------- state ----------------
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("current_id", None)
ss.setdefault("title", None)
ss.setdefault("mode", "General")
ss.setdefault("expr", "diff(x**3 + 5*x, x)")

REASONS = ["Incorrect", "Too short", "Too long", "Confusing", "Wrong format", "Other"]
FOLLOWUPS = {
    "General": ["Explain it more simply", "Give me an example", "Summarize in 3 bullet points"],
    "English": ["Give me 3 practice sentences", "Explain it more simply", "Show another example"],
    "Math": ["Give me a similar problem to try", "Explain it more simply", "Quiz me on this"],
}
FILE_DEFAULTS = {"English": "Proofread this text and explain each correction in detail:",
                 "Math": "Solve the problems in this text step by step:",
                 "General": "Summarize and explain this text:"}
MAX_FILE_CHARS = 6000


def train_prompt(user: str) -> str:
    """Exactly the prompt format the adapters were trained on (used to continue a cut-off answer)."""
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ---------------- chat history (saved as JSON files) ----------------
def make_title(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"^Correct the grammar and explain in detail:\s*", "Grammar: ", t)
    t = re.sub(r"^Explain step by step the derivative of:\s*", "Derivative: ", t)
    return t[:42] + ("…" if len(t) > 42 else "")


def save_chat():
    if not ss["messages"]:
        return
    if not ss["current_id"]:
        ss["current_id"] = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if not ss["title"]:
        first = next((m["content"] for m in ss["messages"] if m["role"] == "user"), "New chat")
        ss["title"] = make_title(first)
    data = {"id": ss["current_id"], "title": ss["title"], "updated": time.time(),
            "starred": ss.get("starred", False), "messages": ss["messages"]}
    path = CHAT_DIR / f"{ss['current_id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def list_chats():
    chats = []
    for p in CHAT_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            chats.append({"id": d["id"], "title": d.get("title", "Chat"), "updated": d.get("updated", 0),
                          "starred": d.get("starred", False),
                          "blob": " ".join(m["content"] for m in d["messages"]).lower(),     # full-text search
                          "mode": next((m.get("mode") for m in reversed(d["messages"]) if m.get("mode")), "General")})
        except Exception:
            continue
    return sorted(chats, key=lambda c: c["updated"], reverse=True)


def new_chat():
    ss["current_id"], ss["title"], ss["messages"], ss["starred"] = None, None, [], False


def open_chat(cid):
    try:
        d = json.loads((CHAT_DIR / f"{cid}.json").read_text(encoding="utf-8"))
    except Exception:
        return
    ss["current_id"], ss["title"], ss["messages"] = d["id"], d.get("title"), d["messages"]
    ss["starred"] = d.get("starred", False)
    last = next((m.get("mode") for m in reversed(d["messages"]) if m.get("mode")), None)
    if last in MODES:
        ss["mode"] = last


def delete_chat(cid):
    (CHAT_DIR / f"{cid}.json").unlink(missing_ok=True)
    if ss["current_id"] == cid:
        new_chat()


def delete_all():
    for p in CHAT_DIR.glob("*.json"):
        p.unlink(missing_ok=True)
    new_chat()


# ---------------- actions: stop, retry, feedback, pin ----------------
def stop_generation():
    stop_event.set()


def regenerate():
    msgs = ss["messages"]
    if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
        idx = len(msgs) - 1
        last, user = msgs.pop(), msgs.pop()
        for k in (f"fb_{ss['current_id']}_{idx}", f"why_{ss['current_id']}_{idx}"):
            ss.pop(k, None)                       # forget the old rating widgets for this slot
        if last.get("mode") in MODES:
            ss["mode"] = last["mode"]
        ss["queued"] = user["content"]
        ss["no_hint"] = False


def queue_example(ex):
    if isinstance(ex, tuple):
        ss["queued"], ss["expr"] = ex[0], ex[1]
    else:
        ss["queued"] = ex


def queue_followup(text):
    ss["queued"], ss["force_ctx"], ss["no_hint"] = text, True, True


def log_feedback(i):
    """Append to feedback.jsonl so thumbs up/down can later become training data."""
    m = ss["messages"][i]
    prev = ss["messages"][i - 1]["content"] if i > 0 else ""
    rec = {"ts": time.time(), "chat": ss["current_id"], "mode": m.get("mode"), "prompt": prev,
           "reply": m["content"], "rating": m.get("rating"), "reason": m.get("reason")}
    with open(CHAT_DIR / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def rate(i, key):
    m, v = ss["messages"][i], ss.get(key)
    if v is None:
        m.pop("rating", None), m.pop("reason", None)
    else:
        m["rating"] = v
        if v == 1:
            m.pop("reason", None)
    save_chat()
    log_feedback(i)


def give_reason(i, key):
    ss["messages"][i]["reason"] = ss.get(key)
    save_chat()
    log_feedback(i)


def toggle_pin(i):
    m = ss["messages"][i]
    m["pinned"] = not m.get("pinned", False)
    save_chat()


def toggle_star(cid):
    path = CHAT_DIR / f"{cid}.json"
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        d["starred"] = not d.get("starred", False)
        path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")     # 'updated' stays unchanged
        if ss["current_id"] == cid:
            ss["starred"] = d["starred"]
    except Exception:
        pass


def fork_edit(i, key):
    """Edit an earlier message: copy the chat up to that point into a NEW chat and ask the edited question there."""
    text = (ss.get(key) or "").strip()
    if not text:
        return
    base = [dict(m) for m in ss["messages"][:i]]
    old_title = ss.get("title")
    ss["messages"], ss["current_id"], ss["starred"] = base, None, False
    ss["title"] = f"{old_title} (edited)" if (base and old_title) else None
    ss["queued"], ss["no_hint"] = text, False


def continue_generation():
    ss["cont"] = True


# ---------------- context window (pinned turns are kept, oldest unpinned turns drop first) ----------------
def build_context(messages, budget, max_turns=None):
    turns = [(messages[i], messages[i + 1]) for i in range(len(messages) - 1)
             if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant"
             and not messages[i + 1].get("error")]
    total = len(turns)
    if max_turns:
        turns = turns[-max_turns:]
    cost = lambda t: count_tokens(t[0]["content"] + " " + t[1]["content"]) + 12
    chosen, used = set(), 0
    for idx, t in enumerate(turns):                           # pinned first
        if t[1].get("pinned") and used + cost(t) <= budget:
            chosen.add(idx); used += cost(t)
    for idx in range(len(turns) - 1, -1, -1):                 # then newest -> oldest
        if idx in chosen:
            continue
        c = cost(turns[idx])
        if used + c > budget:
            break
        chosen.add(idx); used += c
    history = []
    for idx in sorted(chosen):
        history += [{"role": "user", "content": turns[idx][0]["content"]},
                    {"role": "assistant", "content": turns[idx][1]["content"]}]
    pinned = sum(1 for idx in chosen if turns[idx][1].get("pinned"))
    return history, {"total": total, "included": len(chosen), "dropped": total - len(chosen),
                     "pinned": pinned, "used": used, "budget": budget}


def day_label(ts):
    d = datetime.fromtimestamp(ts).date()
    if d == date.today():
        return "Today"
    if d == date.today() - timedelta(days=1):
        return "Yesterday"
    return "Earlier"


def read_chat_input(val):
    """chat_input returns a plain string, or an object with .text/.files when file attachments are on."""
    if val is None:
        return None, None
    if isinstance(val, str):
        return val, None
    get = (lambda k: val.get(k)) if hasattr(val, "get") else (lambda k: getattr(val, k, None))
    files = get("files") or []
    return (get("text") or "").strip(), (files[0] if files else None)


# ---------------- finish a generation that was interrupted (Stop button / any other click) ----------------
if ss.get("gen"):
    stop_event.set()
    g = ss.pop("gen")
    if g.get("text", "").strip():
        n = count_tokens(g["text"])
        if g.get("cont") and ss["messages"] and ss["messages"][-1]["role"] == "assistant":
            last = ss["messages"][-1]
            last["content"] += g["text"]
            last.update(stopped=True, truncated=False, meta=f"⏹ Stopped · +{n} tokens")
        else:
            ss["messages"].append({"role": "assistant", "content": g["text"], "mode": g["mode"], "stopped": True,
                                   "meta": f"⏹ Stopped · {n} tokens", "sympy": g.get("sympy"),
                                   "sent": g.get("sent"), "ctx_turns": g.get("ctx_turns")})
        save_chat()
    ss["announce"] = "Generation stopped."
ss["gen"] = None


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("## 🧠 Tutor")
    st.button("✏️  New chat", on_click=new_chat, use_container_width=True)

    chats = list_chats()
    query = ""
    if len(chats) > 6:
        query = st.text_input("Search chats", placeholder="🔎  Search titles and messages", label_visibility="collapsed").strip().lower()
    shown = [c for c in chats if query in c["title"].lower() or query in c["blob"]]
    shown = sorted(shown, key=lambda c: (not c["starred"], -c["updated"]))[:40]

    if not chats:
        st.caption("Your conversations will appear here.")
    elif not shown:
        st.caption("No chats match your search.")
    last_label = None
    for c in shown:
        label = "⭐ Starred" if c["starred"] else day_label(c["updated"])
        if label != last_label:
            st.markdown(f'<div class="hist-label">{label}</div>', unsafe_allow_html=True)
            last_label = label
        row = columns([0.66, 0.17, 0.17], gap="small")
        row[0].button(f"{MODES.get(c['mode'], MODES['General'])['icon']}  {c['title']}", key=f"open_{c['id']}",
                      on_click=open_chat, args=(c["id"],), use_container_width=True,
                      type="primary" if c["id"] == ss["current_id"] else "secondary")
        row[1].button("★" if c["starred"] else "☆", key=f"star_{c['id']}", on_click=toggle_star, args=(c["id"],),
                      help="Star this chat to keep it at the top", use_container_width=True)
        row[2].button("✕", key=f"del_{c['id']}", on_click=delete_chat, args=(c["id"],),
                      help="Delete this chat", use_container_width=True)

    st.divider()
    with st.expander("⚙️  Settings"):
        max_tokens = st.slider("Max response length (tokens)", 200, 1500, 800, step=100)
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7, step=0.1,
                                help="Lower = more precise and repeatable. Higher = more varied.")
        remember = st.toggle("Use earlier messages as context", value=False,
                             help="Off = every question is answered on its own (how the adapters were trained). "
                                  "On = earlier turns of this chat are included, newest first, plus any you pinned (📌).")
        ctx_budget = st.slider("Context budget (tokens)", 1000, 6000, 3000, step=500,
                               help="How much earlier conversation may be sent with each question.")
        if ss["messages"]:
            md = "\n\n".join(f"**{'You' if m['role'] == 'user' else 'Tutor'}:**\n{m['content']}" for m in ss["messages"])
            st.download_button("⬇️  Export this chat (.md)", md, file_name="tutor_chat.md", use_container_width=True)
        if chats:
            sure = st.checkbox("I want to delete all chats")
            st.button("🗑️  Delete all chats", on_click=delete_all, disabled=not sure, use_container_width=True)
    st.caption(f"🟢 Model ready · {torch.cuda.get_device_name(0)}")


def select_mode(mode_name):
    """Switch the active tutor mode without losing the current conversation."""
    if mode_name in MODES:
        st.session_state["mode"] = mode_name


# ---------------- main area ----------------
st.markdown(
    '<div class="hero"><h1>🧠 English + Math Tutor</h1>'
    '<p>Qwen3-8B with three LoRA adapters: detailed answers for English, math and general questions.</p></div>',
    unsafe_allow_html=True)
st.markdown(f'<div class="sr-only" role="status" aria-live="polite">{ss.pop("announce", "")}</div>',
            unsafe_allow_html=True)

_ver = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
if _ver < (1, 40):
    st.warning(f"Streamlit {st.__version__} is old, so some features are simplified. "
               "Run `pip install -U streamlit` and restart the app for the full interface.")

# Clickable mode cards: the selected mode controls the adapter, prompt, examples,
# and (for Math) the optional SymPy checker below.
mode = ss["mode"] if ss.get("mode") in MODES else "General"
mode_descriptions = {
    "General": "Ask questions, summarize documents, brainstorm, or analyze files.",
    "English": "Grammar correction, writing improvement, and detailed explanations.",
    "Math": "Step-by-step math help with an optional SymPy checker.",
}
mode_cols = st.columns(3, gap="small")
for mode_col, mode_name in zip(mode_cols, MODES):
    selected = mode == mode_name
    with mode_col:
        st.markdown(
            f'<div class="mode-card {"mode-card-active" if selected else ""}">'
            f'<div class="mode-card-title">{MODES[mode_name]["icon"]} {mode_name}'
            f'{" · Active" if selected else ""}</div>'
            f'<div class="mode-card-description">{mode_descriptions[mode_name]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.button(
            "Selected ✓" if selected else f"Use {mode_name}",
            key=f"select_mode_{mode_name}",
            on_click=select_mode,
            args=(mode_name,),
            use_container_width=True,
            type="primary" if selected else "secondary",
        )
mode = ss["mode"] if ss.get("mode") in MODES else "General"
cfg = MODES[mode]
st.caption(f'{cfg["icon"]} **{mode} mode** · {cfg["blurb"]}')

expr = ""
if mode == "Math":
    with st.expander("🧮  SymPy check (optional)", expanded=not ss["messages"]):
        expr = st.text_input("Expression", key="expr",
                             help="Examples: diff(x**3 + 5*x, x), integrate(x**2, x), factor(x**2 - 1)")
        if expr.strip():
            res, perr = run_sympy(expr)
            if perr:
                st.warning(perr)
            else:
                st.latex(sp.latex(res))
                st.caption("The model receives this result as a hint, and the answer is checked against it.")

try:
    raw = st.chat_input(cfg["placeholder"], accept_file=True, file_type=["txt", "md"])
except TypeError:                                   # older Streamlit without file attachments
    raw = st.chat_input(cfg["placeholder"])
prompt, attachment = read_chat_input(raw)
if prompt is None and attachment is None:
    prompt = ss.pop("queued", None)
cont_run = bool(ss.pop("cont", False)) and not prompt and attachment is None and len(ss["messages"]) >= 2 \
    and ss["messages"][-1]["role"] == "assistant"
busy = bool(prompt) or attachment is not None or cont_run

if not ss["messages"] and not busy:
    st.markdown("##### Try one of these, or type your own below")
    for i, ex in enumerate(cfg["examples"]):
        label = ex[0] if isinstance(ex, tuple) else ex
        st.button(label, key=f"ex_{mode}_{i}", on_click=queue_example, args=(ex,), use_container_width=True)
    st.caption("📎 You can also attach a .txt or .md file (up to ~6,000 characters are read).")


def stream_into(slot, prefix, gen_msgs, adapter, raw_prompt=None):
    """Stream a reply into `slot`. Returns (text, error, time_to_first_token, was_stopped, seconds)."""
    err, buf, t0, first, last = {}, "", time.time(), None, 0.0
    for piece in stream_reply(gen_msgs, adapter, temperature, max_tokens, err, raw_prompt):
        if first is None:
            first = time.time() - t0
        buf += piece
        ss["gen"]["text"] = buf
        if time.time() - last > 0.05:                  # throttle redraws; avoids flicker and layout jumps
            render_stream(slot, prefix + buf)
            last = time.time()
    was_stopped = stop_event.is_set()
    render_stream(slot, prefix + buf, final=True)
    return buf, err, first, was_stopped, max(time.time() - t0, 1e-3)


def render_actions(i, m, is_last):
    cid = ss["current_id"]
    if m.get("error"):
        if is_last:
            st.button("🔄  Retry", key=f"retry_{cid}_{i}", on_click=regenerate)
        return
    cols = columns([1.0, 1.1, 1.5, 1.2])
    if hasattr(st, "feedback"):
        fk = f"fb_{cid}_{i}"
        if fk not in ss and m.get("rating") is not None:
            ss[fk] = m["rating"]
        with cols[0]:
            st.feedback("thumbs", key=fk, on_change=rate, args=(i, fk))
    pinned = m.get("pinned", False)
    cols[1].button("📌 Pinned" if pinned else "📌 Pin", key=f"pin_{cid}_{i}", on_click=toggle_pin, args=(i,),
                   type="primary" if pinned else "secondary",
                   help="Pinned answers are always kept in the context window.")
    if is_last:
        cols[2].button("🔄 Retry" if m.get("stopped") else "🔄 Regenerate", key=f"regen_{cid}_{i}", on_click=regenerate)
    with cols[3]:
        with popover("📋 Copy"):
            try:
                st.code(m["content"], language=None, wrap_lines=True)    # st.code has a built-in copy button
            except TypeError:
                st.code(m["content"], language=None)
    if m.get("rating") == 0:
        wk = f"why_{cid}_{i}"
        if wk not in ss and m.get("reason"):
            ss[wk] = m["reason"]
        chooser = getattr(st, "pills", None)
        if chooser:
            chooser("What went wrong?", REASONS, key=wk, on_change=give_reason, args=(i, wk))
        else:
            st.selectbox("What went wrong?", REASONS, index=None, key=wk, on_change=give_reason, args=(i, wk))
    if is_last and m.get("truncated"):
        st.caption("✂️ This answer reached the length limit and may be cut off.")
        st.button("▶  Continue", key=f"cont_{cid}_{i}", on_click=continue_generation)
    if is_last and not m.get("stopped"):
        chips = st.columns(3)
        for c, text in zip(chips, FOLLOWUPS[mode]):
            c.button(text, key=f"fu_{cid}_{i}_{text[:10]}", on_click=queue_followup, args=(text,), use_container_width=True)


msgs = ss["messages"]
for i, m in enumerate(msgs):
    if cont_run and i == len(msgs) - 1:
        continue                                        # redrawn live below while it continues
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🧠"):
        if m["role"] == "assistant":
            st.markdown(f'<span class="pill">{m["mode"]}</span>', unsafe_allow_html=True)
        st.markdown(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])
        if m.get("check") == "match":
            st.caption(f"✅ Final result matches SymPy: `{m['sympy']}`")
        elif m.get("check") == "mismatch":
            st.caption(f"⚠️ SymPy computed `{m['sympy']}` but it was not found in this answer. "
                       "It may be written differently, so double-check the steps.")
        if m["role"] == "user" and not busy:
            with popover("✏️ Edit"):
                ek = f"edit_{ss['current_id']}_{i}"
                st.text_area("Edit this message. It starts a new branch, and this chat stays unchanged.",
                             value=m["content"], key=ek, height=110)
                st.button("Send as new branch", key=f"fork_{ss['current_id']}_{i}", on_click=fork_edit, args=(i, ek))
        if m["role"] == "assistant" and not busy:
            render_actions(i, m, is_last=(i == len(msgs) - 1))

if remember and msgs and not busy:
    _, info = build_context(msgs, ctx_budget)
    line = f"🧠 Context for your next question: {info['included']} of {info['total']} earlier turns · " \
           f"{info['used']:,} / {info['budget']:,} tokens"
    if info["pinned"]:
        line += f" · 📌 {info['pinned']} pinned"
    st.progress(min(info["used"] / info["budget"], 1.0), text=line)
    if info["dropped"]:
        st.info(f"{info['dropped']} older turn(s) are left out to fit the context window. "
                "Pin (📌) an answer to always keep it.")

def fmt_meta(n_tok, secs, ttft):
    ft = f" · first token {ttft:.1f}s" if ttft is not None else ""
    return f"{n_tok} tokens · {n_tok / secs:.1f} tok/s{ft} · {secs:.0f}s"


def sympy_check(sympy_str, text):
    return "match" if normalize_math(sympy_str) in normalize_math(text) else "mismatch"


SKELETON = '<div class="skeleton" role="status" aria-label="The tutor is thinking"></div>'

if busy:
    force_ctx, no_hint = ss.pop("force_ctx", False), ss.pop("no_hint", False)

    if cont_run:                                       # ---- continue a cut-off answer
        base, user_m = ss["messages"][-1], ss["messages"][-2]
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(f'<span class="pill">{base["mode"]}</span>', unsafe_allow_html=True)
            slot, stop_slot = st.empty(), st.empty()
            slot.markdown(base["content"] + " ▌")
            stop_slot.button("⏹  Stop generating", key="stop_btn", on_click=stop_generation)
            ss["gen"] = {"mode": base["mode"], "text": "", "cont": True}
            raw = ""
            if base.get("ctx_turns") is not None:        # rebuild the same earlier-turns context the answer used
                hist, _ = build_context(ss["messages"][:-2], ctx_budget, max_turns=base["ctx_turns"] or None)
                raw = "".join(f"<|im_start|>{h['role']}\n{h['content']}<|im_end|>\n" for h in hist)
            raw += train_prompt(base.get("sent") or user_m["content"]) + base["content"]   # same file text and SymPy hint
            buf, err, ttft, was_stopped, secs = stream_into(
                slot, base["content"], [], MODES[base["mode"]]["adapter"], raw_prompt=raw)
            stop_slot.empty()
        ss["gen"] = None
        n_new = count_tokens(buf) if buf else 0
        base["content"] += buf
        base["truncated"] = (not was_stopped) and (not err) and n_new >= max_tokens - 2
        base["meta"] = (base.get("meta", "").split(" · continued")[0]
                        + f" · continued +{n_new} tokens" + (" · ⏹ stopped" if was_stopped else ""))
        if was_stopped:
            base["stopped"] = True
        if base.get("sympy"):
            base["check"] = sympy_check(base["sympy"], base["content"])
        save_chat()
        ss["announce"] = "Continuation stopped." if was_stopped else "Continuation complete."
        st.rerun()

    file_text = None
    if attachment is not None:
        try:
            file_text = attachment.getvalue().decode("utf-8", errors="replace")
        except Exception:
            file_text = None
    shown_prompt = (prompt or "").strip() or (FILE_DEFAULTS[mode] if file_text is not None else "")
    user_msg = shown_prompt
    if file_text is not None:
        cut = len(file_text) > MAX_FILE_CHARS
        user_msg += f"\n\n---\n{file_text[:MAX_FILE_CHARS]}\n---"
        shown_prompt += f"\n\n📎 *{attachment.name}*" + (f" (first {MAX_FILE_CHARS:,} characters used)" if cut else "")

    sympy_str = None
    if mode == "Math" and expr.strip() and not no_hint:
        res, perr = run_sympy(expr)
        if perr is None:
            sympy_str = str(res)
            user_msg += f"\n[SymPy result: {res}]"
        else:
            user_msg += f"\n[SymPy parsing error: {perr}]"

    history = []
    if remember or force_ctx:
        history, _ = build_context(ss["messages"], ctx_budget, max_turns=1 if (force_ctx and not remember) else None)

    ss["messages"].append({"role": "user", "content": shown_prompt})
    save_chat()                                   # saved immediately so the chat appears in the sidebar
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(shown_prompt)

    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(f'<span class="pill">{mode}</span>', unsafe_allow_html=True)
        slot, stop_slot = st.empty(), st.empty()
        slot.markdown(SKELETON, unsafe_allow_html=True)          # skeleton until the first token arrives
        stop_slot.button("⏹  Stop generating", key="stop_btn", on_click=stop_generation)
        ctx_turns = None if not (remember or force_ctx) else (1 if (force_ctx and not remember) else 0)
        ss["gen"] = {"mode": mode, "text": "", "sympy": sympy_str, "sent": user_msg, "ctx_turns": ctx_turns}
        buf, err, ttft, was_stopped, secs = stream_into(
            slot, "", history + [{"role": "user", "content": user_msg}], cfg["adapter"])
        stop_slot.empty()

    ss["gen"] = None
    n_tok = count_tokens(buf) if buf else 0
    if err and not buf:
        new_msg = {"role": "assistant", "content": f"⚠️ Generation failed: {err['e']}", "mode": mode, "error": True}
    else:
        new_msg = {"role": "assistant", "content": buf, "mode": mode, "meta": fmt_meta(n_tok, secs, ttft),
                   "sent": user_msg, "ctx_turns": ctx_turns}
        if was_stopped:
            new_msg.update(stopped=True, meta=f"⏹ Stopped · {n_tok} tokens")
        elif not err and n_tok >= max_tokens - 2:
            new_msg["truncated"] = True                 # hit the length limit: offer a Continue button
        if err:
            new_msg["meta"] += f" · ⚠️ interrupted: {err['e']}"
        if sympy_str and not was_stopped:
            new_msg["sympy"] = sympy_str
            new_msg["check"] = sympy_check(sympy_str, buf)
    ss["messages"].append(new_msg)
    save_chat()
    ss["announce"] = "Response stopped." if new_msg.get("stopped") else \
                     "Something went wrong." if new_msg.get("error") else "Response complete."
    st.rerun()                                    # refresh the sidebar list, action buttons and context line

st.caption("AI-generated answers can contain mistakes, and there are no web sources: check important facts. "
           "The ✅ badge is an automatic SymPy comparison, not a guarantee.")
