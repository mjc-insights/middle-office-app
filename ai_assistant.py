import os
import io
import json
import time
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv


# ===================== ENV/CONFIG =====================
load_dotenv(override=False)

AI_MODEL_DEFAULT = os.getenv("AI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
VECTOR_DIR = Path("data") / "vectorstore"
ALLOWED_EXTS = {".py", ".md", ".txt", ".csv", ".yaml", ".yml", ".pdf"}
EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", "data/vectorstore"}
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_RAG_TOKENS = 1500


# ===================== CACHEABLE RESOURCES =====================
def resolve_openai_key() -> str:
    """Return user-provided key from session if present, else .env/environment.

    The value is stripped. Never log or print the key.
    """
    try:
        # Prefer session value when available
        sess_val = (st.session_state.get("OPENAI_API_KEY", "") or "").strip()
    except Exception:
        sess_val = ""
    if sess_val:
        return sess_val
    return (os.getenv("OPENAI_API_KEY", "") or "").strip()


def _hash_key_material(api_key: str) -> str:
    import hashlib
    return hashlib.sha256(api_key.encode()).hexdigest()


def get_openai_client(api_key: Optional[str] = None):
    """Create an OpenAI client using a resolved key.

    Caches per key via a hash, not the raw secret. If no key can be
    resolved, raises a RuntimeError instructing to add it in the sidebar.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"OpenAI SDK not available: {e}")

    api_key = (api_key or resolve_openai_key()).strip()
    if not api_key:
        raise RuntimeError("OpenAI API key not set. Add it in the sidebar or .env")

    key_hash = _hash_key_material(api_key)

    @st.cache_resource(show_spinner=False)
    def _client_for_key(_key_hash: str):
        # Closure uses api_key but cache key is only the hash
        return OpenAI(api_key=api_key)

    return _client_for_key(key_hash)


@st.cache_resource(show_spinner=False)
def get_token_encoder():
    try:
        import tiktoken  # pyright: ignore[reportMissingImports]
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback: dummy counter
        class _Dummy:
            def encode(self, s: str):
                return list(s.split())
        return _Dummy()


@st.cache_data(show_spinner=False)
def _read_text_file(path: Path) -> str:
    try:
        # Handle PDFs best-effort (optional)
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore[reportMissingImports]  # optional dependency
                reader = PdfReader(str(path))
                return "\n".join([p.extract_text() or "" for p in reader.pages])
            except Exception:
                return ""  # silently skip if not parseable
        # For all other text-like types
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""


def _iter_repo_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter directories in-place to prune walk
        rel = Path(dirpath).relative_to(root)
        parts = set(p.lower() for p in rel.parts)
        if parts & {x.lower() for x in EXCLUDE_DIRS}:
            continue
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in ALLOWED_EXTS:
                # Skip obviously large/binary artifacts by size (> 2.5MB)
                try:
                    if p.stat().st_size > 2_500_000:
                        continue
                except Exception:
                    pass
                out.append(p)
    return out


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    tokens = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


def _hash_inputs(files: List[Path]) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted(files):
        try:
            st = p.stat()
            h.update(str(p).encode("utf-8"))
            h.update(str(int(st.st_mtime)).encode("utf-8"))
            h.update(str(int(st.st_size)).encode("utf-8"))
        except Exception:
            continue
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def load_or_build_faiss(root: Path) -> Tuple[Optional[object], Dict[int, Dict[str, str]], int]:
    """
    Returns (faiss_index, id_to_meta, embedding_dim).
    If nothing to index, returns (None, {}, 0).
    """
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    idx_path = VECTOR_DIR / "faiss.index"
    meta_path = VECTOR_DIR / "meta.json"
    sig_path = VECTOR_DIR / "signature.json"
    centroid_path = VECTOR_DIR / "centroid.json"

    files = _iter_repo_files(root)
    signature = _hash_inputs(files)

    # Try load existing if signature matches
    try:
        if idx_path.exists() and meta_path.exists() and sig_path.exists():
            saved_sig = json.loads(sig_path.read_text()).get("signature")
            if saved_sig == signature:
                import faiss  # type: ignore
                index = faiss.read_index(str(idx_path))
                id_to_meta = {int(k): v for k, v in json.loads(meta_path.read_text()).items()}
                # Infer dim
                dim = index.d
                return index, id_to_meta, dim
    except Exception:
        pass

    # Build fresh
    if not files:
        return None, {}, 0

    texts: List[str] = []
    metas: List[Dict[str, str]] = []
    with st.status("Indexing repository for RAG…", expanded=False) as status:
        for p in files:
            content = _read_text_file(p)
            chunks = _chunk_text(content)
            for ch in chunks:
                texts.append(ch)
                metas.append({"source": str(p), "type": p.suffix.lower()})
        status.update(label=f"Embedding {len(texts)} chunks…", state="running")

    if not texts:
        return None, {}, 0

    client = get_openai_client(api_key=resolve_openai_key())
    # Batch embeddings to respect payload sizes
    embeddings: List[List[float]] = []
    batch = 128
    for i in range(0, len(texts), batch):
        frag = texts[i:i + batch]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=frag)
            for d in resp.data:
                embeddings.append(d.embedding)
        except Exception:
            # Best-effort: skip failing batch
            continue

    if not embeddings:
        return None, {}, 0

    try:
        import faiss  # pyright: ignore[reportMissingImports]
    except Exception as e:
        raise RuntimeError(f"FAISS not available. Install faiss-cpu. Error: {e}")

    import numpy as np  # localize dependency
    vecs = np.array(embeddings, dtype="float32")
    # Normalize for inner-product similarity
    faiss = __import__("faiss")  # pyright: ignore[reportMissingImports]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # Persist
    try:
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(idx_path))
        meta_map = {i: metas[i] | {"text": texts[i][:5000]} for i in range(len(metas))}
        meta_path.write_text(json.dumps(meta_map))
        sig_path.write_text(json.dumps({"signature": signature, "count": len(texts), "dim": dim}))
        # Precompute and persist centroid for heuristic use
        try:
            centroid = (vecs.mean(axis=0) / (np.linalg.norm(vecs.mean(axis=0)) + 1e-12)).tolist()
            centroid_path.write_text(json.dumps({"dim": dim, "vector": centroid}))
        except Exception:
            pass
    except Exception:
        pass

    return index, {i: metas[i] | {"text": texts[i][:5000]} for i in range(len(metas))}, dim


def _retrieve(query: str, k: int = TOP_K) -> List[Dict[str, str]]:
    root = Path(__file__).resolve().parent
    try:
        index, id_to_meta, dim = load_or_build_faiss(root)
    except Exception as e:
        st.warning(f"RAG unavailable: {e}")
        return []
    if index is None:
        return []

    client = get_openai_client(api_key=resolve_openai_key())
    try:
        emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    except Exception:
        return []

    import numpy as np
    q = np.array([emb], dtype="float32")
    # Normalize like the index
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    scores, idxs = index.search(q, k)
    out: List[Dict[str, str]] = []
    for rank, idx in enumerate(idxs[0].tolist()):
        meta = id_to_meta.get(int(idx)) or {}
        if meta:
            out.append({
                "rank": str(rank + 1),
                "source": meta.get("source", ""),
                "excerpt": meta.get("text", "")[:1000],
            })
    return out


def _index_available() -> bool:
    try:
        if not VECTOR_DIR.exists():
            return False
        idx_path = VECTOR_DIR / "faiss.index"
        meta_path = VECTOR_DIR / "meta.json"
        sig_path = VECTOR_DIR / "signature.json"
        if not (idx_path.exists() and meta_path.exists() and sig_path.exists()):
            return False
        try:
            meta = json.loads(meta_path.read_text())
            return bool(meta)
        except Exception:
            return True  # index files exist; assume available
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def _load_centroid() -> Optional[List[float]]:
    try:
        centroid_path = VECTOR_DIR / "centroid.json"
        if centroid_path.exists():
            data = json.loads(centroid_path.read_text())
            vec = data.get("vector")
            if isinstance(vec, list) and len(vec) > 0:
                return [float(x) for x in vec]
    except Exception:
        return None
    return None


def _embed_and_normalize(text: str) -> Optional[List[float]]:
    try:
        client = get_openai_client(api_key=resolve_openai_key())
        emb = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
    except Exception:
        return None
    try:
        import numpy as np
        v = np.array(emb, dtype="float32")
        n = float(np.linalg.norm(v))
        if n == 0:
            return None
        return (v / n).tolist()
    except Exception:
        return None


def _cosine(a: List[float], b: List[float]) -> float:
    try:
        import numpy as np
        va = np.array(a, dtype="float32")
        vb = np.array(b, dtype="float32")
        return float((va * vb).sum())
    except Exception:
        return 0.0


def _cheap_probe_similarity(query: str) -> float:
    # Return top-1 similarity from FAISS, or -1.0 on failure
    root = Path(__file__).resolve().parent
    try:
        index, id_to_meta, dim = load_or_build_faiss(root)
    except Exception:
        return -1.0
    if index is None:
        return -1.0
    try:
        client = get_openai_client(api_key=resolve_openai_key())
        emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    except Exception:
        return -1.0
    try:
        import numpy as np
        q = np.array([emb], dtype="float32")
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        scores, idxs = index.search(q, 1)
        if scores is None or len(scores.flatten()) == 0:
            return -1.0
        return float(scores.flatten()[0])
    except Exception:
        return -1.0


def heuristic_decision(query: str) -> bool:
    # Guard: require index availability
    if not _index_available():
        return False

    q = (query or "").strip()
    if not q:
        return False

    # Heuristic A — file cues
    try:
        pattern = re.compile(r"app\.py|ui_pages\.py|ai_assistant\.py|\\.py\\b|\\.csv\\b|\\.ya?ml\\b|\\.pdf\\b|function|class|def |error|traceback|stack", re.IGNORECASE)
        if pattern.search(q):
            return True
    except Exception:
        pass

    flag = False

    # Heuristic B — query length
    try:
        if len(q.split()) >= 12:
            flag = True
    except Exception:
        pass

    # Heuristic C — semantic proximity to repo centroid
    if not flag:
        centroid = _load_centroid()
        if centroid:
            q_vec = _embed_and_normalize(q)
            if q_vec:
                sim = _cosine(centroid, q_vec)
                if sim >= 0.18:
                    flag = True

    # Heuristic D — cheap probe top-1
    if not flag and _index_available():
        sim = _cheap_probe_similarity(q)
        if sim >= 0.30:
            flag = True

    return bool(flag and _index_available())


def _dedupe_by_source(context: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for c in context:
        src = str(c.get("source", ""))
        if src and src not in seen:
            seen.add(src)
            out.append(c)
    return out


def _cap_context_tokens(context: List[Dict[str, str]], max_tokens: int = MAX_RAG_TOKENS) -> List[Dict[str, str]]:
    enc = get_token_encoder()
    total = 0
    kept: List[Dict[str, str]] = []
    for c in context:
        ex = c.get("excerpt", "")
        toks = 0
        try:
            toks = len(enc.encode(ex))
        except Exception:
            toks = len((ex or "").split())
        if total + toks > max_tokens:
            break
        kept.append(c)
        total += toks
    return kept


def _build_system_prompt(mode: str, rag_context: Optional[List[Dict[str, str]]] = None) -> str:
    if mode == "Repo-aware":
        # Prefer code files first to strengthen code-level answers
        rag_sorted = sorted(
            (rag_context or []),
            key=lambda c: (0 if str(c.get("source", "")).lower().endswith(".py") else 1),
        )
        # Format with code fences where appropriate
        def _fmt(c: Dict[str, str]) -> str:
            src = str(c.get("source", ""))
            ex = c.get("excerpt", "")
            ext = Path(src).suffix.lower()
            if ext == ".py":
                return f"[{c.get('rank')}] {src}\n```python\n{ex}\n```"
            if ext in {".md", ".markdown"}:
                return f"[{c.get('rank')}] {src}\n```md\n{ex}\n```"
            return f"[{c.get('rank')}] {src}\n```\n{ex}\n```"
        ctx = "\n\n".join([_fmt(c) for c in rag_sorted])
        # Append live facts when available
        tool_facts = st.session_state.get("__ai_tool_facts", "") or ""
        tool_block = f"\nLive repo facts:\n{tool_facts}\n" if tool_facts else ""
        return (
            "You are an assistant for a Streamlit middle-office app. Use the provided repo context when relevant.\n"
            "Answer clearly, be concise, and avoid fabrications. If unsure, say so.\n"
            "When referencing code, mention file paths (e.g., app.py, ui_pages.py).\n"
            "\nFor exception lists, use the preview tool on trades_exceptions_df to get a clean, readable format.\n"
            "When presenting data, format it clearly and avoid overwhelming users with too many columns.\n"
            f"\nContext (top-k):\n{ctx}\n"
            f"{tool_block}"
        )
    return (
        "You are a helpful AI assistant for this app. Provide concise, accurate answers."
    )


def run_openai_chat(messages: List[Dict[str, str]], system_prompt: str, stream: bool = True, container=None) -> Tuple[str, float]:
    client = get_openai_client(api_key=resolve_openai_key())
    full_text = ""
    t0 = time.perf_counter()
    try:
        # Phase 1: tiny intent router determines whether to bias toward tools first
        user_text = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        intent = route_query(user_text) if user_text else ""

        # Phase 2: tool negotiation (non-streamed) using function-calling when data intents
        work_messages = [{"role": "system", "content": system_prompt}] + messages
        tools = _tool_specs() if intent in (INTENT_METRIC_AGG, INTENT_LOOKUP_PREVIEW) else _tool_specs()
        max_steps = 4
        step = 0
        while step < max_steps:
            step += 1
            resp = client.chat.completions.create(
                model=AI_MODEL_DEFAULT,
                messages=work_messages,
                temperature=0.2,
                tools=tools,
                tool_choice="auto" if intent in (INTENT_METRIC_AGG, INTENT_LOOKUP_PREVIEW) else "none",
                stream=False,
            )
            choice = resp.choices[0]
            msg = choice.message
            tool_calls = msg.tool_calls or []
            if not tool_calls:
                # No tool call — assistant content ready; keep messages as-is
                break
            # Append assistant message with tool calls
            work_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    } for tc in tool_calls
                ],
            })
            # Execute tools sequentially and append results
            import json as _json
            for tc in tool_calls:
                fn = (tc.function.name or "").strip()
                try:
                    args = _json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                if fn == "aggregate":
                    result = _tool_aggregate(args)
                elif fn == "unique_values":
                    result = _tool_unique_values(args)
                elif fn == "preview":
                    result = _tool_preview(args)
                else:
                    result = {"error": "unknown_tool"}
                work_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn,
                    "content": _json.dumps(result, ensure_ascii=False),
                })

        # Phase 3: final generation; stream for UX
        if stream:
            with (container or st).chat_message("assistant"):
                placeholder = st.empty()
                # Show thinking text immediately
                placeholder.markdown("_Thinking…_")
                
                # Determine if we should show expanded status for data queries
                show_expanded = intent in (INTENT_METRIC_AGG, INTENT_LOOKUP_PREVIEW)
                status = st.status("Thinking…", expanded=show_expanded)
                
                try:
                    # Update status based on what we're doing
                    if intent in (INTENT_METRIC_AGG, INTENT_LOOKUP_PREVIEW):
                        status.write("Analyzing your question…")
                        status.write("Checking data tables…")
                        if step > 1:  # If we ran tools
                            status.write("Processing data with tools…")
                        status.write("Composing answer…")
                    else:
                        status.write("Analyzing your question…")
                        status.write("Composing answer…")
                    
                    resp2 = client.chat.completions.create(
                        model=AI_MODEL_DEFAULT,
                        messages=work_messages,
                        temperature=0.2,
                        tools=_tool_specs(),
                        tool_choice="none",
                        stream=True,
                    )
                    for event in resp2:
                        try:
                            delta = event.choices[0].delta.content or ""
                        except Exception:
                            delta = ""
                        if delta:
                            full_text += delta
                            placeholder.markdown(full_text)
                finally:
                    status.update(label="Done", state="complete")
        else:
            with (container or st).chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("_Thinking…_")
                
                resp2 = client.chat.completions.create(
                model=AI_MODEL_DEFAULT,
                    messages=work_messages,
                temperature=0.2,
                    tools=_tool_specs(),
                    tool_choice="none",
                stream=False,
            )
                full_text = resp2.choices[0].message.content or ""
                placeholder.markdown(full_text)
    except Exception as e:
        with (container or st).chat_message("assistant"):
            st.warning(f"Model error: {e}")
        return "", 0.0
    dt = time.perf_counter() - t0
    return full_text, dt


def _ensure_session_state():
    if "ai_msgs" not in st.session_state:
        st.session_state["ai_msgs"] = []  # list[dict]: {role, content}
    if "__ai_last_stats" not in st.session_state:
        st.session_state["__ai_last_stats"] = {"latency": 0.0, "prompt_tokens": 0, "output_tokens": 0}


def _estimate_tokens(messages: List[Dict[str, str]], reply: str) -> Tuple[int, int]:
    enc = get_token_encoder()
    try:
        prompt_tokens = sum(len(enc.encode(m.get("content", ""))) for m in messages)
        output_tokens = len(enc.encode(reply))
        return prompt_tokens, output_tokens
    except Exception:
        # Fallback rough estimate
        prompt_tokens = sum(len((m.get("content", "") or "").split()) for m in messages)
        output_tokens = len((reply or "").split())
        return prompt_tokens, output_tokens


# ===================== LIGHTWEIGHT REPO FACTS (CSV-driven) =====================
@st.cache_data(show_spinner=False)
def _get_data_dir_cached() -> Path:
    # Avoid importing app.py here (app defines widgets at import-time),
    # which can trigger CachedWidgetWarning. Resolve local data/ instead.
    p = Path(__file__).resolve().parent / "data"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


@st.cache_data(show_spinner=False)
def _read_csv_safe(path: Path):
    try:
        import pandas as pd
        return pd.read_csv(path)
    except Exception:
        return None


def _repo_insights_for(question: str) -> str:
    """Return compact, question-aware facts computed from session DataFrames with disk fallbacks.

    Rules:
    - Only emit facts for exception/break or stp/ready queries; else return empty string.
    - Prefer in-memory DataFrames from st.session_state, fall back to data/*.csv.
    - Keep the output short and machine-readable; <= ~400 chars.
    """
    q = (question or "").lower()
    wants_exceptions = ("exception" in q) or ("break" in q)
    wants_stp = ("stp" in q) or ("ready" in q)
    wants_pct = ("percent" in q) or ("percentage" in q) or ("pct" in q) or ("rate" in q)
    wants_by_rule = ("by rule" in q) or ("rule" in q)
    wants_reasons = ("reason" in q) or ("reasons" in q)
    wants_today = ("today" in q) or ("right now" in q) or ("now" in q)

    if not (wants_exceptions or wants_stp):
        return ""

    try:
        import pandas as pd
    except Exception:
        pd = None  # type: ignore
    from datetime import datetime as _dt
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        ZoneInfo = None  # type: ignore

    def _today_date():
        try:
            return (_dt.now(ZoneInfo("America/New_York")) if ZoneInfo else _dt.utcnow()).date()
        except Exception:
            return _dt.utcnow().date()

    def _is_valid_df(x) -> bool:
        try:
            return (x is not None) and hasattr(x, "empty") and (not x.empty)
        except Exception:
            return False

    # Acquire DataFrames: session -> disk
    df_exc = st.session_state.get("trades_exceptions_df")
    df_clean = st.session_state.get("trades_clean_df")

    # Resolve data dir using app.get_data_dir if available; else local cache dir
    try:
        from app import get_data_dir as _gdd  # type: ignore
        data_dir = _gdd()
    except Exception:
        data_dir = _get_data_dir_cached()

    if (not _is_valid_df(df_exc)) and pd is not None:
        p_exc = Path(data_dir) / "trades_exceptions.csv"
        if p_exc.exists():
            df_exc = _read_csv_safe(p_exc)
    if (not _is_valid_df(df_clean)) and pd is not None:
        p_clean = Path(data_dir) / "trades_clean.csv"
        if p_clean.exists():
            df_clean = _read_csv_safe(p_clean)

    facts: List[str] = []

    # Helper: robust column resolution by canonical name, case/space-insensitive
    def _resolve_col(df, canonical: str) -> Optional[str]:
        try:
            norm = canonical.lower().replace(" ", "_")
            for c in list(df.columns):
                cc = str(c).lower().replace(" ", "_")
                if cc == norm:
                    return c
        except Exception:
            return None
        return None

    # Compute exception facts
    if wants_exceptions:
        if not _is_valid_df(df_exc):
            # If the user asked for exceptions but we have no artifacts at all
            if not _is_valid_df(df_clean):
                return "data_status=no_artifacts_found;instruction=Run Trade Capture & Data Quality Review first"
        else:
            try:
                df = df_exc.copy()
                # Date filter to today when possible
                dcol = _resolve_col(df, "trade_date")
                if wants_today and dcol is not None and pd is not None:
                    try:
                        df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date
                        df = df[df[dcol] == _today_date()]
                    except Exception:
                        pass
                total_rows = int(len(df))

                # Group by rule code
                rcol = _resolve_col(df, "Rule Code") or _resolve_col(df, "rule_code") or _resolve_col(df, "rule code")
                top_pairs: List[str] = []
                if rcol is not None and pd is not None and total_rows > 0 and (wants_by_rule or wants_today):
                    try:
                        counts = df.groupby(rcol).size().sort_values(ascending=False)
                        for k, v in list(counts.items())[:6]:
                            top_pairs.append(f"{str(k)}:{int(v)}")
                    except Exception:
                        pass
                
                if top_pairs:
                    if wants_today:
                        facts.append(f"exceptions_by_rule_today={','.join(top_pairs)} (total_today={total_rows})")
                    else:
                        facts.append(f"exceptions_by_rule_total={','.join(top_pairs)} (total={total_rows})")
                else:
                    if wants_today:
                        facts.append(f"exceptions_today_total={total_rows}")
                    else:
                        facts.append(f"exceptions_total={total_rows}")

                # Top reasons
                reason_col = _resolve_col(df, "Exception Reason") or _resolve_col(df, "exception_reason")
                if reason_col is not None and pd is not None and total_rows > 0 and (wants_reasons or wants_today):
                    try:
                        rc = (
                            df[reason_col]
                            .astype(str)
                            .str.strip()
                            .replace({"": None})
                            .dropna()
                        )
                        if not rc.empty:
                            rcounts = rc.value_counts().head(3)
                            pairs = [f"{k}:{int(v)}" for k, v in rcounts.items()]
                            if pairs:
                                if wants_today:
                                    facts.append(f"top_exception_reasons_today={','.join(pairs)}")
                                else:
                                    facts.append(f"top_exception_reasons_total={','.join(pairs)}")
                    except Exception:
                        pass
            except Exception:
                pass

    # Compute STP facts
    if wants_stp:
        if not _is_valid_df(df_clean):
            # Only emit fallback if we didn't already emit exception facts
            if not facts:
                return "data_status=no_artifacts_found;instruction=Run Trade Capture & Data Quality Review first"
        else:
            try:
                df = df_clean.copy()
                dcol = _resolve_col(df, "trade_date")
                if wants_today and dcol is not None and pd is not None:
                    try:
                        df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date
                        today_df = df[df[dcol] == _today_date()]
                        facts.append(f"stp_ready_today={int(len(today_df))}")
                    except Exception:
                        facts.append(f"stp_ready_total={int(len(df))}")
                else:
                    facts.append(f"stp_ready_total={int(len(df))}")
                # Optionally compute STP percentage if both clean and exceptions are present
                if wants_pct:
                    total_pool = int(len(df)) + (int(len(df_exc)) if _is_valid_df(df_exc) else 0)
                    if total_pool > 0:
                        pct = round(int(len(df)) / total_pool * 100.0, 2)
                        facts.append(f"stp_ready_pct_total={pct}")
            except Exception:
                pass

    # Final formatting and guard on length
    out = "\n".join([f for f in facts if f])
    if len(out) > 400:
        out = out[:397] + "..."
    return out


# ===================== DATA DICTIONARY BUILDER =====================
# Canonical -> synonyms map used for downstream semantic alignment (not wired yet)
ALIAS_MAP: dict = {
    "ticker": ["ticker", "symbol", "security", "security ticker", "bbg ticker"],
    "counterparty": ["counterparty", "broker", "executing broker", "custodian"],
    "settlement_date": ["settlement_date", "settle date", "settl_date", "settlement_dt"],
    "trade_date": ["trade_date", "trade dt", "tradedate", "trade_dt"],
    "status": ["status", "settlement status", "affirmation status"],
    "quantity": ["quantity", "qty", "shares", "units"],
    "price": ["price", "px", "trade price", "avg price"],
    "notional_amount": ["notional", "notional_amount", "trade_value", "gross_amount", "net_amount", "value"],
}


def build_data_dictionary() -> dict:
    """Inspect live session DataFrames and return a compact JSON-serializable data dictionary.

    Does not modify global state. Not wired into prompts/UI by design.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore
    from datetime import datetime as _dt

    table_keys = [
        "trades_exceptions_df",
        "trades_clean_df",
        "enriched_trades_df",
        "trades_all_df",
    ]

    def _is_df(x) -> bool:
        try:
            if pd is None:
                return False
            import pandas as _pd  # local alias to ensure type availability
            return isinstance(x, _pd.DataFrame) and not x.empty
        except Exception:
            return False

    def _infer_col_type(series) -> str:
        if pd is None:
            return "string"
        try:
            from pandas.api import types as ptypes  # type: ignore
        except Exception:
            ptypes = None  # type: ignore

        try:
            if ptypes is not None:
                if ptypes.is_bool_dtype(series):
                    return "boolean"
                if ptypes.is_integer_dtype(series):
                    return "integer"
                if ptypes.is_float_dtype(series):
                    return "number"
                if ptypes.is_datetime64_any_dtype(series):
                    return "datetime"
        except Exception:
            pass

        # Heuristic date detection on string/object columns
        try:
            s = series.dropna()
            if hasattr(s, "astype"):
                s = s.astype(str).str.strip()
            s = s[s != ""]
            if len(s) == 0:
                return "string"
            # Match full-string formats: YYYY-MM-DD or M/D/YYYY
            iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
            us_mask = s.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$", na=False)
            if bool((iso_mask | us_mask).all()):
                return "date"
        except Exception:
            pass

        return "string"

    def _sample_values(series, limit: int = 15) -> list:
        out: list = []
        seen = set()
        try:
            for v in series:
                # Skip null-like
                if v is None:
                    continue
                try:
                    import math
                    if isinstance(v, float) and math.isnan(v):
                        continue
                except Exception:
                    pass
                s = str(v)
                if not s:
                    continue
                if s in seen:
                    continue
                seen.add(s)
                out.append(s[:40])
                if len(out) >= limit:
                    break
        except Exception:
            return out
        return out

    tables: dict = {}
    for key in table_keys:
        try:
            df = st.session_state.get(key)
        except Exception:
            df = None
        if not _is_df(df):
            continue
        try:
            n_rows = int(len(df))
        except Exception:
            n_rows = 0
        columns: dict = {}
        try:
            for col in list(df.columns):
                try:
                    col_series = df[col]
                except Exception:
                    continue
                col_type = _infer_col_type(col_series)
                samples = _sample_values(col_series, limit=15)
                columns[str(col)] = {"type": col_type, "samples": samples}
        except Exception:
            columns = {}
        tables[key] = {"n_rows": n_rows, "columns": columns}

    gen_ts = _dt.utcnow().isoformat(timespec="seconds") + "Z"
    return {"generated_at": gen_ts, "tables": tables, "aliases": ALIAS_MAP}


# ===================== DATA TOOLS (FUNCTION-CALLING) =====================
# Whitelisted session DataFrames for tool operations
TABLE_WHITELIST = {"trades_exceptions_df", "trades_clean_df", "enriched_trades_df", "trades_all_df"}


def _tool_specs() -> List[Dict[str, object]]:
    enum_tables = sorted(list(TABLE_WHITELIST))
    return [
        {
            "type": "function",
            "function": {
                "name": "aggregate",
                "description": "Group-by and aggregation over a whitelisted in-memory table.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "enum": enum_tables},
                        "dims": {"type": "array", "items": {"type": "string"}},
                        "measures": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "op": {"type": "string", "enum": ["count", "sum", "avg", "min", "max"]},
                                    "col": {"type": "string"}
                                },
                                "required": ["op", "col"]
                            },
                            "minItems": 1
                        },
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "col": {"type": "string"},
                                    "op": {"type": "string", "enum": ["eq", "neq", "in", "not_in", "lt", "lte", "gt", "gte", "between", "contains", "startswith", "endswith"]},
                                    "value": {}
                                },
                                "required": ["col", "op", "value"]
                            }
                        },
                        "order": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "col": {"type": "string"},
                                    "dir": {"type": "string", "enum": ["asc", "desc"]}
                                },
                                "required": ["col", "dir"]
                            }
                        },
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100}
                    },
                    "required": ["table", "measures"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "unique_values",
                "description": "Return distinct values for a column in a whitelisted in-memory table.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "enum": enum_tables},
                        "column": {"type": "string"},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "col": {"type": "string"},
                                    "op": {"type": "string", "enum": ["eq", "neq", "in", "not_in", "lt", "lte", "gt", "gte", "between", "contains", "startswith", "endswith"]},
                                    "value": {}
                                },
                                "required": ["col", "op", "value"]
                            }
                        },
                        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}
                    },
                    "required": ["table", "column"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "preview",
                "description": "Return sample rows (post-filter) from a whitelisted in-memory table for transparency. For exceptions data, this returns a clean, readable list format instead of a wide table.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "enum": enum_tables},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "col": {"type": "string"},
                                    "op": {"type": "string", "enum": ["eq", "neq", "in", "not_in", "lt", "lte", "gt", "gte", "between", "contains", "startswith", "endswith"]},
                                    "value": {}
                                },
                                "required": ["col", "op", "value"]
                            }
                        },
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
                    },
                    "required": ["table"]
                }
            }
        }
    ]


def _ci_resolve(df_cols: List[str], name: str) -> Optional[str]:
    target = (name or "").strip().lower()
    for c in df_cols:
        if str(c).strip().lower() == target:
            return c
    return None


def _parse_date_like(val):
    try:
        import pandas as pd  # type: ignore
        if isinstance(val, list):
            return [pd.to_datetime(x, errors="coerce") for x in val]
        return pd.to_datetime(val, errors="coerce")
    except Exception:
        return val


def _coerce_numeric(s):
    try:
        import pandas as pd  # type: ignore
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return s


def _normalize_status_series(s):
    try:
        ss = s.astype(str).str.strip().str.lower()
        closed = {"settled", "closed", "matched", "affirmed"}
        openish = {"open", "pending", "unmatched", "unaﬃrmed", "unaffirmed"}
        return ss.replace({v: "Closed" for v in closed}).replace({v: "Open" for v in openish})
    except Exception:
        return s


def _apply_filters(df, filters: Optional[List[Dict]]):
    if not filters:
        return df
    import pandas as pd  # type: ignore
    mask = pd.Series([True] * len(df), index=df.index)
    for f in filters:
        try:
            col_in = str(f.get("col", ""))
            op = str(f.get("op", ""))
            val = f.get("value")
            col = _ci_resolve(list(df.columns), col_in)
            if col is None:
                continue
            s = df[col]
            if col.lower() == "status":
                s = _normalize_status_series(s)
            if isinstance(val, (str, list)) and op in ("lt", "lte", "gt", "gte", "between"):
                val = _parse_date_like(val)
            if op == "eq":
                m = s == val
            elif op == "neq":
                m = s != val
            elif op == "in":
                m = s.isin(val if isinstance(val, list) else [val])
            elif op == "not_in":
                m = ~s.isin(val if isinstance(val, list) else [val])
            elif op == "lt":
                m = s < val
            elif op == "lte":
                m = s <= val
            elif op == "gt":
                m = s > val
            elif op == "gte":
                m = s >= val
            elif op == "between":
                lo, hi = (val or [None, None])[:2]
                m = (s >= lo) & (s <= hi)
            elif op == "contains":
                m = s.astype(str).str.contains(str(val), case=False, na=False)
            elif op == "startswith":
                m = s.astype(str).str.startswith(str(val), na=False)
            elif op == "endswith":
                m = s.astype(str).str.endswith(str(val), na=False)
            else:
                m = pd.Series([True] * len(df), index=df.index)
            mask &= m.fillna(False)
        except Exception:
            continue
    return df[mask]


def _tool_table(name: str):
    if name not in TABLE_WHITELIST:
        return None
    try:
        import pandas as pd  # noqa: F401
    except Exception:
        return None
    df = st.session_state.get(name)
    try:
        import pandas as _pd
        if isinstance(df, _pd.DataFrame) and not df.empty:
            return df
    except Exception:
        return None
    # Disk fallback for core artifacts (read-only)
    try:
        from app import get_data_dir as _gdd  # type: ignore
        data_dir = _gdd()
    except Exception:
        data_dir = _get_data_dir_cached()
    try:
        mapping = {
            "trades_clean_df": "trades_clean.csv",
            "trades_exceptions_df": "trades_exceptions.csv",
        }
        csv_name = mapping.get(name)
        if csv_name:
            p = Path(data_dir) / csv_name
            if p.exists():
                df2 = _read_csv_safe(p)
                try:
                    if isinstance(df2, _pd.DataFrame) and not df2.empty:
                        return df2
                except Exception:
                    pass
    except Exception:
        pass

    # Secondary fallback: pick the first available whitelisted table in session or disk
    fallback_order = [
        "trades_exceptions_df",
        "trades_clean_df",
        "enriched_trades_df",
        "trades_all_df",
    ]
    for alt in fallback_order:
        try:
            alt_df = st.session_state.get(alt)
            if isinstance(alt_df, _pd.DataFrame) and not alt_df.empty:
                return alt_df
        except Exception:
            pass
        # disk for core artifacts
        if alt in mapping:
            try:
                p = Path(data_dir) / mapping[alt]
                if p.exists():
                    df2 = _read_csv_safe(p)
                    if isinstance(df2, _pd.DataFrame) and not df2.empty:
                        return df2
            except Exception:
                pass
    return None


def _jsonify_scalar(x):
    try:
        import numpy as np
        if isinstance(x, (np.generic,)):
            return x.item()
    except Exception:
        pass
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    s = str(x) if not isinstance(x, (int, float, bool, type(None))) else x
    if isinstance(s, str) and len(s) > 120:
        s = s[:120]
    return s


def _tool_aggregate(args: dict) -> dict:
    try:
        table = str(args.get("table", ""))
        df = _tool_table(table)
        if df is None:
            return {"error": "table_not_loaded", "hint": "Run Trade Capture & Data Quality Review first"}
        dims = [str(d) for d in (args.get("dims") or [])]
        measures = args.get("measures") or []
        filters = args.get("filters") or []
        order = args.get("order") or []
        limit = int(args.get("limit") or 100)
        limit = max(1, min(500, limit))

        df_f = _apply_filters(df, filters)

        resolved_dims = []
        for d in dims:
            c = _ci_resolve(list(df_f.columns), d)
            if c is None:
                return {"error": "unknown_column", "message": f"Unknown dim: {d}", "available": list(map(str, df_f.columns))}
            resolved_dims.append(c)

        import pandas as pd  # type: ignore

        # Resolve measure columns, supporting derived notional/exposure (quantity * price)
        def _resolve_column_or_derived(df_in, name: str):
            nm = (name or "").strip()
            direct = _ci_resolve(list(df_in.columns), nm)
            if direct is not None:
                return direct, None, direct
            nm_l = nm.lower()
            if nm_l in {"notional", "notional_amount", "trade_value", "gross_amount", "net_amount", "value", "exposure", "exposure_abs", "cost", "cost_basis"}:
                qname = None
                pname = None
                for q in ["quantity", "qty", "shares", "units"]:
                    qname = _ci_resolve(list(df_in.columns), q)
                    if qname is not None:
                        break
                for p in ["price", "px", "trade price", "avg price", "unit cost", "unit_cost_num", "Unit Cost"]:
                    pname = _ci_resolve(list(df_in.columns), p)
                    if pname is not None:
                        break
                if qname is not None and pname is not None:
                    qv = _coerce_numeric(df_in[qname]).abs()
                    pv = _coerce_numeric(df_in[pname]).abs()
                    series = (qv * pv)
                    return "notional_amount", series, "notional_amount"
            return None, None, None

        rows = []
        col_names = resolved_dims.copy()
        meas_specs = []
        for m in measures:
            op = str(m.get("op", "")).lower()
            col = str(m.get("col", ""))
            if op == "count" and col in ("*", ""):
                out_col = "count_*"
                meas_specs.append((op, col, out_col, None))
            else:
                rc, derived_series, label = _resolve_column_or_derived(df_f, col)
                if rc is None and derived_series is None:
                    return {"error": "unknown_column", "message": f"Unknown measure column: {col}", "available": list(map(str, df_f.columns))}
                use_label = label or rc or col
                out_col = f"{op}_{use_label}"
                meas_specs.append((op, rc or use_label, out_col, derived_series))
            col_names.append(meas_specs[-1][2])

        if resolved_dims:
            gb = df_f.groupby(resolved_dims, dropna=False, sort=False)
            for keys, grp in gb:
                if not isinstance(keys, tuple):
                    keys = (keys,)
                row_vals = list(keys)
                for op, rc, out_name, derived_series in meas_specs:
                    if op == "count":
                        v = int(len(grp))
                    else:
                        if derived_series is not None:
                            s = _coerce_numeric(derived_series.loc[grp.index])
                        else:
                            s = _coerce_numeric(grp[rc])
                        if op == "sum":
                            v = float(pd.to_numeric(s, errors="coerce").sum(skipna=True))
                        elif op == "avg":
                            v = float(pd.to_numeric(s, errors="coerce").mean(skipna=True))
                        elif op == "min":
                            v = _jsonify_scalar(pd.to_numeric(s, errors="coerce").min(skipna=True))
                        elif op == "max":
                            v = _jsonify_scalar(pd.to_numeric(s, errors="coerce").max(skipna=True))
                        else:
                            v = None
                    row_vals.append(_jsonify_scalar(v))
                rows.append(row_vals)
            res_df = pd.DataFrame(rows, columns=col_names)
        else:
            row_vals = []
            for op, rc, out_name, derived_series in meas_specs:
                if op == "count":
                    v = int(len(df_f))
                else:
                    if derived_series is not None:
                        s = _coerce_numeric(derived_series)
                    else:
                        s = _coerce_numeric(df_f[rc])
                    if op == "sum":
                        v = float(pd.to_numeric(s, errors="coerce").sum(skipna=True))
                    elif op == "avg":
                        v = float(pd.to_numeric(s, errors="coerce").mean(skipna=True))
                    elif op == "min":
                        v = _jsonify_scalar(pd.to_numeric(s, errors="coerce").min(skipna=True))
                    elif op == "max":
                        v = _jsonify_scalar(pd.to_numeric(s, errors="coerce").max(skipna=True))
                    else:
                        v = None
                row_vals.append(_jsonify_scalar(v))
            res_df = pd.DataFrame([row_vals], columns=[m[2] for m in meas_specs])

        for ord_spec in order:
            try:
                c = str(ord_spec.get("col"))
                d = str(ord_spec.get("dir", "asc")).lower()
                if c in res_df.columns:
                    res_df.sort_values(by=c, ascending=(d == "asc"), inplace=True)
            except Exception:
                continue

        res_df = res_df.head(limit)
        cols = [str(c) for c in res_df.columns]
        data_rows = [[_jsonify_scalar(v) for v in r] for r in res_df.itertuples(index=False, name=None)]
        from datetime import date as _date
        return {"columns": cols, "rows": data_rows, "as_of": _date.today().isoformat()}
    except Exception as e:
        return {"error": "aggregate_failed", "message": str(e)}


def _tool_unique_values(args: dict) -> dict:
    try:
        table = str(args.get("table", ""))
        df = _tool_table(table)
        if df is None:
            return {"error": "table_not_loaded", "hint": "Run Trade Capture & Data Quality Review first"}
        col_in = str(args.get("column", ""))
        filters = args.get("filters") or []
        limit = int(args.get("limit") or 100)
        limit = max(1, min(1000, limit))
        col = _ci_resolve(list(df.columns), col_in)
        if col is None:
            return {"error": "unknown_column", "available": list(map(str, df.columns))}
        df_f = _apply_filters(df, filters)
        s = df_f[col]
        vals = []
        seen = set()
        for v in s.dropna().tolist():
            j = _jsonify_scalar(v)
            if j in seen:
                continue
            seen.add(j)
            vals.append(j)
            if len(vals) >= limit:
                break
        from datetime import date as _date
        return {"column": str(col), "values": vals, "as_of": _date.today().isoformat()}
    except Exception as e:
        return {"error": "unique_values_failed", "message": str(e)}


def _tool_preview(args: dict) -> dict:
    try:
        table = str(args.get("table", ""))
        df = _tool_table(table)
        if df is None:
            return {"error": "table_not_loaded", "hint": "Run Trade Capture & Data Quality Review first"}
        filters = args.get("filters") or []
        limit = int(args.get("limit") or 20)
        limit = max(1, min(100, limit))
        df_f = _apply_filters(df, filters)
        df_h = df_f.head(limit)
        
        # Special formatting for exceptions table to make it more readable
        if table == "trades_exceptions_df" and not df_h.empty:
            return _format_exceptions_list(df_h, limit)
        
        # Default table format for other tables
        cols = [str(c) for c in df_h.columns]
        rows = [[_jsonify_scalar(v) for v in r] for r in df_h.itertuples(index=False, name=None)]
        from datetime import date as _date
        return {"columns": cols, "rows": rows, "as_of": _date.today().isoformat()}
    except Exception as e:
        return {"error": "preview_failed", "message": str(e)}


def _format_exceptions_list(df, limit: int) -> dict:
    """Format exceptions data as a clean, readable list instead of a wide table."""
    try:
        import pandas as pd
        from datetime import date as _date
        
        # Select key columns for readability
        key_cols = []
        col_mapping = {}
        
        # Map common column names to display names
        for col in df.columns:
            col_str = str(col).lower()
            if 'security' in col_str and 'id' in col_str:
                key_cols.append(col)
                col_mapping[col] = 'Security'
            elif 'counterparty' in col_str and 'legal' in col_str:
                key_cols.append(col)
                col_mapping[col] = 'Counterparty'
            elif 'exception' in col_str and 'reason' in col_str:
                key_cols.append(col)
                col_mapping[col] = 'Issue'
            elif 'rule' in col_str and 'code' in col_str:
                key_cols.append(col)
                col_mapping[col] = 'Rule'
            elif 'severity' in col_str and 'label' in col_str:
                key_cols.append(col)
                col_mapping[col] = 'Severity'
            elif 'quantity' in col_str:
                key_cols.append(col)
                col_mapping[col] = 'Qty'
            elif 'price' in col_str:
                key_cols.append(col)
                col_mapping[col] = 'Price'
            elif 'trade' in col_str and 'date' in col_str:
                key_cols.append(col)
                col_mapping[col] = 'Trade Date'
        
        # If no key columns found, use first few columns
        if not key_cols:
            key_cols = list(df.columns)[:6]
            col_mapping = {col: str(col) for col in key_cols}
        
        # Format as bullet point list
        formatted_items = []
        for i, (_, row) in enumerate(df.head(limit).iterrows(), 1):
            # Get the main identifier (Security + Counterparty or just Security)
            security = None
            counterparty = None
            issue = None
            rule = None
            severity = None
            qty = None
            price = None
            trade_date = None
            
            for col in key_cols:
                if col in df.columns:
                    value = _jsonify_scalar(row[col])
                    if value is not None and str(value).strip() != '' and str(value).lower() != 'nan':
                        col_str = str(col).lower()
                        if 'security' in col_str and 'id' in col_str:
                            security = value
                        elif 'counterparty' in col_str and 'legal' in col_str:
                            counterparty = value
                        elif 'exception' in col_str and 'reason' in col_str:
                            issue = value
                        elif 'rule' in col_str and 'code' in col_str:
                            rule = value
                        elif 'severity' in col_str and 'label' in col_str:
                            severity = value
                        elif 'quantity' in col_str:
                            qty = value
                        elif 'price' in col_str:
                            price = value
                        elif 'trade' in col_str and 'date' in col_str:
                            trade_date = value
            
            # Create a clean bullet point entry
            main_desc = f"**{security}**" if security else f"Exception #{i}"
            if counterparty:
                main_desc += f" with {counterparty}"
            if qty and price:
                main_desc += f" ({qty} @ {price})"
            
            bullet_item = f"• {main_desc}"
            
            # Add sub-bullets for key details
            if issue:
                bullet_item += f"\n  - **Issue**: {issue}"
            if rule:
                bullet_item += f"\n  - **Rule**: {rule}"
            if severity:
                bullet_item += f"\n  - **Severity**: {severity}"
            if trade_date:
                bullet_item += f"\n  - **Trade Date**: {trade_date}"
            
            formatted_items.append(bullet_item)
        
        return {
            "format": "list",
            "items": formatted_items,
            "total_count": len(df),
            "as_of": _date.today().isoformat()
        }
    except Exception as e:
        # Fallback to regular table format
        cols = [str(c) for c in df.columns]
        rows = [[_jsonify_scalar(v) for v in r] for r in df.head(limit).itertuples(index=False, name=None)]
        from datetime import date as _date
        return {"columns": cols, "rows": rows, "as_of": _date.today().isoformat()}


# ===================== TINY INTENT ROUTER =====================
# Intent constants (private to this file)
INTENT_METRIC_AGG = "metric_agg"
INTENT_LOOKUP_PREVIEW = "lookup_preview"
INTENT_REFERENCE_RAG = "reference_rag"
INTENT_EXPLAIN = "explain"


def _classify_user_intent(user_input: str, data_dict: Optional[dict]) -> str:
    q = (user_input or "").strip().lower()
    if not q:
        return INTENT_EXPLAIN

    # Keyword buckets
    agg_kw = [
        "how many", "count", "total", "sum", "average", "avg", "min", "max",
        "top", "by ", "group", "rank", "distribution", "breakdown",
        "exposure", "past due", "overdue", "unsettled", "most common", "most frequent",
        "notional", "value",
    ]
    lookup_kw = [
        "list", "show", "which", "unique", "distinct", "tickers", "counterparties",
        "preview", "sample", "first 10",
    ]
    rag_kw = [
        "what is", "explain", "definition", "documentation", "policy", "rule", "how does",
        "where is", "help", "guide", "holiday schedule", "validation",
    ]

    def _has_any(terms: list[str]) -> bool:
        return any(t in q for t in terms)

    # Base classification
    if _has_any(agg_kw):
        base = INTENT_METRIC_AGG
    elif _has_any(lookup_kw):
        base = INTENT_LOOKUP_PREVIEW
    elif _has_any(rag_kw):
        base = INTENT_REFERENCE_RAG
    else:
        base = INTENT_EXPLAIN

    # Optional alias boost: if alias present + aggregation term, prefer Metric/Agg
    try:
        aliases = (data_dict or {}).get("aliases") or {}
        alias_hit = False
        for _canon, syns in aliases.items():
            for s in syns:
                if s and s.lower() in q:
                    alias_hit = True
                    break
            if alias_hit:
                break
        if alias_hit and _has_any(agg_kw):
            return INTENT_METRIC_AGG
    except Exception:
        pass

    return base


def route_query(user_input: str) -> str:
    data_dict = None
    try:
        data_dict = build_data_dictionary()
    except Exception:
        data_dict = None
    return _classify_user_intent(user_input, data_dict)


def show_ai_assistant_page():
    st.title("AI Assistant")
    st.caption("🚀 Powered by OpenAI")

    _ensure_session_state()

    # Sidebar controls: Context mode (dropdown) and stats
    with st.sidebar:
        # Connectivity test button (moved here from main area)
        if st.button("Test OpenAI connectivity", help="Run a quick call to verify your OpenAI API key and model access."):
            try:
                client = get_openai_client(api_key=resolve_openai_key())
            except Exception as e:
                st.error(f"Client init failed: {e}")
            else:
                ok_embed = False
                ok_chat = False
                with st.status("Testing embeddings…", expanded=False) as s1:
                    try:
                        r = client.embeddings.create(model=EMBED_MODEL, input=["ping"])
                        vec_len = len(r.data[0].embedding) if r and r.data else 0
                        s1.update(label=f"Embeddings OK ({EMBED_MODEL}, dim={vec_len})", state="complete")
                        ok_embed = True
                    except Exception as e:
                        s1.update(label=f"Embeddings failed: {e}", state="error")
                with st.status("Testing chat completion…", expanded=False) as s2:
                    try:
                        r = client.chat.completions.create(
                            model=AI_MODEL_DEFAULT,
                            messages=[
                                {"role": "system", "content": "Reply with the single word: pong"},
                                {"role": "user", "content": "ping"},
                            ],
                            temperature=0.0,
                            stream=False,
                        )
                        msg = (r.choices[0].message.content or "").strip() if r and r.choices else ""
                        s2.update(label=f"Chat OK ({AI_MODEL_DEFAULT}) — reply: {msg[:60]}", state="complete")
                        ok_chat = True
                    except Exception as e:
                        s2.update(label=f"Chat failed: {e}", state="error")
                if ok_embed or ok_chat:
                    st.success("OpenAI connectivity looks good.")
                else:
                    st.warning("OpenAI calls failed. Check OPENAI_API_KEY, network, or model access. You can set AI_MODEL or EMBED_MODEL via environment variables if needed.")

        default_mode = st.session_state.get("context_mode", "Auto") or "Auto"
        if default_mode == "Auto (recommended)":
            default_mode = "Auto"
        context_mode = st.selectbox(
            "Context mode",
            options=["Off", "Auto", "On"],
            index=["Off", "Auto", "On"].index(default_mode) if default_mode in ["Off", "Auto", "On"] else 1,
            help="Auto = only retrieve when likely helpful based on your question and repo similarity. Saves tokens and time.",
        )
        st.session_state["context_mode"] = context_mode
        stats = st.session_state.get("__ai_last_stats", {})
        st.caption(
            f"Latency: {float(stats.get('latency', 0.0)):.2f}s | Prompt tokens≈ {int(stats.get('prompt_tokens', 0))} | Output tokens≈ {int(stats.get('output_tokens', 0))}"
        )
        # System prompt selector (moved to sidebar, below stats)
        sys_mode = st.selectbox(
            "System prompt",
            options=["Basic", "Repo-aware"],
            index=1,
            help=(
                "Defines how the assistant frames its behavior before each reply.\n"
                "• Basic — general assistant; never injects repo context.\n"
                "• Repo-aware — if Context mode is Auto/On and relevant matches are found, inject retrieved repo snippets and “live repo facts” into the prompt so answers can cite and reason over your repo."
            ),
        )

        if st.button("Clear Chat"):
            st.session_state["ai_msgs"] = []
            st.rerun()

    # Controls (main area) — no system prompt selector here anymore
    # Center chat content with side padding
    _pad_left, chat_col, _pad_right = st.columns([1, 3, 1])

    # Local helper to submit a query via the same flow (chips or typed)
    def _submit_query(q: str, target_container=None) -> None:
        q = (q or "").strip()
        if not q:
            return
        st.session_state["ai_msgs"].append({"role": "user", "content": q})

        rag_context: List[Dict[str, str]] = []
        # Compute lightweight live facts for this question
        try:
            st.session_state["__ai_tool_facts"] = _repo_insights_for(q)
        except Exception:
            st.session_state["__ai_tool_facts"] = ""

        # Decide retrieval based on context mode
        try:
            if context_mode == "On":
                do_retrieve = True
            elif context_mode == "Off":
                do_retrieve = False
            else:
                do_retrieve = heuristic_decision(q)
        except Exception:
            do_retrieve = False

        if do_retrieve:
            try:
                with st.spinner("Retrieving context…"):
                    rag_context = _retrieve(q, k=min(TOP_K, 5))
                rag_context = _dedupe_by_source(rag_context)
                rag_context = _cap_context_tokens(rag_context, MAX_RAG_TOKENS)
            except Exception as e:
                st.warning(f"RAG retrieval failed: {e}")
                rag_context = []

        # Inject only when Repo-aware and we actually have hits
        system_prompt = _build_system_prompt(
            sys_mode,
            rag_context if (sys_mode == "Repo-aware" and do_retrieve and rag_context) else None,
        )

        reply, latency = run_openai_chat(st.session_state["ai_msgs"], system_prompt, stream=True, container=target_container)
        if reply:
            st.session_state["ai_msgs"].append({"role": "assistant", "content": reply})
            # Footer stats
            p_tok, o_tok = _estimate_tokens(st.session_state["ai_msgs"], reply)
            st.session_state["__ai_last_stats"] = {"latency": latency, "prompt_tokens": p_tok, "output_tokens": o_tok}
        else:
            st.session_state["ai_msgs"].append({"role": "assistant", "content": "(no response)"})


    # Render history (centered)
    with chat_col:
        _msgs = st.session_state["ai_msgs"]
        if not _msgs:
            st.markdown("<div style='height: 26vh'></div>", unsafe_allow_html=True)
        for m in _msgs:
            with st.chat_message(m.get("role", "assistant")):
                st.markdown(m.get("content", ""))

        # Suggested prompt chips (only show when there is no prior message)
        if not _msgs:
            st.caption("Try:")
            with st.container():
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button("Provide a list of all exceptions.", use_container_width=True):
                        st.session_state["__draft_query"] = "Provide a list of all exceptions."
                        st.rerun()
                with c2:
                    if st.button("What are the most common exceptions?", use_container_width=True):
                        st.session_state["__draft_query"] = "What are the most common exceptions?"
                        st.rerun()
                with c3:
                    if st.button("List counterparties with unsettled exposure as of today. Group by counterparty, and sort by amount descending.", use_container_width=True):
                        st.session_state["__draft_query"] = "List counterparties with unsettled exposure as of today. Group by counterparty, and sort by amount descending."
                        st.rerun()

    # Chat input (centered) and fixed response area immediately below
    with chat_col:
        if st.session_state.get("ai_msgs"):
            # Keep spacer above the input so it doesn't tether the response area to chips
            st.markdown("<div style='height: 16vh'></div>", unsafe_allow_html=True)
        # One-time draft injection for prefill from chips
        prefill = st.session_state.pop("__draft_query", None)
        if prefill is not None:
            st.session_state["__chat_input_value"] = prefill
        user_input = st.chat_input("Ask something…", key="__chat_input_value")
        response_area = st.container()
        # Centralized submit: only when user presses Enter in chat input
        if user_input:
            _submit_query(user_input, target_container=response_area)
            # Rerun to refresh history above and keep layout stable
            st.rerun()

    # Stats now shown in sidebar; no main-area convenience actions

    # Debug / Connectivity moved to sidebar

    # Sources for current context (centered; UI unchanged)
    with chat_col:
        if st.session_state.get("ai_msgs"):
            # Peek last user input to recompute sources on screen only (non-blocking)
            last_q = next((m.get("content", "") for m in reversed(st.session_state["ai_msgs"]) if m.get("role") == "user"), "")
            if last_q and st.checkbox("Show retrieved sources", value=False, key="__ai_show_sources"):
                try:
                    with st.spinner("Retrieving sources…"):
                        src_ctx = _retrieve(last_q, k=min(TOP_K, 5))
                    src_ctx = _dedupe_by_source(src_ctx)
                    src_ctx = _cap_context_tokens(src_ctx, MAX_RAG_TOKENS)
                    if src_ctx:
                        st.markdown("Sources (top-k):")
                        for c in src_ctx:
                            st.markdown(f"- {c.get('source','')} (rank {c.get('rank','')})")
                    else:
                        st.caption("No sources found.")
                except Exception:
                    st.caption("Sources unavailable.")


# Expose only the requested symbol
__all__ = ["show_ai_assistant_page"]


