"""
app.py
======
Streamlit GUI for the Siamese Signature Verification System.

Run with:
    cd signature-verification
    streamlit run app.py

Features
--------
- Two side-by-side upload boxes (reference + query signature).
- Instant image preview after upload.
- "Verify Signature" button (enabled only when both images are uploaded).
- Result displayed as a coloured box (green = Genuine, red = Forged).
- st.metric for distance and confidence + a progress bar.
- Sidebar with threshold slider, "How it works" explanation, and sample images.
- Error handling for invalid files, missing model, same-file upload, etc.
- Footer disclaimer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

# ── Make src/ importable when running from the project root ─────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config (must be the FIRST Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="Signature Verifier",
    page_icon="✍️",
    layout="wide",
)

# ── Lazy import of inference module (so the app still loads if model is
#    missing — we show a friendly error instead of crashing at import time)
from src.infer import verify, is_model_ready, THRESHOLD_PATH, MODEL_PATH

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
SAMPLE_DIR     = ROOT / "data"
if not (SAMPLE_DIR / "full_org").exists() and (SAMPLE_DIR / "signatures" / "full_org").exists():
    GENUINE_DIR = SAMPLE_DIR / "signatures" / "full_org"
    FORGED_DIR  = SAMPLE_DIR / "signatures" / "full_forg"
else:
    GENUINE_DIR = SAMPLE_DIR / "full_org"
    FORGED_DIR  = SAMPLE_DIR / "full_forg"
THRESHOLD_FILE = ROOT / "models" / "threshold.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_default_threshold() -> float:
    """Read the EER threshold saved by evaluate.py."""
    if THRESHOLD_FILE.exists():
        with open(THRESHOLD_FILE) as fh:
            return float(json.load(fh)["threshold"])
    return 0.5   # sensible fallback if evaluate.py hasn't been run yet


def pil_from_upload(uploaded_file) -> Image.Image | None:
    """
    Convert a Streamlit UploadedFile to a PIL Image.
    Returns None and shows an error if the file cannot be decoded.
    """
    try:
        img = Image.open(BytesIO(uploaded_file.read()))
        img.verify()   # check it is a valid image header
        # Re-open after verify() (verify() consumes the stream)
        uploaded_file.seek(0)
        img = Image.open(BytesIO(uploaded_file.read()))
        return img
    except UnidentifiedImageError:
        st.error(
            f"❌ **{uploaded_file.name}** could not be read as an image.  "
            "Please upload a valid PNG, JPG, or JPEG file."
        )
        return None
    except Exception as exc:
        st.error(f"❌ Unexpected error reading **{uploaded_file.name}**: {exc}")
        return None


def get_sample_pairs() -> list[tuple[Path, Path, str]]:
    """
    Return test-writer sample pairs (both genuine pairs and forgery pairs).
    Returns list of (path1, path2, label_display).
    """
    pairs = []
    if not GENUINE_DIR.exists() or not FORGED_DIR.exists():
        return pairs

    ids_file = ROOT / "models" / "test_writer_ids.json"
    if ids_file.exists():
        try:
            with open(ids_file) as fh:
                test_ids = json.load(fh)[:3]
        except Exception:
            test_ids = [2, 7, 9]
    else:
        test_ids = [2, 7, 9]

    for wid in test_ids:
        org1  = GENUINE_DIR / f"original_{wid}_1.png"
        org2  = GENUINE_DIR / f"original_{wid}_2.png"
        forg1 = FORGED_DIR / f"forgeries_{wid}_1.png"

        if org1.exists() and org2.exists():
            pairs.append((org1, org2, f"Writer {wid} - Genuine Pair (Original + Original)"))
        if org1.exists() and forg1.exists():
            pairs.append((org1, forg1, f"Writer {wid} - Forgery Pair (Original + Forgery)"))

    return pairs


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    default_thr = load_default_threshold()

    threshold = st.slider(
        "Decision threshold",
        min_value = 0.01,
        max_value = 2.0,
        value     = default_thr,
        step      = 0.01,
        help      = (
            "Distance below this value → Genuine, above → Forged.  "
            "Increase to make the system more lenient (fewer false rejections); "
            "decrease to make it stricter (fewer false acceptances)."
        ),
    )

    

    st.divider()
    st.subheader("🖼️ Try sample images")
    sample_pairs = get_sample_pairs()
    if sample_pairs:
        sample_labels = [p[2] for p in sample_pairs]
        chosen_label  = st.selectbox("Pick a sample pair", sample_labels)
        chosen_idx    = sample_labels.index(chosen_label)
        load_sample   = st.button("Load sample pair")
    else:
        st.info(
            "Sample images not available.  Place the CEDAR dataset in  "
            "`data/full_org/` and `data/full_forg/`."
        )
        load_sample = False
        chosen_idx  = 0


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("✍️ Signature Forgery Detection")
st.markdown(
    "Upload **two signature images** — a known genuine reference and the "
    "signature you want to verify — and the AI will predict whether the second "
    "is likely genuine or a possible forgery."
)

st.divider()

# ── Session state for loaded/uploaded images ────────────────────────────────
if "ref_img" not in st.session_state:
    st.session_state.ref_img = None
if "query_img" not in st.session_state:
    st.session_state.query_img = None
if "sample_info" not in st.session_state:
    st.session_state.sample_info = None

# ── Upload columns ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Reference Signature *(known genuine)*")
    ref_file = st.file_uploader(
        "Upload reference",
        type=["png", "jpg", "jpeg"],
        key="ref_upload",
        label_visibility="collapsed",
    )

with col2:
    st.subheader("🔍 Signature to Verify")
    query_file = st.file_uploader(
        "Upload query",
        type=["png", "jpg", "jpeg"],
        key="query_upload",
        label_visibility="collapsed",
    )

# ── Handle "Load sample pair" from sidebar ──────────────────────────────────
if load_sample and sample_pairs:
    ref_path, query_path, label = sample_pairs[chosen_idx]
    st.session_state.ref_img = Image.open(ref_path)
    st.session_state.query_img = Image.open(query_path)
    st.session_state.sample_info = f"Loaded sample: **{label}**"

# Parse uploaded files (overrides loaded sample)
if ref_file is not None:
    st.session_state.ref_img = pil_from_upload(ref_file)
    st.session_state.sample_info = None

if query_file is not None:
    st.session_state.query_img = pil_from_upload(query_file)
    st.session_state.sample_info = None

if st.session_state.sample_info:
    st.info(st.session_state.sample_info)

ref_img   = st.session_state.ref_img
query_img = st.session_state.query_img

# ── Preview images ──────────────────────────────────────────────────────────
if ref_img is not None or query_img is not None:
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        if ref_img is not None:
            st.image(ref_img, caption="Reference Signature", use_container_width=True)
    with preview_col2:
        if query_img is not None:
            st.image(query_img, caption="Signature to Verify", use_container_width=True)

# ── Same-file warning ────────────────────────────────────────────────────────
same_file_warning = False
if (ref_file is not None and query_file is not None
        and ref_file.name == query_file.name):
    st.warning(
        "⚠️ Both uploads have the same filename.  "
        "Please make sure you are uploading two DIFFERENT signature images."
    )
    same_file_warning = True

# ── Verify button ────────────────────────────────────────────────────────────
st.divider()

both_ready   = (ref_img is not None) and (query_img is not None)
button_label = "🔎 Verify Signature"

clicked = st.button(
    button_label,
    disabled=(not both_ready) or same_file_warning,
    type="primary",
    use_container_width=True,
)

if not both_ready:
    st.caption("⬆️ Upload both signature images to enable the verify button.")

# ── Result ───────────────────────────────────────────────────────────────────
if not is_model_ready():
    st.info(
        "💡 **Model checkpoint or threshold not found in `models/`.**  "
        "Train and evaluate the model using:\n\n"
        "```bash\npython -m src.train\npython -m src.evaluate\n```"
    )

if clicked:
    if not is_model_ready():
        st.error(
            "❌ **Model checkpoint or threshold file not found.**  "
            "Please train and evaluate the model first by running:\n\n"
            "```bash\npython -m src.train\npython -m src.evaluate\n```"
        )
    else:
        with st.spinner("Analysing signatures …"):
            try:
                result = verify(
                    ref_img,
                    query_img,
                    threshold_override=threshold,
                )
            except FileNotFoundError as exc:
                st.error(f"❌ {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Unexpected error during verification: {exc}")
                st.stop()

        verdict    = result["verdict"]
        distance   = result["distance"]
        confidence = result["confidence"]

        st.divider()
        st.subheader("📊 Verification Result")

        # Coloured result box
        if verdict == "Genuine":
            st.success(
                f"✅ **Likely GENUINE** — the query signature probably belongs to "
                f"the same writer as the reference."
            )
        else:
            st.error(
                f"🚨 **Possible FORGERY** — the query signature may not belong to "
                f"the same writer as the reference."
            )

        # Metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(
                label="Embedding Distance",
                value=f"{distance:.4f}",
                delta=f"threshold = {threshold:.4f}",
                delta_color="off",
            )
        with metric_col2:
            st.metric(
                label="Confidence",
                value=f"{confidence:.1f} %",
            )

        # Confidence progress bar
        st.progress(
            int(confidence),
            text=f"Confidence: {confidence:.1f} %",
        )

        # Probabilistic note
        st.caption(
            "ℹ️ Results are probabilistic — \"likely\" and \"possible\" are "
            "intentional.  The confidence score reflects the distance from the "
            "decision threshold, not an absolute certainty."
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:grey; font-size:0.85em;'>"
    "📘 This is an academic project and should <strong>not</strong> be used as the "
    "sole basis for legal or financial decisions."
    "</p>",
    unsafe_allow_html=True,
)

