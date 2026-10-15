from __future__ import annotations
import hashlib
import json
import sys
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))

from config import GROQ_API_KEY, TAVILY_API_KEY, PLANT_MODEL_PATH, CHAT_DB_PATH
from chat_store import ChatStore
from rag import AyurVanaRAG
from web_search import web_search_configured, search_web, summarize_web_results
from plant_disease.predicator import model_exists, predict_image

st.set_page_config(page_title="AyurVana AI", page_icon="🌿", layout="wide")
st.markdown(
    """<style>
    .main-title {font-size:2.2rem;font-weight:750;letter-spacing:-0.03em;}
    .muted {color:#68736b;font-size:0.92rem;}
    div[data-testid="stMetric"] {background:rgba(100,130,100,0.07);padding:12px;border-radius:12px;}
    </style>""",
    unsafe_allow_html=True,
)

store = ChatStore(CHAT_DB_PATH)

# A random owner token scopes SQLite history to this browser's conversation workspace.
# It is stored in the URL to let refreshes return to the same workspace; sharing the URL shares access.
owner_id = st.query_params.get("workspace")
if not owner_id or len(owner_id) < 24:
    owner_id = uuid.uuid4().hex
    st.query_params["workspace"] = owner_id
st.session_state["owner_id"] = owner_id

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None


def get_rag():
    return AyurVanaRAG()


def cached_rag():
    # Cached resource is safe once the index is built; restart/clear cache after rebuilding it.
    return _get_rag_cached()


@st.cache_resource(show_spinner="Loading book index and embedding model…")
def _get_rag_cached():
    return get_rag()


def display_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander("Sources and retrieved evidence", expanded=False):
        book_i = 0
        web_i = 0
        for source in sources:
            if source.get("kind") == "web":
                web_i += 1
                st.markdown(f"**Web source {web_i}: [{source.get('title', 'Source')}]({source.get('url', '')})**")
                st.write(source.get("text", "")[:1600])
            else:
                book_i += 1
                st.markdown(
                    f"**Book source {book_i} — PDF page {source.get('page', '?')}, "
                    f"chunk {source.get('chunk', '?')}**"
                )
                if source.get("distance") is not None:
                    st.caption(f"Vector distance: {float(source['distance']):.4f} (lower is closer, not an accuracy score)")
                st.write(source.get("text", ""))


def new_chat():
    sid = store.create_session(owner_id)
    st.session_state.active_session_id = sid
    st.session_state.session_picker = sid


def render_chatbot():
    st.markdown('<div class="main-title">🌿 AyurVana Book Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Answers start with Aṣṭāṅga Saṃgraha. Web research is added only when the retrieved book passages do not sufficiently answer the question.</div>', unsafe_allow_html=True)

    sessions = store.list_sessions(owner_id)
    if not sessions:
        new_chat()
        sessions = store.list_sessions(owner_id)
    valid_ids = [s["session_id"] for s in sessions]
    if st.session_state.active_session_id not in valid_ids:
        st.session_state.active_session_id = valid_ids[0]
    with st.sidebar:
        st.subheader("Chat sessions")
        if st.button("＋ New conversation", use_container_width=True):
            new_chat()
            st.rerun()
        sessions = store.list_sessions(owner_id)
        valid_ids = [s["session_id"] for s in sessions]
        labels = {s["session_id"]: f"{s['title']}  ·  {s['updated_at'][:10]}" for s in sessions}
        if st.session_state.get("session_picker") not in valid_ids:
            st.session_state.session_picker = st.session_state.active_session_id
        selected = st.selectbox(
            "Saved conversations",
            options=valid_ids,
            format_func=lambda sid: labels.get(sid, "Conversation"),
            key="session_picker",
        )
        st.session_state.active_session_id = selected
        use_web = st.toggle("Use web fallback when book is insufficient", value=True, key="use_web_fallback")
        if use_web and not web_search_configured():
            st.warning("Web fallback is enabled but TAVILY_API_KEY is not configured. Book-only answers still work.")
        if not GROQ_API_KEY:
            st.error("Missing GROQ_API_KEY. Add it to .env or Streamlit Secrets.")

    sid = st.session_state.active_session_id
    history = store.get_messages(sid, owner_id)
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                display_sources(message["sources"])

    prompt = st.chat_input("Ask about Doshas, herbs, formulations, or another related topic…")
    if prompt:
        prior_history = [{"role": m["role"], "content": m["content"]} for m in history[-8:]]
        store.add_message(sid, owner_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Checking the book first…"):
                try:
                    rag = cached_rag()
                    answer, sources = rag.answer(
                        prompt,
                        history=prior_history,
                        allow_web_fallback=bool(use_web),
                    )
                except Exception as exc:
                    answer = f"I could not complete the request: `{exc}`"
                    sources = []
            st.markdown(answer)
            display_sources(sources)
        store.add_message(sid, owner_id, "assistant", answer, sources)

    history = store.get_messages(sid, owner_id)
    if history:
        export = {
            "session_id": sid,
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "messages": [
                {"role": m["role"], "content": m["content"], "created_at": m["created_at"], "sources": m.get("sources", [])}
                for m in history
            ],
        }
        st.sidebar.download_button(
            "Download chat history (JSON)",
            data=json.dumps(export, ensure_ascii=False, indent=2),
            file_name=f"ayurvana_chat_{sid[:8]}.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.sidebar.button("Delete this conversation", use_container_width=True):
            store.delete_session(sid, owner_id)
            st.session_state.active_session_id = None
            st.rerun()


def render_scanner():
    st.markdown('<div class="main-title">🌱 Plant Disease Scanner</div>', unsafe_allow_html=True)
    st.markdown("Upload a clear leaf photo. The model classifies only the categories present in the dataset used for training.")
    if not model_exists():
        st.warning(f"No trained classifier was found at `{PLANT_MODEL_PATH}`. Download the Kaggle dataset and train the model first.")
        with st.expander("Training setup", expanded=True):
            st.code(
                'pip install -r requirements-train.txt\n'
                'python plant_disease/train.py --data-dir "data/plant_disease_dataset" --epochs 10 --batch-size 32',
                language="powershell",
            )
            st.markdown("Dataset: [Plant Diseases (6:2:2) on Kaggle](https://www.kaggle.com/datasets/daoliu/plant-622)")
        return

    uploaded = st.file_uploader("Upload a crop/leaf image", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        st.info("Use a well-lit, in-focus photo showing the affected leaf. Try to avoid distant whole-field images.")
    else:
        raw = uploaded.getvalue()
        image_hash = hashlib.sha256(raw).hexdigest()
        image = Image.open(BytesIO(raw)).convert("RGB")
        col_img, col_info = st.columns([1, 1.25])
        with col_img:
            st.image(image, caption=uploaded.name, use_container_width=True)
        with col_info:
            st.write("**Before analysis**")
            st.write("The result is a model prediction, not a confirmed diagnosis. The model cannot reliably assess disease severity unless a separate severity model is trained.")
            if st.button("Analyze plant image", type="primary", use_container_width=True):
                try:
                    with st.spinner("Analyzing leaf image…"):
                        prediction = predict_image(image, top_k=3)
                    st.session_state.current_plant_result = {
                        "image_hash": image_hash,
                        "filename": uploaded.name,
                        "prediction": prediction,
                    }
                    store.save_scan(owner_id, uploaded.name, prediction)
                    st.session_state.pop("plant_guidance", None)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        result = st.session_state.get("current_plant_result")
        if result and result.get("image_hash") == image_hash:
            prediction = result["prediction"]
            st.markdown("---")
            top = prediction["top_predictions"][0]
            pred_col, score_col, severity_col = st.columns(3)
            pred_col.metric("Predicted class", top["display_name"])
            score_col.metric("Model score", f"{top['confidence'] * 100:.1f}%")
            severity_col.metric("Severity", "Not assessed")
            if top["confidence"] < 0.55:
                st.warning("Low model score: the prediction is uncertain. Capture a clearer image or seek agricultural-extension confirmation.")
            elif top["confidence"] < 0.75:
                st.info("Moderate model score. Compare the top predictions and verify symptoms with a local expert.")
            if top["is_healthy"]:
                st.success("The model's top class is a healthy-leaf category. This does not rule out all diseases or nutrient issues.")
            else:
                st.error("The model's top class is a disease category. Treat it as a screening result, not a confirmed diagnosis.")
            with st.expander("Top predictions", expanded=True):
                for item in prediction["top_predictions"]:
                    st.write(f"**{item['display_name']}** — {item['confidence'] * 100:.1f}% model score")
            st.caption("Severity is not inferred from class confidence. The dataset provides disease labels, not validated severity labels.")

            if not TAVILY_API_KEY:
                st.info("To show linked treatment/management and prevention information, configure TAVILY_API_KEY. The scanner will not invent a treatment recommendation.")
            else:
                if st.button("Find sourced management & prevention guidance", key=f"guidance_{image_hash[:12]}"):
                    query = (
                        f"{prediction['condition']} on {prediction['crop']} management treatment prevention "
                        "integrated pest management university extension agriculture"
                    )
                    try:
                        with st.spinner("Searching agricultural sources…"):
                            results = search_web(query, max_results=5)
                            if results:
                                guidance = summarize_web_results(
                                    f"What management/treatment and prevention options are supported for {prediction['condition']} on {prediction['crop']}?",
                                    results,
                                    extra_context=(
                                        f"Visual classifier prediction: {prediction['display_name']} "
                                        f"({prediction['confidence']*100:.1f}% model score; not a confirmed diagnosis)."
                                    ),
                                )
                            else:
                                guidance = "No relevant web results were returned. Try another query or confirm the disease with an agricultural extension service."
                        st.session_state.plant_guidance = {"image_hash": image_hash, "answer": guidance, "sources": results}
                    except Exception as exc:
                        st.error(f"Web research failed: {exc}")
                guidance_state = st.session_state.get("plant_guidance")
                if guidance_state and guidance_state.get("image_hash") == image_hash:
                    st.markdown("### Management / treatment and prevention (web-sourced)")
                    st.markdown(guidance_state["answer"])

    scans = store.list_scans(owner_id, limit=15)
    with st.expander("Recent scan history", expanded=False):
        if not scans:
            st.caption("No scans saved in this workspace yet.")
        else:
            for scan in scans:
                st.write(
                    f"**{scan['predicted_label']}** — {scan['confidence']*100:.1f}% model score\n"
                    f"{scan['filename']} · {scan['created_at']}"
                )


def render_dashboard():
    st.markdown('<div class="main-title">📊 AyurVana Dashboard</div>', unsafe_allow_html=True)
    scans = store.list_scans(owner_id, limit=1000)
    sessions = store.list_sessions(owner_id)
    total_messages = sum(len(store.get_messages(s["session_id"], owner_id)) for s in sessions)
    c1, c2, c3 = st.columns(3)
    c1.metric("Saved conversations", len(sessions))
    c2.metric("Chat messages", total_messages)
    c3.metric("Plant scans", len(scans))
    if scans:
        healthy = sum(int(s["is_healthy"]) for s in scans)
        diseased = len(scans) - healthy
        c4, c5 = st.columns(2)
        c4.metric("Top class marked healthy", healthy)
        c5.metric("Top class marked disease", diseased)
        st.subheader("Recent scans")
        st.dataframe(
            [
                {
                    "timestamp": s["created_at"],
                    "image": s["filename"],
                    "prediction": s["predicted_label"],
                    "crop": s["crop"],
                    "condition": s["condition"],
                    "model_score": f"{s['confidence']*100:.1f}%",
                }
                for s in scans
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No plant scans yet. Use Plant Disease Scanner to analyze a leaf image.")
    st.caption("This dashboard is scoped to the current workspace token, not a shared global user database.")


with st.sidebar:
    st.markdown("## AyurVana AI")
    page = st.radio("Go to", ["Book Chatbot", "Plant Disease Scanner", "Dashboard"], key="main_page")
    st.caption("Book-first RAG • web citations • plant disease screening")

if page == "Book Chatbot":
    render_chatbot()
elif page == "Plant Disease Scanner":
    render_scanner()
else:
    render_dashboard()
