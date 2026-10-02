from __future__ import annotations

import time
import inspect
import importlib
import torch
import streamlit as st

# Optimize PyTorch CPU performance using available logical cores
try:
    torch.set_num_threads(4)
except Exception:
    pass

from model.blip_model import BlipVQAModel
from model.clip_model import ClipVQAModel
from model.object_detector import ObjectDetector
from model.ocr_model import OCRModel
from utils.inference import answer_question, analyze_question, build_candidate_answers
from utils.preprocess import load_image


st.set_page_config(
    page_title="OmniSight AI | Visual QA & Object Grounding",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom High-Contrast Modern Cyber-Dark Aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
                    radial-gradient(circle at 90% 90%, rgba(236, 72, 153, 0.08) 0%, transparent 45%),
                    #0A0E1A !important;
        color: #F8FAFC !important;
    }

    /* CRITICAL HIGH CONTRAST TYPOGRAPHY OVERRIDES */
    label, [data-testid="stWidgetLabel"], .stWidgetLabel, .stWidgetLabel p, .stWidgetLabel span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
    }

    p, span, div {
        color: #E2E8F0;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Radio button options - crisp white */
    .stRadio label, [data-testid="stRadio"] label, [data-testid="stRadio"] span, [data-testid="stRadio"] p {
        color: #F1F5F9 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }

    /* Hero Banner */
    .omni-hero {
        position: relative;
        padding: 2.2rem 2.5rem;
        border-radius: 1.5rem;
        background: linear-gradient(135deg, rgba(26, 34, 56, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        margin-bottom: 2rem;
        overflow: hidden;
    }
    .omni-hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899, #06B6D4);
    }
    .hero-badge-wrap {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.5);
        color: #C7D2FE !important;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0;
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #FFFFFF 40%, #E2E8F0 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #CBD5E1 !important;
        font-size: 1.05rem;
        margin-top: 0.6rem;
        max-width: 800px;
        line-height: 1.55;
    }

    /* Glass Cards */
    .glass-card {
        padding: 1.6rem;
        border-radius: 1.35rem;
        background: rgba(20, 27, 45, 0.85);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.25rem;
    }
    .card-heading {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 1.1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.6rem;
    }

    /* File Uploader Customization - Clean dark dropzone */
    [data-testid="stFileUploader"] {
        margin-bottom: 1rem;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1.5px dashed rgba(99, 102, 241, 0.45) !important;
        border-radius: 1rem !important;
        padding: 1.5rem !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #818CF8 !important;
        background-color: rgba(30, 41, 59, 0.9) !important;
    }
    [data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] div {
        color: #CBD5E1 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: rgba(99, 102, 241, 0.25) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(99, 102, 241, 0.6) !important;
        font-weight: 600 !important;
    }

    /* Text Inputs */
    .stTextInput input, .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 0.85rem !important;
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #818CF8 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.35) !important;
    }

    /* Alert / Info Boxes - High Contrast */
    .stAlert, [data-testid="stAlert"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 1rem !important;
        padding: 1rem 1.25rem !important;
    }
    .stAlert p, [data-testid="stAlert"] p {
        color: #F8FAFC !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }

    /* Metric Stat Pills */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    .stat-pill {
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
    }
    .stat-label {
        color: #94A3B8 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin-top: 0.25rem;
    }

    /* Answer Banner */
    .answer-banner {
        padding: 1.5rem 1.75rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15);
        margin-bottom: 1.25rem;
    }
    .answer-tag {
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #34D399 !important;
        margin-bottom: 0.35rem;
    }
    .answer-text {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.25;
    }

    /* Suggestion Chips */
    .chip-container div[data-testid="column"] button {
        background: rgba(30, 41, 59, 0.85) !important;
        color: #F1F5F9 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border-radius: 0.75rem !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        padding: 0.5rem 0.75rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    .chip-container div[data-testid="column"] button:hover {
        background: rgba(99, 102, 241, 0.3) !important;
        border-color: #818CF8 !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
    }

    /* Main CTA button */
    .main-cta-wrap div.stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 50%, #4338CA 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        padding: 0.85rem 1.75rem !important;
        border-radius: 0.85rem !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.45) !important;
        width: 100% !important;
        transition: all 0.25s ease !important;
    }
    .main-cta-wrap div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.65) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }

    /* Meta tags */
    .query-meta-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.4rem 0.85rem;
        border-radius: 0.6rem;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.85rem;
        color: #E2E8F0 !important;
        margin-right: 0.5rem;
        margin-top: 0.4rem;
    }
    .query-meta-chip strong {
        color: #FFFFFF !important;
    }

    /* Expander header */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.5) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 0.75rem !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border-radius: 0.6rem 0.6rem 0 0 !important;
        color: #CBD5E1 !important;
        padding: 0.5rem 1rem !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.25) !important;
        color: #FFFFFF !important;
        border-bottom: 2px solid #818CF8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_clip_model(backend: str) -> ClipVQAModel:
    return ClipVQAModel(backend=backend)


@st.cache_resource
def get_blip_model() -> BlipVQAModel:
    return BlipVQAModel()


@st.cache_resource(show_spinner=False)
def get_detector(version: str = "v8_perfect_grounding") -> ObjectDetector:
    import importlib
    import model.object_detector
    importlib.reload(model.object_detector)
    return model.object_detector.ObjectDetector()


@st.cache_resource
def get_ocr_model() -> OCRModel:
    return OCRModel()


class LazyProxy:
    """Delays loading heavy models until they are actually invoked by the pipeline."""

    def __init__(self, loader) -> None:
        self._loader = loader
        self._instance = None

    def __getattr__(self, name: str):
        if self._instance is None:
            self._instance = self._loader()
        return getattr(self._instance, name)


def main() -> None:
    # Sidebar with model insights and branding alternatives
    with st.sidebar:
        st.markdown("### 👁️ OmniSight AI Engine")
        st.caption("Advanced Visual Question Answering with Dual-Stage Spatial Object Grounding.")
        st.markdown("---")

        st.markdown("**🧠 Active Architecture**")
        st.markdown(
            """
            - **VQA Reasoning**: BLIP-VQA (Quantized INT8)
            - **Matching & Similarity**: OpenAI CLIP-ViT-B/32
            - **Object Localization**: MobileNet-V3 FPN (<0.5s)
            - **Landscape Grounding**: CLIP Spatial Region Localizer
            - **OCR Engine**: EasyOCR Multi-scale
            """
        )

        st.markdown("---")
        st.markdown("**🏷️ Suggested Project Names**")
        st.markdown(
            """
            1. **OmniSight AI** *(Current Flagship)*
            2. **SpectraVision QA**
            3. **IrisIQ**
            4. **VisionPulse AI**
            5. **AcuityAI**
            """
        )

        st.markdown("---")
        if st.button("🧹 Clear Model Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared! Reloading fresh instances.")

    # Hero Header
    st.markdown(
        """
        <div class="omni-hero">
            <div class="hero-badge-wrap">
                <span class="hero-badge">✦ OmniSight AI</span>
                <span class="hero-badge"><span class="pulse-dot"></span> Neural Engine v5.0 Active</span>
            </div>
            <h1 class="hero-title">Visual Intelligence & Object Grounding</h1>
            <p class="hero-subtitle">
                Upload any image and ask open-ended questions. OmniSight reasons over visual content, answers accurately, and pinpoints queried objects with high-precision red spotlighting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.05, 1.25], gap="large")

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">📸 Visual Input & Question</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Target Image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Supports high-resolution images, automatically optimized for CPU processing.",
        )

        # Quick Suggestion Chips
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #CBD5E1; margin-bottom: 0.4rem;'>💡 Quick Suggestion Chips (click to fill question):</p>", unsafe_allow_html=True)
        st.markdown('<div class="chip-container">', unsafe_allow_html=True)
        chip_cols = st.columns(3)
        with chip_cols[0]:
            if st.button("🏔️ Mountain?", use_container_width=True):
                st.session_state["preset_q"] = "is there any mountain in this image ?"
        with chip_cols[1]:
            if st.button("👤 Person?", use_container_width=True):
                st.session_state["preset_q"] = "is there any person in image ?"
        with chip_cols[2]:
            if st.button("🐆 Wildlife?", use_container_width=True):
                st.session_state["preset_q"] = "is there any leopard or animal in this image ?"
        st.markdown('</div>', unsafe_allow_html=True)

        default_question = st.session_state.get("preset_q", "")
        question = st.text_input(
            "Ask a question about the image",
            value=default_question,
            placeholder="e.g. Is there any person in this image? or What is the main object?",
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="card-heading">⚙️ Engine Configuration</div>', unsafe_allow_html=True)

        mode = st.radio(
            "Reasoning Engine Mode",
            ["BLIP generation (Recommended)", "CLIP ranking (Multi-Choice)"],
            horizontal=True,
        )

        is_blip = "BLIP" in mode

        with st.expander("🛠️ Advanced Settings", expanded=False):
            backend = st.radio(
                "CLIP Backend",
                ["PyTorch (Quantized INT8)", "ONNX Runtime"],
                horizontal=True,
                disabled=is_blip,
            )
            custom_candidates = st.text_area(
                "Custom Candidate Answers (one per line)",
                placeholder="red\nblue\ngreen",
                height=100,
                disabled=is_blip,
            )
            top_k = st.slider(
                "Top-K Ranked Answers",
                min_value=1,
                max_value=10,
                value=5,
                disabled=is_blip,
            )

        st.markdown('<div class="main-cta-wrap">', unsafe_allow_html=True)
        run = st.button("🚀 Analyze & Ground Objects", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">🎯 Visual Inspection & Results</div>', unsafe_allow_html=True)

        if not run:
            if uploaded_file is not None:
                image = load_image(uploaded_file)
                st.image(image, caption="Ready for analysis", use_container_width=True)
            else:
                st.info("👈 Upload an image and type a question to launch visual reasoning.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if uploaded_file is None:
            st.warning("Please upload an image first.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if not question.strip():
            st.warning("Please type a question about the image.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        analysis = analyze_question(question)

        progress = st.progress(0, text="Preprocessing image...")
        image = load_image(uploaded_file)
        progress.progress(20, text="Analyzing question semantics...")

        start_time = time.time()

        if is_blip:
            progress.progress(40, text="Engaging BLIP Neural Reasoning...")
            model = get_blip_model()
            if not model.available:
                progress.progress(100, text="Error")
                st.error(model.load_error or "BLIP could not be loaded.")
                st.markdown("</div>", unsafe_allow_html=True)
                return
            progress.progress(70, text="Synthesizing natural language answer...")
            answer = model.answer(image=image, question=question)
            final_answer_text = answer.answer
            confidence_val = answer.confidence
            ranked_display = [{"answer": answer.answer, "score": answer.confidence}]
            explanation = None
        else:
            candidates = [line.strip() for line in custom_candidates.splitlines() if line.strip()]
            if not candidates:
                candidates = analysis.candidates or build_candidate_answers(question)

            chosen_backend = "onnx" if "ONNX" in backend else "torch"
            progress.progress(40, text="Engaging CLIP Multimodal Matcher...")
            clip_model = get_clip_model(chosen_backend)
            if not clip_model.available:
                st.warning(clip_model.load_error or "CLIP could not be loaded.")

            detector = LazyProxy(get_detector)
            ocr_model = LazyProxy(get_ocr_model)
            blip_model = LazyProxy(get_blip_model)

            progress.progress(70, text="Ranking candidate answers...")
            result = answer_question(
                clip_model=clip_model,
                blip_model=blip_model,
                detector=detector,
                ocr_model=ocr_model,
                image=image,
                question=question,
                candidates=candidates,
                top_k=top_k,
            )
            final_answer_text = result.answer
            confidence_val = result.confidence
            ranked_display = result.ranked_answers
            explanation = result.explanation

        elapsed = time.time() - start_time
        progress.progress(100, text=f"Finished in {elapsed:.2f}s")

        # Answer Banner
        st.markdown(
            f"""
            <div class="answer-banner">
                <div class="answer-tag">✦ Predicted Answer</div>
                <div class="answer-text">{final_answer_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Performance & Confidence Stat Cards
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="stat-pill">
                    <div class="stat-label">Model Confidence</div>
                    <div class="stat-value">{confidence_val:.1%}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-label">Response Time</div>
                    <div class="stat-value">{elapsed:.2f}s</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Query Metadata Pills
        st.markdown(
            f"""
            <div style="margin-bottom: 1.25rem;">
                <span class="query-meta-chip">🏷️ Intent: <strong>{analysis.intent.replace('_', ' ').capitalize()}</strong></span>
                <span class="query-meta-chip">⚙️ Engine: <strong>{'BLIP-VQA' if is_blip else 'CLIP-ZeroShot'}</strong></span>
                <span class="query-meta-chip">⚡ Speed: <strong>INT8 Quantized</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Visual Grounding & Red Mark Localization ---
        st.markdown('<div class="card-heading">📍 Object Localization (Red Spotlight)</div>', unsafe_allow_html=True)

        is_negative_vqa = final_answer_text.strip().lower() in {
            "no", "no.", "none", "nothing", "false", "there is no", "not visible", "not present", "zero"
        } or final_answer_text.strip().lower().startswith("no,") or final_answer_text.strip().lower().startswith("no ")

        detector_inst = get_detector(version="v8_perfect_grounding")
        target_obj = detector_inst.extract_target_from_question(question, answer=final_answer_text)

        tab_spotlight, tab_original, tab_details = st.tabs(["🎯 Object Spotlight", "🖼️ Original Image", "📊 Deep Reasoning"])

        with tab_spotlight:
            if is_negative_vqa:
                if target_obj:
                    st.info(f"🔍 **Confirmed Absent**: The model verified there is **no {target_obj}** in this image (VQA Answer: *{final_answer_text}*). No false red marks generated.")
                else:
                    st.info(f"🔍 **Confirmed Absent**: Negative visual presence confirmed by reasoning engine (Answer: *{final_answer_text}*).")
            else:
                with st.spinner("Pinpointing queried concept coordinates..."):
                    import inspect
                    sig = inspect.signature(detector_inst.locate_and_mark)
                    if "clip_model" not in sig.parameters:
                        st.cache_resource.clear()
                        import model.object_detector
                        importlib.reload(model.object_detector)
                        detector_inst = model.object_detector.ObjectDetector()

                    clip_inst = get_clip_model("torch")
                    marked_img, detections, target_obj = detector_inst.locate_and_mark(
                        image, question, clip_model=clip_inst, answer=final_answer_text
                    )

                if marked_img is not None and detections:
                    st.image(
                        marked_img,
                        caption=f"🎯 Located {len(detections)} {target_obj or 'concept'}(s) with red spotlight",
                        use_container_width=True,
                    )
                    det_summary = ", ".join(f"**{d.label.capitalize()}** ({d.score:.0%})" for d in detections)
                    st.success(f"Pinpointed: {det_summary}")
                elif target_obj:
                    st.info(f"🔎 Target **'{target_obj}'** was not detected in this image with sufficient confidence.")
                else:
                    st.caption("No specific physical object was identified from the question to spotlight.")

        with tab_original:
            st.image(image, caption="Uploaded input image", use_container_width=True)

        with tab_details:
            if explanation:
                st.caption(f"**Reasoning Strategy**: {explanation}")

            st.markdown("<p style='font-weight:700; color:#FFFFFF;'>Ranked Candidate Probabilities:</p>", unsafe_allow_html=True)
            for item in ranked_display:
                score_pct = item['score'] * 100
                st.markdown(f"- **{item['answer']}** — `{score_pct:.1f}%`")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
