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

# Custom Ultra-Modern Glassmorphism & Cyber Aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global typography and background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(236, 72, 153, 0.06) 0%, transparent 40%),
                    #090D16;
        color: #F1F5F9;
    }

    /* Hero Banner */
    .omni-hero {
        position: relative;
        padding: 2.25rem 2.5rem;
        border-radius: 1.5rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
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
        margin-bottom: 0.6rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #A5B4FC;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
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
        background: linear-gradient(135deg, #FFFFFF 30%, #CBD5E1 70%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 0.6rem;
        max-width: 780px;
        line-height: 1.5;
    }

    /* Cards */
    .glass-card {
        padding: 1.5rem;
        border-radius: 1.25rem;
        background: rgba(18, 24, 41, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.25rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }
    .card-heading {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    .stat-pill {
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
    }
    .stat-label {
        color: #94A3B8;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 0.25rem;
    }

    /* Answer box */
    .answer-banner {
        padding: 1.4rem 1.6rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.1);
        margin-bottom: 1.25rem;
    }
    .answer-tag {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #34D399;
        margin-bottom: 0.3rem;
    }
    .answer-text {
        font-size: 1.75rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.2;
    }

    /* Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 50%, #4338CA 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.75rem 1.5rem;
        border-radius: 0.85rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.35);
        transition: all 0.25s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(99, 102, 241, 0.5);
        border-color: rgba(255, 255, 255, 0.3);
    }

    /* Input customizations */
    .stTextInput input, .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 0.75rem !important;
        color: #F8FAFC !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }

    /* Info chips */
    .query-meta-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.75rem;
        border-radius: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.85rem;
        color: #CBD5E1;
        margin-right: 0.5rem;
        margin-top: 0.4rem;
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
def get_detector(version: str = "v5_omnisight") -> ObjectDetector:
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
        st.caption("💡 Quick Question Suggestions (click to populate):")
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

        run = st.button("🚀 Analyze & Ground Objects", type="primary", use_container_width=True)
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

        detector_inst = get_detector(version="v5_omnisight")
        target_obj = detector_inst.extract_target_from_question(question)

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
                        image, question, clip_model=clip_inst
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

            st.markdown("**Ranked Candidate Probabilities:**")
            for item in ranked_display:
                score_pct = item['score'] * 100
                st.markdown(f"- **{item['answer']}** — `{score_pct:.1f}%`")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
