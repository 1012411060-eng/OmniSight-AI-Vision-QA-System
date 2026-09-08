from __future__ import annotations

import streamlit as st

from model.blip_model import BlipVQAModel
from model.clip_model import ClipVQAModel
from model.object_detector import ObjectDetector
from model.ocr_model import OCRModel
from utils.inference import answer_question, analyze_question, build_candidate_answers
from utils.preprocess import load_image


st.set_page_config(
    page_title="CPU-Friendly VQA Demo",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 60px rgba(15, 23, 42, 0.25);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        line-height: 1.1;
    }
    .hero p {
        margin: 0.45rem 0 0;
        opacity: 0.88;
        font-size: 1rem;
    }
    .card {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        background: rgba(248, 250, 252, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-bottom: 0.75rem;
    }
    .soft {
        padding: 0.85rem 1rem;
        border-radius: 0.9rem;
        background: rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin-bottom: 0.75rem;
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


@st.cache_resource
def get_detector() -> ObjectDetector:
    return ObjectDetector()


@st.cache_resource
def get_ocr_model() -> OCRModel:
    return OCRModel()


def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Image Question Answering</h1>
            <p>CPU-friendly demo with smarter question routing and optional ONNX acceleration.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1.1], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        question = st.text_input("Ask a question about the image", placeholder="Is there a car in the picture?")
        mode = st.radio("Answer mode", ["CLIP ranking", "BLIP generation"], horizontal=True)
        backend = st.radio(
            "CLIP backend",
            ["ONNX Runtime", "PyTorch"],
            horizontal=True,
            disabled=mode == "BLIP generation",
        )
        custom_candidates = st.text_area(
            "Optional candidate answers, one per line",
            placeholder="red\nblue\ngreen",
            height=140,
            disabled=mode == "BLIP generation",
        )
        top_k = st.slider("Show top-k answers", min_value=1, max_value=10, value=5, disabled=mode == "BLIP generation")
        run = st.button("Run VQA", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.subheader("Result")
        if not run:
            st.info("Upload an image and ask a question to see the answer.")
            return

        if uploaded_file is None:
            st.warning("Please upload an image first.")
            return

        if not question.strip():
            st.warning("Please type a question.")
            return

        analysis = analyze_question(question)

        progress = st.progress(0, text="Preparing image...")
        image = load_image(uploaded_file)
        progress.progress(15, text="Rendering image...")
        st.image(image, caption="Input image", use_container_width=True)

        st.markdown(
            f"""
            <div class="soft">
                <strong>Question type:</strong> {analysis.intent.replace('_', ' ')}<br/>
                <strong>Status:</strong> {'supported' if analysis.supported else 'not supported'}<br/>
                <strong>Why:</strong> {analysis.reason}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not analysis.supported:
            progress.progress(100, text="Done")
            st.warning(analysis.reason)
            st.caption("Try asking about visible things like colors, objects, counts, locations, or actions.")
            return

        if mode == "BLIP generation":
            progress.progress(50, text="Loading BLIP model...")
            model = get_blip_model()
            if not model.available:
                progress.progress(100, text="Done")
                st.error(model.load_error or "BLIP could not be loaded.")
                return
            progress.progress(75, text="Generating answer...")
            answer = model.answer(image=image, question=question)
            progress.progress(100, text="Done")
            st.success(f"Answer: {answer.answer}")
            st.metric("Confidence", f"{answer.confidence:.1%}")
            st.caption("BLIP confidence is an approximate generation score, not a calibrated probability.")

            with st.expander("Top answers", expanded=True):
                st.write(f"1. {answer.answer}")
        else:
            candidates = [line.strip() for line in custom_candidates.splitlines() if line.strip()]
            if not candidates:
                candidates = analysis.candidates or build_candidate_answers(question)

            chosen_backend = "onnx" if backend == "ONNX Runtime" else "torch"
            progress.progress(50, text=f"Loading CLIP model ({backend})...")
            clip_model = get_clip_model(chosen_backend)
            if not clip_model.available:
                st.warning(clip_model.load_error or "CLIP could not be loaded.")
            detector = get_detector()
            ocr_model = get_ocr_model()
            blip_model = get_blip_model()
            progress.progress(75, text="Answering the question...")

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
            progress.progress(100, text="Done")

            st.success(f"Answer: {result.answer}")
            st.metric("Confidence", f"{result.confidence:.1%}")

            with st.expander("Top answers", expanded=True):
                for item in result.ranked_answers:
                    st.write(f"{item['answer']}  -  {item['score']:.4f}")

            if result.explanation:
                st.caption(result.explanation)

            if result.strategy == "clip_rank":
                with st.expander("Used candidates"):
                    st.write(analysis.candidates or build_candidate_answers(question))

        st.caption("Tip: CLIP on ONNX Runtime is the faster CPU path. BLIP is used for direct attribute questions like color, yes/no, and count.")


if __name__ == "__main__":
    main()
