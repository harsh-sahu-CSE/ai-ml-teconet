from __future__ import annotations

import os
import time

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from preprocessing import preprocess, PreprocessingResult
from model_pipeline import (
    SentimentPipeline,
    SentimentPrediction,
    ModelConfig,
    build_pipeline,
    load_pipeline,
    predict,
    predict_batch,
)


st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)


CUSTOM_CSS = """
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Global background ── */
  .stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #111827 60%, #0d0f1a 100%);
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(17, 24, 39, 0.95);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
  }

  /* ── Headings ── */
  h1, h2, h3 { color: #f1f5f9 !important; letter-spacing: -0.02em; }

  /* ── Input textarea ── */
  .stTextArea textarea {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    transition: border-color 0.25s ease;
  }
  .stTextArea textarea:focus {
    border-color: rgba(99, 102, 241, 0.75) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.6rem !important;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
    width: 100%;
  }
  .stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button:active { transform: translateY(0px) !important; }

  /* ── Cards ── */
  .card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
  }

  /* ── Sentiment badge ── */
  .badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .badge-positive  { background: rgba(16,185,129,0.18); color: #10b981; border: 1px solid rgba(16,185,129,0.4); }
  .badge-neutral   { background: rgba(245,158,11,0.18);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.4); }
  .badge-negative  { background: rgba(239,68,68,0.18);   color: #ef4444; border: 1px solid rgba(239,68,68,0.4); }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: rgba(30, 41, 59, 0.55) !important;
    border: 1px solid rgba(99, 102, 241, 0.18) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.4) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 12px !important;
  }

  /* ── Divider ── */
  hr { border-color: rgba(99, 102, 241, 0.15) !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 4px; }

  /* ── Mobile responsiveness ── */
  @media (max-width: 768px) {
    /* Shrink hero heading on small screens */
    .hero-title {
      font-size: 1.7rem !important;
    }

    /* Reduce main block padding so content isn't clipped */
    .block-container {
      padding-left: 1rem !important;
      padding-right: 1rem !important;
      max-width: 100% !important;
    }

    /* Reduce card padding */
    .card {
      padding: 1rem !important;
    }

    /* Ensure textarea is always fully visible */
    .stTextArea {
      width: 100% !important;
    }
    .stTextArea textarea {
      width: 100% !important;
      min-height: 120px !important;
      font-size: 16px !important; /* prevents iOS auto-zoom on focus */
    }

    /* Make tables horizontally scrollable */
    [data-testid="stDataFrame"] {
      overflow-x: auto !important;
    }
  }

  @media (max-width: 480px) {
    .hero-title {
      font-size: 1.4rem !important;
    }
    .card {
      padding: 0.75rem !important;
    }
    .block-container {
      padding-left: 0.75rem !important;
      padding-right: 0.75rem !important;
    }
  }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


SENTIMENT_COLORS: dict[str, str] = {
    "Positive": "#10b981",
    "Neutral":  "#f59e0b",
    "Negative": "#ef4444",
}

SENTIMENT_ICONS: dict[str, str] = {
    "Positive": "😊",
    "Neutral":  "😐",
    "Negative": "😞",
}

CHECKPOINT_DIR: str = "checkpoints/best_model"
HF_REPO_ID: str = "SAGAR-SAHU/tweet-sentiment-analyzer"


@st.cache_resource(show_spinner=False)
def _load_model() -> SentimentPipeline:
    if os.path.isdir(CHECKPOINT_DIR):
        return load_pipeline(CHECKPOINT_DIR)

    if HF_REPO_ID:
        from huggingface_hub import snapshot_download

        hf_token: str | None = st.secrets.get("HF_TOKEN", None)
        local_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
        return load_pipeline(local_dir)

    raise FileNotFoundError(
        f"No checkpoint found at '{CHECKPOINT_DIR}' and HF_REPO_ID is not set. "
        "Either run python train.py locally, or set HF_REPO_ID in app.py."
    )


def _try_load_model() -> SentimentPipeline | None:
    try:
        return _load_model()
    except FileNotFoundError as exc:
        return None, str(exc)


def _build_gauge(confidence: float, label: str) -> go.Figure:
    """Build a semi-circular gauge showing prediction confidence."""
    color: str = SENTIMENT_COLORS[label]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "color": "#f1f5f9", "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#475569",
                "tickfont": {"color": "#94a3b8", "size": 11},
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33],  "color": "rgba(239,68,68,0.12)"},
                {"range": [33, 66], "color": "rgba(245,158,11,0.12)"},
                {"range": [66, 100],"color": "rgba(16,185,129,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": confidence * 100,
            },
        },
        title={"text": "Confidence", "font": {"color": "#94a3b8", "size": 13, "family": "Inter"}},
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_proba_bar(probabilities: dict[str, float]) -> go.Figure:
    """Build a horizontal bar chart of all 3 class probabilities."""
    labels: list[str] = list(probabilities.keys())
    values: list[float] = [v * 100 for v in probabilities.values()]
    colors: list[str] = [SENTIMENT_COLORS[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont={"color": "#94a3b8", "size": 13, "family": "Inter"},
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=180,
        margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "range": [0, 115],
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
        },
        yaxis={
            "tickfont": {"color": "#e2e8f0", "size": 13, "family": "Inter"},
            "showgrid": False,
        },
        showlegend=False,
    )
    return fig


def _build_batch_distribution(predictions: list[SentimentPrediction]) -> go.Figure:
    """Donut chart showing label distribution across a batch."""
    counts: dict[str, int] = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for p in predictions:
        counts[p.label] += 1

    labels: list[str] = list(counts.keys())
    values: list[int] = list(counts.values())
    colors: list[str] = [SENTIMENT_COLORS[l] for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker_colors=colors,
        textinfo="label+percent",
        textfont={"color": "#f1f5f9", "size": 13, "family": "Inter"},
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend={"font": {"color": "#94a3b8", "family": "Inter"}},
    )
    return fig


def _build_confidence_histogram(predictions: list[SentimentPrediction]) -> go.Figure:
    """Histogram of confidence scores across the batch."""
    confidences: list[float] = [p.confidence * 100 for p in predictions]
    label_colors: list[str] = [SENTIMENT_COLORS[p.label] for p in predictions]

    fig = px.histogram(
        x=confidences,
        nbins=20,
        color_discrete_sequence=["#6366f1"],
        labels={"x": "Confidence (%)"},
    )
    fig.update_layout(
        height=240,
        margin=dict(t=20, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title_font": {"color": "#94a3b8"}, "tickfont": {"color": "#94a3b8"}, "gridcolor": "rgba(255,255,255,0.06)"},
        yaxis={"title_font": {"color": "#94a3b8"}, "tickfont": {"color": "#94a3b8"}, "gridcolor": "rgba(255,255,255,0.06)"},
        bargap=0.05,
    )
    return fig


def _render_single_result(
    raw_text: str,
    prep: PreprocessingResult,
    pred: SentimentPrediction,
) -> None:
    label: str = pred.label
    icon: str = SENTIMENT_ICONS[label]
    badge_class: str = f"badge-{label.lower()}"

    st.markdown("---")

    col_label, col_meta = st.columns([1, 1])

    with col_label:
        st.markdown(
            f'<div class="card" style="text-align:center;">'
            f'<div style="font-size:3rem; line-height:1;">{icon}</div>'
            f'<div style="margin-top:0.5rem;">'
            f'<span class="badge {badge_class}">{label}</span>'
            f'</div>'
            f'<div style="color:#94a3b8; font-size:0.82rem; margin-top:0.6rem;">Sentiment</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_meta:
        lang_display: str = prep.detected_lang.upper() if prep.detected_lang != "unknown" else "—"
        st.metric("Confidence", f"{pred.confidence:.1%}")
        st.metric("Language", lang_display)
        st.metric("Tokens", str(len(prep.tokens)))

    col_gauge, col_bar = st.columns([1, 1])

    with col_gauge:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(
            _build_gauge(pred.confidence, label),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_bar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#94a3b8; font-size:0.85rem; margin-bottom:0.5rem;">'
            'Probability Distribution</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _build_proba_bar(pred.probabilities),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔍  Preprocessing Detail", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original Text**")
            st.code(prep.original_text, language=None)
            if prep.detected_lang not in ("en", "unknown"):
                st.markdown("**Translated Text**")
                st.code(prep.translated_text, language=None)
        with col_b:
            st.markdown("**Cleaned Text**")
            st.code(prep.cleaned_text, language=None)
            st.markdown("**Final Tokens**")
            st.code(" · ".join(prep.tokens) if prep.tokens else "(empty)", language=None)


def _render_batch_results(
    predictions: list[SentimentPrediction],
    prep_results: list[PreprocessingResult],
    original_texts: list[str],
) -> None:
    """Render aggregate charts and a sortable results table for batch mode."""
    st.markdown("---")
    st.markdown("### 📊 Batch Analysis Results")

    total: int = len(predictions)
    pos: int = sum(1 for p in predictions if p.label == "Positive")
    neu: int = sum(1 for p in predictions if p.label == "Neutral")
    neg: int = sum(1 for p in predictions if p.label == "Negative")
    avg_conf: float = sum(p.confidence for p in predictions) / total

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", str(total))
    c2.metric("😊 Positive", str(pos))
    c3.metric("😐 Neutral",  str(neu))
    c4, c5 = st.columns(2)
    c4.metric("😞 Negative", str(neg))
    c5.metric("Avg Confidence", f"{avg_conf:.1%}")

    col_donut, col_hist = st.columns([1, 1])

    with col_donut:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#94a3b8; font-size:0.85rem; margin-bottom:0.3rem;">Label Distribution</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _build_batch_distribution(predictions),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_hist:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#94a3b8; font-size:0.85rem; margin-bottom:0.3rem;">Confidence Distribution</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _build_confidence_histogram(predictions),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Results Table")

    rows: list[dict] = []
    for i, (orig, prep, pred) in enumerate(
        zip(original_texts, prep_results, predictions), start=1
    ):
        rows.append({
            "#":           i,
            "Original Text": orig[:120] + ("…" if len(orig) > 120 else ""),
            "Language":    prep.detected_lang.upper(),
            "Sentiment":   pred.label,
            "Confidence":  f"{pred.confidence:.1%}",
            "P(Positive)": f"{pred.probabilities['Positive']:.1%}",
            "P(Neutral)":  f"{pred.probabilities['Neutral']:.1%}",
            "P(Negative)": f"{pred.probabilities['Negative']:.1%}",
        })

    df: pd.DataFrame = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sentiment": st.column_config.TextColumn("Sentiment"),
            "Confidence": st.column_config.TextColumn("Confidence"),
        },
    )

    csv_bytes: bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇  Download Results as CSV",
        data=csv_bytes,
        file_name="sentiment_results.csv",
        mime="text/csv",
    )


def _render_sidebar() -> tuple[str, bool, bool]:
    st.sidebar.markdown(
        '<h2 style="color:#a5b4fc; margin-bottom:0.2rem;">⚙️ Settings</h2>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    mode: str = st.sidebar.radio(
        "Analysis Mode",
        options=["Single Text", "Batch (CSV Upload)"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color:#94a3b8; font-size:0.85rem;">Preprocessing Options</p>',
        unsafe_allow_html=True,
    )
    translate_flag: bool = st.sidebar.toggle(
        "Auto-translate to English", value=True
    )
    remove_stops: bool = st.sidebar.toggle(
        "Remove stopwords", value=True
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="color:#64748b; font-size:0.78rem; line-height:1.6;">
        <b style="color:#94a3b8;">Model</b><br>
        distilbert-base-uncased<br>
        3-class · custom head<br><br>
        <b style="color:#94a3b8;">Pipeline</b><br>
        preprocessing.py<br>
        model_pipeline.py
        </div>
        """,
        unsafe_allow_html=True,
    )

    return mode, translate_flag, remove_stops


def _render_header() -> None:
    st.markdown(
        """
        <div style="text-align:center; padding: 1.6rem 0 0.6rem 0;">
          <h1 class="hero-title" style="
            font-size: clamp(1.4rem, 5vw, 2.6rem);
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
          ">🧠 Tweet Sentiment Analyzer</h1>
          <p style="color:#64748b; font-size:clamp(0.8rem, 3vw, 1rem); margin:0;">
            Transformer-powered · BERT · 3-class Sentiment Analysis
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _render_header()
    mode, translate_flag, remove_stops = _render_sidebar()

    with st.spinner("⏳ Loading model — this may take a minute on first run…"):
        model_load_result = _try_load_model()

    if isinstance(model_load_result, tuple):
        pipeline: SentimentPipeline | None = None
        load_error: str = model_load_result[1]
    else:
        pipeline = model_load_result
        load_error = ""

    if pipeline is None:
        st.error(
            f"**Model checkpoint not found.**\n\n"
            f"{load_error}\n\n"
            "**To train the model:**\n"
            "```python\n"
            "from model_pipeline import build_pipeline, ModelConfig, train\n\n"
            "pipe = build_pipeline(ModelConfig())\n"
            "train(pipe, train_texts, train_labels, val_texts, val_labels,\n"
            "      save_dir='checkpoints/best_model')\n"
            "```\n"
            "Then restart the app."
        )
        st.stop()

    if mode == "Single Text":
        st.markdown(
            '<p style="color:#94a3b8; margin-bottom:0.4rem;">Enter text to analyse:</p>',
            unsafe_allow_html=True,
        )
        user_text: str = st.text_area(
            label="Input text",
            placeholder=(
                'Enter Tweet Here!\n'
                'or paste a Hindi review…'
            ),
            height=150,
            label_visibility="collapsed",
        )

        run_btn = st.button("Analyse Sentiment", use_container_width=False)

        if run_btn:
            if not user_text.strip():
                st.warning("Please enter some text before analysing.")
                st.stop()

            with st.spinner("Preprocessing text…"):
                prep: PreprocessingResult = preprocess(
                    user_text,
                    remove_stops=remove_stops,
                    translate=translate_flag,
                )

            with st.spinner("Running BERT inference…"):
                start_time: float = time.perf_counter()
                pred: SentimentPrediction = predict(pipeline, prep.cleaned_text)
                elapsed_ms: float = (time.perf_counter() - start_time) * 1000

            st.caption(f"Inference time: {elapsed_ms:.1f} ms")
            _render_single_result(user_text, prep, pred)

    else:
        st.markdown(
            """
            <div class="card" style="margin-bottom:1rem;">
              <p style="color:#94a3b8; margin:0; font-size:0.9rem;">
                Upload a CSV file with a <code style="color:#a5b4fc;">text</code> column.
                Each row will be preprocessed and classified individually.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                df_input: pd.DataFrame = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Failed to read CSV: {exc}")
                st.stop()

            if "text" not in df_input.columns:
                st.error(
                    "CSV must contain a column named **`text`**. "
                    f"Found columns: {list(df_input.columns)}"
                )
                st.stop()

            raw_texts: list[str] = df_input["text"].astype(str).tolist()
            st.info(f"Loaded **{len(raw_texts)} rows**. Click below to run analysis.")

            if st.button("Run Batch Analysis", use_container_width=False):
                progress_bar = st.progress(0, text="Preprocessing…")

                prep_results: list[PreprocessingResult] = []
                for i, t in enumerate(raw_texts):
                    prep_results.append(
                        preprocess(
                            t,
                            remove_stops=remove_stops,
                            translate=translate_flag,
                        )
                    )
                    progress_bar.progress(
                        int((i + 1) / len(raw_texts) * 50),
                        text=f"Preprocessing {i + 1}/{len(raw_texts)}…",
                    )

                cleaned_texts: list[str] = [r.cleaned_text for r in prep_results]

                progress_bar.progress(55, text="Running BERT inference…")
                start_time = time.perf_counter()
                predictions: list[SentimentPrediction] = predict_batch(
                    pipeline, cleaned_texts
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                progress_bar.progress(100, text="Done ✓")

                st.caption(
                    f"Batch inference: {elapsed_ms:.0f} ms  "
                    f"({elapsed_ms / len(predictions):.1f} ms/sample)"
                )

                _render_batch_results(predictions, prep_results, raw_texts)


if __name__ == "__main__":
    main()
