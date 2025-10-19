import re, os, tempfile, math
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from functools import lru_cache
from transformers import pipeline
import emoji
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # deterministic language detection

# ----------------------
# Models and pipelines
# ----------------------
SENTIMENT_CHOICES = {
    "SST-2 (binary, reviews)": "distilbert-base-uncased-finetuned-sst-2-english",  # POSITIVE/NEGATIVE
    "Twitter (EN, 3-class)": "cardiffnlp/twitter-roberta-base-sentiment-latest",    # NEGATIVE/NEUTRAL/POSITIVE
    "Twitter (Multilingual, 3-class)": "cardiffnlp/twitter-xlm-roberta-base-sentiment"
}
EMOTION_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"  # sadness/joy/love/anger/fear/surprise

@lru_cache(maxsize=8)
def get_pipe(task, model_id):
    # device_map="auto" if GPU is present in Colab; else CPU fallback
    return pipeline(task, model=model_id, tokenizer=model_id if "cardiffnlp" in model_id else None)

# ----------------------
# Text utilities
# ----------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")

def clean_line(s, demojize=True, strip_social=True, lower=False):
    if demojize:
        s = emoji.demojize(s, language='en')
    if strip_social:
        s = URL_RE.sub("", s)
        s = MENTION_RE.sub("", s)
        # Keep hashtag token but turn into plain word
        s = HASHTAG_RE.sub(r"\1", s)
    s = s.strip()
    if lower:
        s = s.lower()
    return s

def detect_langs(lines, probe=30):
    # quick language probe on a sample
    sample = [l for l in lines if l.strip()][:probe]
    counts = {}
    for s in sample:
        try:
            code = detect(s)
            counts[code] = counts.get(code, 0) + 1
        except:
            counts["unk"] = counts.get("unk", 0) + 1
    total = sum(counts.values()) or 1
    share_en = counts.get("en", 0) / total
    return counts, share_en

# ----------------------
# Core analyzer
# ----------------------
def run_analysis(
    text_block, file_obj, text_col, mode, sentiment_model_choice, auto_model, demojize_opt,
    strip_social_opt, lower_opt, batch_size
):
    # Collect lines from textbox and/or CSV
    lines = []
    if text_block:
        lines.extend([l.rstrip() for l in text_block.splitlines() if l.strip()])

    df_in = None
    if file_obj is not None:
        try:
            df_in = pd.read_csv(file_obj.name)
            use_col = text_col if (text_col and text_col in df_in.columns) else None
            if not use_col:
                # naive auto-pick
                for c in ["text", "message", "msg", "content", "body"]:
                    if c in df_in.columns:
                        use_col = c
                        break
            if not use_col:
                return (pd.DataFrame([{"error": "CSV loaded, but no text column selected/found."}]),
                        plt.figure(), gr.update(value=None), "No language info")
            lines.extend([str(x) for x in df_in[use_col].astype(str).tolist()])
        except Exception as e:
            return (pd.DataFrame([{"error": f"Failed to read CSV: {e}"}]),
                    plt.figure(), gr.update(value=None), "No language info")

    if not lines:
        return (pd.DataFrame([{"error": "Enter text or upload CSV with a text column."}]),
                plt.figure(), gr.update(value=None), "No language info")

    # Preprocess
    proc = [clean_line(l, demojize=demojize_opt, strip_social=strip_social_opt, lower=lower_opt) for l in lines]

    # Language probe to optionally switch sentiment model
    lang_counts, share_en = detect_langs(proc, probe=min(30, len(proc)))
    lang_info = f"Lang probe (top): {dict(sorted(lang_counts.items(), key=lambda x: -x[1])[:3])}, EN share≈{round(share_en,2)}"

    # Choose model
    if mode == "Sentiment":
        if auto_model:
            model_id = SENTIMENT_CHOICES["Twitter (EN, 3-class)"] if share_en >= 0.6 else SENTIMENT_CHOICES["Twitter (Multilingual, 3-class)"]
        else:
            model_id = SENTIMENT_CHOICES[sentiment_model_choice]
        pipe = get_pipe("sentiment-analysis", model_id)
    else:
        pipe = get_pipe("text-classification", EMOTION_MODEL)

    # Batched inference for speed
    outputs = []
    for i in range(0, len(proc), batch_size):
        batch = proc[i:i+batch_size]
        outs = pipe(batch, batch_size=batch_size, truncation=True)
        # normalize to list[dict]
        for out in outs:
            out0 = out[0] if isinstance(out, list) else out
            outputs.append({"label": out0["label"], "score": float(out0["score"])})

    # Build results DataFrame
    rows = []
    for idx, (raw, out) in enumerate(zip(lines, outputs), 1):
        rows.append({
            "idx": idx,
            "text": raw,
            "label": out["label"],
            "score": round(out["score"], 4)
        })
    df = pd.DataFrame(rows)

    # Distribution plot (matplotlib)
    counts = df["label"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title("Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.tight_layout()

    # Export to CSV
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)

    return df, fig, tmp.name, lang_info

# ----------------------
# UI
# ----------------------
with gr.Blocks(title="Chat Mood Analyzer — Ultimate") as demo:
    gr.Markdown("## Chat Mood Analyzer — Ultimate Edition")

    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(lines=10, label="Paste chat (one message per line)")
            file_in = gr.File(label="Or upload CSV", file_types=[".csv"], file_count="single")
            text_col = gr.Textbox(value="", label="CSV text column (auto-detect if blank)")

            mode = gr.Radio(["Sentiment", "Emotion"], value="Sentiment", label="Analysis mode")

            with gr.Accordion("Sentiment model settings", open=False):
                auto_model = gr.Checkbox(value=True, label="Auto-pick tweet-aware EN vs Multilingual")
                sentiment_model = gr.Dropdown(
                    choices=list(SENTIMENT_CHOICES.keys()),
                    value="Twitter (EN, 3-class)",
                    label="Manual model (used if Auto is OFF)"
                )
                gr.Markdown("Tip: Twitter models understand slang/emojis better than SST‑2 review models.")

            with gr.Accordion("Preprocessing", open=False):
                demojize_opt = gr.Checkbox(value=True, label="Convert emojis to text (:face_with_tears_of_joy:)")
                strip_social_opt = gr.Checkbox(value=True, label="Strip URLs/@mentions/#hashtags")
                lower_opt = gr.Checkbox(value=False, label="Lowercase text")

            batch_size = gr.Slider(1, 64, value=16, step=1, label="Batch size")

            run = gr.Button("Analyze", variant="primary")
            clear = gr.ClearButton([txt, file_in])

        with gr.Column():
            out_table = gr.Dataframe(label="Per-message results", wrap=True)
            out_plot = gr.Plot(label="Label distribution")
            download = gr.File(label="Download results (.csv)")
            lang_probe = gr.Markdown()

    evt = run.click(
        fn=run_analysis,
        inputs=[txt, file_in, text_col, mode, sentiment_model, auto_model,
                demojize_opt, strip_social_opt, lower_opt, batch_size],
        outputs=[out_table, out_plot, download, lang_probe],
        concurrency_limit=4,
        show_progress=True
    )

    demo.queue(max_size=64, default_concurrency_limit=2)
    demo.launch(share=True)
