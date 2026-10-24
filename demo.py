"""
demo.py
=======
Interactive Gradio demo for the Bengali Answer Evaluation System.

Loads the pre-trained GAT-GNN and XGBoost+MLP scorer, then scores a single
student answer against a reference answer for a given question.

Usage
-----
    python demo.py [--share]

Flags
-----
    --share   Launch with a public Gradio link (useful for Kaggle / Colab).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

# ── Bengali font resolution (done once at import time) ────────────────────────
def _resolve_bengali_font():
    """Return (font_path, FontProperties) for the best available Bengali font."""
    _CANDIDATES = [
        # Windows — per-user Noto variable font install location
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts",
                     "NotoSansBengali-VariableFont_wdth,wght.ttf"),
    ]
    # Also search the font manager registry
    for _entry in _fm.fontManager.ttflist:
        _n = _entry.name.lower()
        if ("noto" in _n and "bengali" in _n) or ("lohit" in _n and "bengali" in _n):
            _CANDIDATES.insert(0, _entry.fname)

    for _path in _CANDIDATES:
        if _path and __import__("os").path.exists(_path):
            try:
                _fp = _fm.FontProperties(fname=_path)
                _fm.fontManager.addfont(_path)
                print(f"[OK] Bengali font for KG graph: {__import__('os').path.basename(_path)}")
                return _path, _fp
            except Exception:
                pass
    print("[WARN] No Bengali font found — KG labels may show boxes.")
    return None, None

_BENGALI_FONT_PATH, _BENGALI_FP = _resolve_bengali_font()

try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "Gradio is not installed.  Run:  pip install gradio>=4.0"
    )

from src.config import (
    PASS_THRESHOLD,
    MODELS_DIR,
    gnn_model_filename,
    scorer_model_filename,
)
from src.evaluation.overrides import apply_hard_overrides
from src.experiment import build_components
from src.features.pipeline import extract_pair_features, get_gnn_similarities, scorer_matrix
from src.models.answer_scorer import AnswerScorer
from src.models.siamese_gnn import SiameseGNN
from src.utils.helpers import ensure_utf8_stdio


# ── Helpers ───────────────────────────────────────────────────────────────────

# Sentence-length thresholds (words) that map input to a model tier
_SINGLE_MAX_WORDS  = 15   # ≤15 words avg → single-sentence model
_MEDIUM_MAX_WORDS  = 40   # 16–40 words avg → medium-sentences model
                          # >40 words avg → long-sentences model

# Priority-ordered list of (stem_keyword, tier_label) pairs.
# The stem keyword must appear in the .pth filename.
_TIER_ORDER = [
    ("single",  "Single-Sentence"),
    ("medium",  "Medium-Sentences"),
    ("long",    "Long-Sentences"),
]


def _scan_model_pairs(models_dir: str = MODELS_DIR) -> dict:
    """
    Scan *models_dir* for matching GNN (.pth) + scorer (.npz) pairs.

    Returns a dict keyed by tier keyword ("single", "medium", "long"),
    each value being {"gnn": path, "scorer": path}.
    Only tiers where BOTH files exist are included.
    """
    pairs = {}
    if not os.path.isdir(models_dir):
        return pairs

    pth_files = {f for f in os.listdir(models_dir) if f.endswith(".pth")}
    npz_files = {f for f in os.listdir(models_dir) if f.endswith(".npz")}

    # Prefer models trained on original + clean synthetic data ('_plus_clean')
    prefer = lambda f: (0 if "_plus_clean" in f else 1, f)
    for stem, label in _TIER_ORDER:
        # Match any .pth file whose name contains the tier keyword
        gnn_matches = sorted((f for f in pth_files if stem in f.lower()), key=prefer)
        if not gnn_matches:
            continue
        # The scorer must come from the same training run as the GNN
        scorer = gnn_matches[0].replace("nlp_dominator_gnn_", "nlp_dominator_scorer_").replace(".pth", ".npz")
        if scorer in npz_files:
            pairs[stem] = {
                "gnn":    os.path.join(models_dir, gnn_matches[0]),
                "scorer": os.path.join(models_dir, scorer),
                "label":  label,
            }

    return pairs


def _pick_model_tier(ref_text: str, stu_text: str,
                     available_pairs: dict) -> tuple:
    """
    Decide which model tier to use based on average word count of the
    reference + student answers, then pick the closest available tier.

    Returns (gnn_path, scorer_path, tier_label) or (None, None, "heuristic").
    """
    avg_words = (len(ref_text.split()) + len(stu_text.split())) / 2.0

    if avg_words <= _SINGLE_MAX_WORDS:
        preferred = ["single", "medium", "long"]
    elif avg_words <= _MEDIUM_MAX_WORDS:
        preferred = ["medium", "single", "long"]
    else:
        preferred = ["long", "medium", "single"]

    for tier in preferred:
        if tier in available_pairs:
            info = available_pairs[tier]
            return info["gnn"], info["scorer"], info["label"], avg_words

    return None, None, "heuristic", avg_words


def _verdict(score: float) -> str:
    if score >= 80:
        return "✅ Excellent"
    elif score >= 60:
        return "✔️ Good"
    elif score >= PASS_THRESHOLD:
        return "⚠️  Passing"
    else:
        return "❌ Insufficient"


def _truncate_label(text: str, max_len: int = 16) -> str:
    s = str(text)
    return s if len(s) <= max_len else f"{s[:max_len-1]}…"


def _render_kg_preview(ref_kg, stu_kg):
    """Render reference and student KGs side-by-side for Gradio Image output."""
    try:
        # Apply Bengali font globally for this figure so node labels render correctly
        with matplotlib.rc_context({
            "font.family": (_BENGALI_FP.get_name() if _BENGALI_FP else "DejaVu Sans"),
        }):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
            fig.patch.set_facecolor("#f8fafc")

            panels = [
                ("Reference KG", ref_kg, "#0ea5a4"),
                ("Student KG",   stu_kg, "#f97316"),
            ]

            for ax, (title, graph, color) in zip(axes, panels):
                ax.set_title(title, fontsize=11, fontweight="bold", color="#0f172a",
                             fontproperties=_BENGALI_FP)
                ax.set_axis_off()

                if graph is None or graph.number_of_nodes() == 0:
                    ax.text(
                        0.5, 0.5, "No graph nodes",
                        ha="center", va="center", fontsize=10, color="#64748b",
                        transform=ax.transAxes,
                    )
                    continue

                pos = nx.spring_layout(graph, seed=42, k=1.1)
                nx.draw_networkx_edges(
                    graph, pos, ax=ax,
                    edge_color="#94a3b8", width=1.2, alpha=0.7,
                )
                nx.draw_networkx_nodes(
                    graph, pos, ax=ax,
                    node_color=color, node_size=900,
                    alpha=0.95, linewidths=1.5, edgecolors="#ffffff",
                )

                # draw_networkx_labels doesn't accept FontProperties directly;
                # draw them manually so we can set fontproperties per Text object.
                labels = {node: _truncate_label(graph.nodes[node].get("label", node))
                          for node in graph.nodes()}
                x_coords, y_coords = zip(*[pos[n] for n in graph.nodes()])
                ax.set_xlim(min(x_coords) - 0.15, max(x_coords) + 0.15)
                ax.set_ylim(min(y_coords) - 0.15, max(y_coords) + 0.15)

                for node, (x, y) in pos.items():
                    ax.text(
                        x, y, labels[node],
                        ha="center", va="center",
                        fontsize=8, color="#0f172a", fontweight="bold",
                        fontproperties=_BENGALI_FP,
                        clip_on=True,
                    )

            fig.tight_layout(pad=1.4)
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            plt.close(fig)
            return rgb
    except Exception as e:
        print(f"[WARN] Could not render KG preview: {e}")
        plt.close("all")
        return None


# ── HuggingFace auto-download ─────────────────────────────────────────────

_HF_REPO_ID = "k040902/bengali_question_ans"

# Files stored in the HF repo: GNN + scorer per tier, trained on the original
# data and on original + clean synthetic data ('_plus_clean', preferred)
_HF_MODEL_FILES = [
    f(f"dataset_{tier}{suffix}")
    for f in (gnn_model_filename, scorer_model_filename)
    for suffix in ("", "_plus_clean")
    for tier in ("single_sentence", "medium_sentences", "long_sentences")
]


def _ensure_models_downloaded(models_dir: str = MODELS_DIR) -> None:
    """
    For every file in _HF_MODEL_FILES, download it from HuggingFace into
    *models_dir* if it is not already present locally.
    """
    os.makedirs(models_dir, exist_ok=True)
    missing = [
        f for f in _HF_MODEL_FILES
        if not os.path.exists(os.path.join(models_dir, f))
    ]
    if not missing:
        print("[INFO] All model files already present in models/.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[WARN] huggingface_hub not installed. "
              "Run: pip install huggingface_hub\n"
              "      Skipping model download.")
        return

    print(f"[INFO] {len(missing)} model file(s) missing. "
          f"Downloading from {_HF_REPO_ID} …")

    for filename in missing:
        dest = os.path.join(models_dir, filename)
        print(f"  ↓  {filename} …", end=" ", flush=True)
        try:
            cached_path = hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=filename,
                repo_type="model",
                local_dir=models_dir,      # save directly into models/
            )
            # With local_dir the file usually already sits at dest; copying a
            # file onto itself raised WinError 32 and was reported as FAILED.
            if not os.path.exists(dest):
                import shutil
                shutil.copy2(cached_path, dest)
            print(f"done  →  {dest}")
        except Exception as e:
            print(f"FAILED ({e})")

    print("[INFO] Download step complete.")


# ── Global objects (loaded once) ───────────────────────────────────────────────

_COMPONENTS     = {}   # filled by _load_components()
_MODEL_PAIRS    = {}   # {"single": {...}, "medium": {...}, "long": {...}}
_GNN_CACHE      = {}   # tier_key -> SiameseGNN (loaded lazily per tier)
_SCORER_CACHE   = {}   # tier_key -> (AnswerScorer, ready:bool)


def _load_components():
    """Load NLP pipeline components (once).  Model weights loaded lazily per request."""
    global _COMPONENTS, _MODEL_PAIRS

    if _COMPONENTS:          # already loaded
        return

    # ── Step 0: ensure trained model files exist (download from HF if missing) ─
    _ensure_models_downloaded(MODELS_DIR)

    print("[INFO] Loading NLP components …")
    _COMPONENTS = build_components()

    # Scan models/ directory for available trained pairs
    _MODEL_PAIRS = _scan_model_pairs(MODELS_DIR)
    if _MODEL_PAIRS:
        labels = ", ".join(f"{v['label']} ({v['gnn']})" for v in _MODEL_PAIRS.values())
        print(f"[OK] Found model pairs: {labels}")
    else:
        print("[WARN] No trained model pairs found in models/. Will use heuristic scoring.")


def _get_gnn_and_scorer(tier_key: str, gnn_path: str, scorer_path: str):
    """
    Lazily load and cache GNN + scorer for a given tier.
    Returns (gnn, scorer, ready) — ready only when BOTH loaded, because a
    scorer is only valid together with the GNN it was trained with.
    """
    if tier_key not in _GNN_CACHE:
        gnn, scorer, ready = SiameseGNN(), AnswerScorer(), False
        try:
            gnn.load_state_dict(torch.load(gnn_path, map_location="cpu", weights_only=True))
            gnn.eval()
            scorer.load_model(scorer_path)
            ready = True
            print(f"[OK] Models loaded: {gnn_path}, {scorer_path}")
        except Exception as e:
            print(f"[WARN] Could not load {gnn_path} / {scorer_path}: {e}. Falling back to heuristic.")
        _GNN_CACHE[tier_key] = gnn
        _SCORER_CACHE[tier_key] = (scorer, ready)

    gnn, (scorer, ready) = _GNN_CACHE[tier_key], _SCORER_CACHE[tier_key]
    return gnn, scorer, ready


# ── Core inference function ────────────────────────────────────────────────────

def predict_single(question: str, ref_text: str, stu_text: str):
    """
    Score a single (question, reference_answer, student_answer) triple.
    Automatically selects the best-matching trained model from models/ based
    on the average word count of the reference + student answers.

    Features come from the same extract_pair_features() used for training,
    so the demo scores exactly as evaluate.py does.

    Returns
    -------
    verdict : str
    explanation : str
    score : float  (0-100)
    graph_preview : np.ndarray | None
    """
    _load_components()

    # ── Auto-select model tier based on sentence length ───────────────────────
    gnn_path, scorer_path, tier_label, avg_words = _pick_model_tier(
        ref_text, stu_text, _MODEL_PAIRS
    )

    if gnn_path is not None:
        # Derive a stable cache key from the filenames (not full path)
        tier_key = os.path.splitext(os.path.basename(gnn_path))[0]
        gnn, scorer, scorer_ready = _get_gnn_and_scorer(tier_key, gnn_path, scorer_path)
    else:
        gnn, scorer, scorer_ready = None, None, False

    f = extract_pair_features(question, ref_text, stu_text, _COMPONENTS)
    base_sim, penalty, entity_mm = f["base_sim"], f["penalty"], f["entity_mm"]
    coverage, rouge_l = f["coverage"], f["rouge_l"]
    neg_flag, wrong_num = f["neg_mm"], f["wrong_num"]
    ref_kg, stu_kg = f["ref_kg"], f["stu_kg"]

    # ── GNN similarity + scorer ───────────────────────────────────────────────
    gnn_sim = float(get_gnn_similarities(gnn, [f["ref_pyg"]], [f["stu_pyg"]])[0]) if scorer_ready else 0.0
    raw_score = None
    used_heuristic = False
    if scorer_ready:
        try:
            scalar_feats = {k: np.array([v]) for k, v in f.items() if isinstance(v, float)}
            raw_score = float(scorer.predict_batch(scorer_matrix(scalar_feats, [gnn_sim]))[0])
        except Exception as e:
            print(f"[WARN] Scorer prediction failed ({e}). Falling back to heuristic scoring.")

    if raw_score is None:
        used_heuristic = True
        # Heuristic fallback keeps the demo usable when trained weights are unavailable
        # (an untrained GNN is not used: its similarity would be noise).
        raw_score = 100.0 * (
            0.45 * base_sim
            + 0.25 * coverage
            + 0.15 * (1.0 - entity_mm)
            + 0.15 * rouge_l
        )
        raw_score -= 10.0 * max(0.0, penalty)
        raw_score = float(np.clip(raw_score, 0.0, 100.0))

    # ── Hard overrides (single-sentence references only) ──────────────────────
    final_arr, override_log = apply_hard_overrides(
        np.array([raw_score]), [neg_flag], [wrong_num], [f["multi_sentence"]]
    )
    final_score = float(final_arr[0])

    # ── Build explanation ─────────────────────────────────────────────────────
    model_info = (
        "heuristic fallback (no trained weights found)"
        if used_heuristic
        else f"{tier_label}  [{os.path.basename(gnn_path)}]"
    )
    explanation = (
        f"━━━ Model Selection ━━━\n"
        f"  Avg word count (ref+stu) : {avg_words:.1f} words\n"
        f"  Tier selected            : {model_info}\n"
        f"  Thresholds               : single ≤{_SINGLE_MAX_WORDS}w · medium ≤{_MEDIUM_MAX_WORDS}w · long >{_MEDIUM_MAX_WORDS}w\n"
        f"\n"
        f"━━━ Feature Breakdown ━━━\n"
        f"  Cosine similarity (LaBSE)            : {base_sim:.4f}\n"
        f"  GNN graph similarity                 : {gnn_sim:.4f}\n"
        f"  Karak / semantic-role penalty        : {penalty:.4f}\n"
        f"  Entity mismatch penalty              : {entity_mm:.4f}\n"
        f"  Sentence coverage score              : {coverage:.4f}\n"
        f"  Negation mismatch flag               : {int(neg_flag)}\n"
        f"  Wrong-number flag                    : {int(wrong_num)}\n"
        f"  ROUGE-L                              : {rouge_l:.4f}\n"
        f"\n"
        f"━━━ Scores ━━━\n"
        f"  Raw ensemble score           : {raw_score:.2f} / 100\n"
        f"  Final score (after overrides): {final_score:.2f} / 100\n"
    )
    for reason in override_log[0]:
        explanation += f"\n⚠️  {reason}.\n"
    if f["karak_explanations"]:
        explanation += "\nRole / structure checks:\n" + "".join(
            f"  - {e}\n" for e in f["karak_explanations"])
    if used_heuristic:
        explanation += "\nℹ️  Using heuristic fallback scoring because no trained model weights were found in models/.\n"

    explanation += (
        f"\nKG stats → reference: {ref_kg.number_of_nodes()} nodes / {ref_kg.number_of_edges()} edges; "
        f"student: {stu_kg.number_of_nodes()} nodes / {stu_kg.number_of_edges()} edges.\n"
    )

    graph_preview = _render_kg_preview(ref_kg, stu_kg)

    verdict = _verdict(final_score)
    return verdict, explanation, round(final_score, 2), graph_preview


def predict_single_safe(question: str, ref_text: str, stu_text: str):
    """Prevent Gradio from collapsing into generic 'Error' on unexpected exceptions."""
    try:
        return predict_single(question, ref_text, stu_text)
    except Exception as e:
        msg = f"Runtime error during evaluation: {e}"
        print(f"[ERROR] {msg}")
        return "❌ Error", msg, 0.0, None


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EXAMPLES = [
    [
        "বাংলাদেশ কবে স্বাধীন হয়েছিল?",
        "বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়েছিল।",
        "বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়েছিল।",
    ],
    [
        "বাংলাদেশ কবে স্বাধীন হয়েছিল?",
        "বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়েছিল।",
        "বাংলাদেশ ১৯৭২ সালে স্বাধীন হয়েছিল।",
    ],
    [
        "সূর্য কোথায় ওঠে?",
        "সূর্য পূর্বদিকে ওঠে।",
        "সূর্য পূর্বদিকে ওঠে না।",
    ],
    [
        "কে রাবণকে বধ করেছিলেন?",
        "রাম রাবণকে বধ করেছিলেন।",
        "রাবণ রামকে বধ করেছিলেন।",
    ],
    [
        "মানুষের শরীরে কয়টি হাড় থাকে?",
        "মানুষের শরীরে ২০৬টি হাড় থাকে।",
        "মানুষের শরীরে ২০৫টি হাড় থাকে।",
    ],
    [
        "গীতাঞ্জলি কে লিখেছিলেন?",
        "রবীন্দ্রনাথ ঠাকুর গীতাঞ্জলি লিখেছিলেন।",
        "রবীন্দ্রনাথ ঠাকুর গীতাঞ্জলি।",
    ],
]


def build_interface() -> gr.Blocks:
    css = """
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Noto+Sans+Bengali:wght@400;600;700&display=swap');

        :root {
            --bg-1: #fff8ef;
            --bg-2: #ecfeff;
            --ink-1: #111827;
            --ink-2: #334155;
            --brand: #ea580c;
            --brand-2: #0f766e;
            --card: rgba(255, 255, 255, 0.78);
            --border: rgba(148, 163, 184, 0.35);
        }

        .gradio-container {
            font-family: 'Space Grotesk', 'Noto Sans Bengali', sans-serif !important;
            background:
                radial-gradient(1200px 400px at -10% -20%, #fed7aa 0%, transparent 60%),
                radial-gradient(1200px 400px at 110% 0%, #a7f3d0 0%, transparent 60%),
                linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
        }

        .hero {
            background: linear-gradient(130deg, rgba(255, 255, 255, 0.8), rgba(255, 237, 213, 0.82));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
            margin-bottom: 8px;
        }

        .panel {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(6px);
            padding: 8px;
        }

        button.primary {
            background: linear-gradient(90deg, var(--brand), #fb923c) !important;
            border: none !important;
            color: #fff !important;
            font-weight: 700 !important;
        }

        button.secondary {
            background: linear-gradient(90deg, var(--brand-2), #2dd4bf) !important;
            border: none !important;
            color: #fff !important;
            font-weight: 700 !important;
        }

        .section-title {
            color: var(--ink-1);
            letter-spacing: 0.2px;
        }

        .muted {
            color: var(--ink-2);
            font-size: 0.95rem;
        }

        @media (max-width: 900px) {
            .hero { padding: 14px; }
            .panel { border-radius: 14px; }
        }

    footer { display: none !important; }
    """

    with gr.Blocks(title="Bengali Answer Evaluator", css=css) as demo:
        gr.Markdown(
            """
            <div class="hero">
              <h1 style="margin:0 0 8px 0;color:#111827;">🏛️ Bengali Answer Evaluation Studio</h1>
              <div class="muted"><b>Knowledge Graph</b> · LaBSE · BNLP POS · Siamese GAT-GNN · Karak Validator · XGBoost+MLP</div>
              <div class="muted" style="margin-top:8px;">Paste a question, reference answer, and student answer in Bengali, then click <b>Evaluate</b> to inspect score, explanation, and KG graph structure.</div>
            </div>
            """,
            elem_classes=["panel"],
        )

        with gr.Row():
            with gr.Column(scale=2, elem_classes=["panel"]):
                question_input  = gr.Textbox(
                    label="Question (প্রশ্ন)",
                    placeholder="যেমন: বাংলাদেশ কবে স্বাধীন হয়েছিল?",
                    lines=2,
                )
                reference_input = gr.Textbox(
                    label="Reference Answer (সঠিক উত্তর)",
                    placeholder="যেমন: বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়েছিল।",
                    lines=3,
                )
                student_input   = gr.Textbox(
                    label="Student Answer (ছাত্রের উত্তর)",
                    placeholder="যেমন: বাংলাদেশ ১৯৭২ সালে স্বাধীন হয়েছিল।",
                    lines=3,
                )
                with gr.Row():
                    submit_btn = gr.Button("🚀 Evaluate Answer", variant="primary", scale=3)
                    clear_btn  = gr.Button("🧹 Clear", variant="secondary", scale=1)

            with gr.Column(scale=1, elem_classes=["panel"]):
                gr.Markdown("### 📊 Result", elem_classes=["section-title"])
                score_output = gr.Number(label="Score (0–100)", precision=2)
                label_output = gr.Textbox(label="Verdict", interactive=False)
                graph_output = gr.Image(
                    label="Knowledge Graph Preview",
                    type="numpy",
                    interactive=False,
                    height=300,
                )

        gr.Markdown("### 🔍 Detailed Explanation", elem_classes=["section-title"])
        explanation_output = gr.Textbox(
            label="Component Breakdown",
            lines=18,
            interactive=False,
        )

        gr.Markdown("### 📚 Example Test Cases")
        gr.Examples(
            examples=EXAMPLES,
            inputs=[question_input, reference_input, student_input],
        )

        gr.HTML(
            """
            <div style="text-align:center;margin-top:10px;color:#888;font-size:12px;">
              🏛️ Knowledge Graph-Based Bengali Answer Evaluation · IIIT Hyderabad 2025<br/>
              Pipeline: LaBSE + BNLP POS + KG + Siamese GAT-GNN + Karak Validator + XGBoost+MLP
            </div>
            """
        )

        # ── Wire-up ─────────────────────────────────────────────────────────
        submit_btn.click(
            fn=predict_single_safe,
            inputs=[question_input, reference_input, student_input],
            outputs=[label_output, explanation_output, score_output, graph_output],
        )
        clear_btn.click(
            fn=lambda: ("", "", "", "", "", 0.0, None),
            inputs=[],
            outputs=[question_input, reference_input, student_input,
                     label_output, explanation_output, score_output, graph_output],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Bengali Answer Evaluation — Gradio Demo")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link.")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    # Pre-load all heavy components before the server starts
    _load_components()

    demo = build_interface()
    demo.launch(
        share=args.share,
        server_port=args.server_port,
        debug=False,
    )


if __name__ == "__main__":
    main()
