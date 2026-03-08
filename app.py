"""
app.py - Gradio web interface for Bengali Answer Evaluation System.

Run with:  python app.py
"""

import os
import tempfile
import torch
import networkx as nx
import matplotlib.pyplot as plt
import gradio as gr

from src.inference import load_pretrained_models, grade_single
from src.preprocessing import TextPreprocessor
from src.coreference import BengaliCoreferenceResolver
from src.dependency_parser import BengaliDependencyParser
from src.triple_extractor import TripleExtractor
from src.knowledge_graph import KnowledgeGraphConstructor
from src.embeddings import BanglaBERTEmbedder
from src.karak_validator import KarakValidator

# ── component initialisation (lazy, shared across requests) ───────────────────

_components = {}

def _init_components():
    """Initialise NLP components once and cache them."""
    if _components:
        return

    import stanza
    from transformers import pipeline as hf_pipeline

    print("Initialising NLP components...")

    nlp = None
    try:
        stanza.download("bn", verbose=False)
        nlp = stanza.Pipeline("bn", use_gpu=False, verbose=False)
    except Exception as e:
        print(f"  Stanza failed: {e}. Using SOV fallback.")

    ner_pipe = None
    try:
        ner_pipe = hf_pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl",
                               aggregation_strategy="simple", device=-1)
    except Exception as e:
        print(f"  NER pipeline failed: {e}.")

    _components["text_processor"] = TextPreprocessor(nlp_pipeline=nlp)
    _components["dep_parser"] = BengaliDependencyParser(nlp_pipeline=nlp)
    _components["coref_resolver"] = BengaliCoreferenceResolver(nlp_pipeline=nlp)
    _components["triple_extractor"] = TripleExtractor()
    _components["kg_constructor"] = KnowledgeGraphConstructor()
    _components["embedder"] = BanglaBERTEmbedder()
    _components["validator"] = KarakValidator()
    _components["ner_pipe"] = ner_pipe

    gnn, scorer = load_pretrained_models()
    _components["gnn"] = gnn
    _components["scorer"] = scorer

    print("Components ready.")

# ── grading function ──────────────────────────────────────────────────────────

def grade_answer_web(question: str, reference_answer: str, student_answer: str):
    """Gradio handler: returns (score_text, detail_text, kg_image_path)."""
    try:
        if not question.strip() or not reference_answer.strip() or not student_answer.strip():
            return "⚠ Please fill in all three fields.", "", None

        _init_components()

        gnn    = _components.get("gnn")
        scorer = _components.get("scorer")
        if gnn is None or scorer is None:
            return ("⚠ No trained models found!\n"
                    "Run main.py to train, or place nlp_dominator_gnn.pth and "
                    "nlp_dominator_scorer.pkl in the working directory."), "", None

        result = grade_single(
            question, reference_answer, student_answer,
            gnn_model=gnn,
            scorer=scorer,
            text_processor=_components["text_processor"],
            dep_parser=_components["dep_parser"],
            triple_extractor=_components["triple_extractor"],
            kg_constructor=_components["kg_constructor"],
            embedder=_components["embedder"],
            validator=_components["validator"],
            coref_resolver=_components["coref_resolver"],
            ner_pipeline_obj=_components["ner_pipe"],
        )

        score = result["score"]
        grade_emoji = "🟢" if score >= 70 else ("🟡" if score >= 40 else "🔴")
        grade_label = "Good" if score >= 70 else ("Partial" if score >= 40 else "Poor")
        score_text = f"{grade_emoji} Score: {score:.1f} / 100  ({grade_label})"

        lines = [
            f"📊 Semantic Similarity (LaBSE):       {result['baseline_sim']:.4f}",
            f"📊 Structural Similarity (GAT-GNN):   {result['graph_sim']:.4f}",
            f"📊 Karak Penalty:                     {result['penalty']:.2f}",
            f"📊 Entity Mismatch (soft):            {result['entity_mismatch']:.4f}",
            "",
        ]
        if result["student_resolved"] != student_answer:
            lines.append(f"🔗 Coref Resolved: \"{result['student_resolved']}\"")
            lines.append("")

        lines.append(f"📝 Reference Triples: {result['ref_triples'] or '(none)'}")
        lines.append(f"📝 Student Triples:   {result['stu_triples'] or '(none)'}")
        lines.append("")

        if result["penalty"] > 0 and result["explanations"]:
            lines.append("⚠ Semantic Role Errors:")
            for exp in result["explanations"]:
                lines.append(f"   • {exp}")
        else:
            lines.append("✅ No semantic role or entity errors detected.")

        explanation_text = "\n".join(lines)

        # KG comparison plot
        ref_kg = _components["kg_constructor"].build_graph(result["ref_triples"])
        stu_kg = _components["kg_constructor"].build_graph(result["stu_triples"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, G, title, node_color in [
            (axes[0], ref_kg, "Reference Answer KG", "#A8D8EA"),
            (axes[1], stu_kg, "Student Answer KG",   "#FFCCCB"),
        ]:
            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G, seed=42)
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, node_size=2000)
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=11)
                nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=15, edge_color="gray")
                nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, "relation"),
                                             font_size=9, ax=ax)
            ax.set_title(title, fontsize=13)
            ax.axis("off")

        plt.suptitle(f"Knowledge Graph Comparison  |  Score: {score:.1f}/100",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plot_path = os.path.join(tempfile.gettempdir(), "kg_comparison.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return score_text, explanation_text, plot_path

    except Exception:
        import traceback
        return "❌ Error occurred", traceback.format_exc(), None

# ── Gradio interface ──────────────────────────────────────────────────────────

EXAMPLES = [
    ["সূর্য কোথায় ওঠে?", "সূর্য পূর্বদিকে ওঠে।", "সূর্য পশ্চিমদিকে ওঠে।"],
    ["কে রাবণকে বধ করেছিলেন?", "রাম রাবণকে বধ করেছিলেন।", "রাবণ রামকে বধ করেছিলেন।"],
    ["কে রাবণকে বধ করেছিলেন?", "রাম রাবণকে বধ করেছিলেন।", "তিনি রাবণকে বধ করেছিলেন।"],
    ["পৃথিবী কীসের চারদিকে ঘোরে?", "পৃথিবী সূর্যের চারদিকে ঘোরে।", "পৃথিবী সূর্যের চারদিকে ঘোরে।"],
    ["গাছ কী তৈরি করে?", "গাছ অক্সিজেন তৈরি করে।", "গাছ অক্সিজেন তৈরি করে না।"],
]

demo = gr.Interface(
    fn=grade_answer_web,
    inputs=[
        gr.Textbox(label="📌 Question (প্রশ্ন)", placeholder="এখানে প্রশ্ন লিখুন...", lines=2),
        gr.Textbox(label="✅ Reference Answer (সঠিক উত্তর)", placeholder="সঠিক উত্তর...", lines=2),
        gr.Textbox(label="🧑‍🎓 Student Answer (ছাত্রের উত্তর)", placeholder="ছাত্রের উত্তর...", lines=2),
    ],
    outputs=[
        gr.Textbox(label="🎯 Score"),
        gr.Textbox(label="📋 Detailed Analysis", lines=15),
        gr.Image(label="📊 Knowledge Graph Comparison", type="filepath"),
    ],
    title="🎓 Bengali Answer Evaluation System",
    description=(
        "Enter a **question**, the **correct reference answer**, and the **student's answer** in Bengali.\n\n"
        "**Architecture:** LaBSE + Coref Resolution + Rule-based Triple Extraction + "
        "GAT-GNN Knowledge Graphs + NER Soft Entity Matching + "
        "Karak (কারক) Validation + XGBoost + MLP Ensemble (0–100)."
    ),
    examples=EXAMPLES,
    flagging_mode="never",
)

if __name__ == "__main__":
    print("🚀 Launching Bengali Answer Evaluation Web App...")
    try:
        gr.close_all()
    except Exception:
        pass
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=None,   # let Gradio pick the first free port (7860, 7861, …)
        theme=gr.themes.Soft(),
    )
