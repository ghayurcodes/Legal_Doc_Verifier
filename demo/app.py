"""
Legal Document Authenticity Verifier — Gradio Web UI
Run this on your PC after downloading trained model weights from Colab.

Command to run:
    cd legal_doc_verifier
    python demo/app.py
"""

import gradio as gr
import torch
from transformers import RobertaTokenizer
import sys
import os

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese_cnn  import SiameseNet
from models.roberta_nlp  import DeceptionClassifier
from agents.preprocessing import PreprocessingAgent
from agents.signature_agent import SignatureAgent
from agents.text_agent     import TextAgent
from agents.supervisor     import SupervisorAgent
from xai.gradcam           import GradCAMAgent
from xai.shap_text         import SHAPTextAgent


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# We run everything on CPU here — GPU not needed for inference
# The .pt files are the trained weights you downloaded from Colab
# ─────────────────────────────────────────────────────────────

DEVICE = torch.device('cpu')

# Paths to saved model weights (downloaded from Colab)
SIAMESE_WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'siamese_best.pt')
ROBERTA_WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'roberta_deception.pt')

print("Loading models... (this may take a moment)")

# Load Siamese CNN
siamese_model = SiameseNet(embedding_dim=128)
siamese_model.load_state_dict(torch.load(SIAMESE_WEIGHTS, map_location=DEVICE))
siamese_model.eval()
print("[OK] Siamese CNN loaded")

# Load RoBERTa tokenizer + classifier
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
nlp_model = DeceptionClassifier()
nlp_model.load_state_dict(torch.load(ROBERTA_WEIGHTS, map_location=DEVICE, weights_only=False))
nlp_model.eval()
print("[OK] RoBERTa loaded")


# ─────────────────────────────────────────────────────────────
# INITIALISE ALL AGENTS
# ─────────────────────────────────────────────────────────────

prep_agent  = PreprocessingAgent()
sig_agent   = SignatureAgent(siamese_model, DEVICE)
txt_agent   = TextAgent(nlp_model, tokenizer, DEVICE)
sup_agent   = SupervisorAgent(sig_weight=0.6, text_weight=0.4, threshold=0.4)
cam_agent   = GradCAMAgent(siamese_model)
shap_agent  = SHAPTextAgent(nlp_model, tokenizer, DEVICE)

print("[OK] All agents ready\n")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# This function is called every time the user clicks "Verify"
# ─────────────────────────────────────────────────────────────

def is_valid_signature_image(img_path, label="Image"):
    """
    Smart input validation for signature images.
    label: "Reference Signature" or "Test Signature" for clear error messages.
    Returns (True, "") or (False, user_friendly_error_string).
    """
    from PIL import Image as PILImage
    import numpy as np

    try:
        img = PILImage.open(img_path)
        w, h = img.size

        # Check 1: Minimum dimensions
        if w < 30 or h < 30:
            return False, (
                f"{label} is too small ({w}x{h} px). "
                "Please upload a proper signature scan."
            )

        # Check 2: Aspect ratio — only reject very extreme portrait images
        ratio = w / h
        if ratio < 0.3:
            return False, (
                f"{label} looks like a portrait photo "
                f"({w}x{h} px — height much larger than width). "
                "Please upload a horizontal signature image."
            )

        # Check 3: Reject completely blank / solid color images
        # Use std deviation — a real signature always has variation in pixels.
        # Threshold is very low (5) so even thin-stroked signatures pass.
        arr = np.array(img.convert('L'), dtype=float)
        if arr.std() < 5:
            return False, (
                f"{label} appears to be blank or a solid color. "
                "Please upload a signature image."
            )

        return True, ""

    except Exception:
        return False, (
            f"{label}: Could not read the file. "
            "Please upload a valid PNG or JPG image."
        )


def verify_document(ref_sig_path, test_sig_path, claim_text):
    """
    ref_sig_path  : reference (known genuine) signature image path
    test_sig_path : signature being tested
    claim_text    : the document text to analyze for deception
    """

    # ── Step 1: Basic presence checks ────────────────────────
    if ref_sig_path is None or test_sig_path is None:
        return None, "⚠️ Please upload both signature images."
    if not claim_text or claim_text.strip() == "":
        return None, "⚠️ Please enter the document text."
    if len(claim_text.strip()) < 20:
        return None, "⚠️ Please enter more text (at least a full sentence)."

    # ── Step 2: Same image uploaded twice? ───────────────────
    if ref_sig_path == test_sig_path:
        return None, (
            "⚠️ Both image boxes have the same image. "
            "Please upload two different signatures — "
            "the Reference (known genuine) and the Test signature."
        )

    # ── Step 3: Validate each image ──────────────────────────
    ok, reason = is_valid_signature_image(ref_sig_path, "Reference Signature")
    if not ok:
        return None, f"⚠️ {reason}"

    ok, reason = is_valid_signature_image(test_sig_path, "Test Signature")
    if not ok:
        return None, f"⚠️ {reason}"

    # ── Step 2: Preprocess inputs ─────────────────────────────
    ref_tensor  = prep_agent.prepare_signature(ref_sig_path)
    test_tensor = prep_agent.prepare_signature(test_sig_path)
    clean_text  = prep_agent.prepare_text(claim_text)

    # ── Step 3: Run models ────────────────────────────────────
    sig_score       = sig_agent.verify(ref_tensor, test_tensor)
    deception_score = txt_agent.analyze(clean_text)

    # ── Step 4: Supervisor decision ───────────────────────────
    result = sup_agent.decide(sig_score, deception_score)

    # ── Step 5: XAI explanations ──────────────────────────────
    gradcam_img = cam_agent.generate(test_tensor, test_sig_path)
    top_words   = shap_agent.get_top_words(clean_text, n=5)

    # ── Step 6: Format output ─────────────────────────────────
    verdict_emoji = "🚨 SUSPICIOUS" if result['verdict'] == "SUSPICIOUS" else "✅ AUTHENTIC"
    sig_emoji     = "❌ FORGED"    if result['signature_verdict'] == "FORGED"    else "✅ GENUINE"
    txt_emoji     = "⚠️ DECEPTIVE" if result['text_verdict']      == "DECEPTIVE" else "✅ TRUTHFUL"

    # Format SHAP word list
    shap_lines = "\n".join([
        f"- **{w}** → score: `{v:+.3f}` {'🔴' if v > 0 else '🟢'}"
        for w, v in top_words
    ])

    verdict_text = f"""
## {verdict_emoji}

---

### 📊 Detailed Scores

| Check | Score | Result |
|---|---|---|
| Signature Similarity | `{result['signature_score']:.3f}` | {sig_emoji} |
| Text Deception | `{result['deception_score']:.3f}` | {txt_emoji} |
| **Combined Risk** | **`{result['combined_risk']:.3f}`** | **{verdict_emoji}** |

> Signature genuine if ≥ 0.70 · Text deceptive if ≥ 0.45 · Combined suspicious if ≥ 0.40

---

### 🧠 Key Words Driving Deception Score
🔴 pushed toward deceptive &nbsp;|&nbsp; 🟢 pushed toward truthful

{shap_lines}
    """.strip()

    return gradcam_img, verdict_text



# ─────────────────────────────────────────────────────────────
# GRADIO INTERFACE
# ─────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ───────────────────────────────────────────────── */
body, .gradio-container {
    background: #0d1117 !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
    color: #e6edf3 !important;
}

/* ── Header banner ────────────────────────────────────────── */
.header-box {
    background: linear-gradient(135deg, #0d1f3c 0%, #1a3a5c 100%);
    border: 1px solid #c9973a;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 20px;
}
.header-box h1 { color: #c9973a !important; margin: 0 0 6px 0; font-size: 1.9em; }
.header-box p  { color: #a0b4c8 !important; margin: 0; font-size: 0.95em; }

/* ── Panels ───────────────────────────────────────────────── */
.panel {
    background: #161b22 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

/* ── Labels ───────────────────────────────────────────────── */
label span, .svelte-1b6s6s { color: #a0b4c8 !important; font-weight: 600 !important; }

/* ── Inputs / Textbox ─────────────────────────────────────── */
textarea, input[type=text] {
    background: #0d1117 !important;
    border: 1px solid #2d3748 !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
textarea:focus, input[type=text]:focus {
    border-color: #c9973a !important;
    box-shadow: 0 0 0 2px rgba(201,151,58,0.25) !important;
}

/* ── Upload area ──────────────────────────────────────────── */
.upload-container, [data-testid="image"] {
    background: #0d1117 !important;
    border: 2px dashed #2d3748 !important;
    border-radius: 10px !important;
}
.upload-container:hover { border-color: #c9973a !important; }

/* ── Verify button ────────────────────────────────────────── */
.verify-btn button {
    background: linear-gradient(135deg, #c9973a, #e8b84b) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 28px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.5px;
}
.verify-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(201,151,58,0.4) !important;
}

/* ── Results section ──────────────────────────────────────── */
.results-panel {
    background: #161b22 !important;
    border: 1px solid #c9973a !important;
    border-radius: 10px !important;
    padding: 20px !important;
}

/* ── All image upload boxes — contain image inside boundary ── */
[data-testid="image"] {
    overflow: hidden !important;
}
[data-testid="image"] img {
    max-height: 220px !important;
    width: 100% !important;
    object-fit: contain !important;
    border-radius: 8px !important;
    display: block !important;
}

/* ── Heatmap result image — taller ───────────────────────── */
.results-panel [data-testid="image"] img {
    max-height: 380px !important;
    object-fit: contain !important;
}

/* ── Fullscreen lightbox — fill the screen ───────────────── */
dialog img,
[role="dialog"] img,
.gradio-image-preview img {
    max-height: 90vh !important;
    max-width: 90vw !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
}

/* ── Markdown report ──────────────────────────────────────── */
.prose h2 { color: #c9973a !important; }
.prose h3 { color: #8ab4d4 !important; }
.prose table { border-collapse: collapse; width: 100%; }
.prose td, .prose th {
    border: 1px solid #2d3748 !important;
    padding: 8px 12px !important;
    color: #e6edf3 !important;
}
.prose th { background: #1a3a5c !important; }
.prose code {
    background: #1a2332 !important;
    color: #c9973a !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

/* ── Footer ───────────────────────────────────────────────── */
.footer-text { color: #4a6280 !important; font-size: 0.82em !important; text-align: center; }
"""

with gr.Blocks(title="Legal Document Authenticity Verifier") as demo:

    # ── Header ────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-box">
        <h1>⚖️ Legal Document Authenticity Verifier</h1>
        <p>Forensic-grade signature verification &amp; text deception analysis &nbsp;·&nbsp;
           Powered by Siamese CNN &amp; RoBERTa &nbsp;·&nbsp; XAI via Grad-CAM &amp; SHAP</p>
    </div>
    """)

    # ── Input row ─────────────────────────────────────────────
    gr.Markdown("### 📁 Step 1 — Upload Signatures")
    with gr.Row(equal_height=True):
        ref_sig = gr.Image(
            label="Reference Signature  (known genuine)",
            type="filepath",
            height=220,
            elem_classes=["panel"]
        )
        test_sig = gr.Image(
            label="Test Signature  (under examination)",
            type="filepath",
            height=220,
            elem_classes=["panel"]
        )

    gr.Markdown("### 📝 Step 2 — Paste Document Text")
    claim_text = gr.Textbox(
        label="Document Text / Written Claims",
        placeholder="Paste the written content of the document here...",
        lines=6,
        elem_classes=["panel"]
    )

    # ── Verify button ─────────────────────────────────────────
    with gr.Row():
        verify_btn = gr.Button(
            "🔎  Verify Document Authenticity",
            variant="primary",
            size="lg",
            elem_classes=["verify-btn"]
        )

    gr.Markdown("---")
    gr.Markdown("### 📋 Step 3 — Analysis Results")

    # ── Results row ───────────────────────────────────────────
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gradcam_output = gr.Image(
                label="🔥 Grad-CAM Heatmap — Suspicious Stroke Regions",
                height=380,
                elem_classes=["results-panel"]
            )
        with gr.Column(scale=1):
            verdict_output = gr.Markdown(
                label="Authenticity Report",
                elem_classes=["results-panel"]
            )

    # ── Wire button ────────────────────────────────────────────
    verify_btn.click(
        fn=verify_document,
        inputs=[ref_sig, test_sig, claim_text],
        outputs=[gradcam_output, verdict_output]
    )

    # ── Footer ─────────────────────────────────────────────────
    gr.HTML("""
    <div class="footer-text" style="margin-top:24px; padding: 12px; border-top: 1px solid #2d3748;">
        Signature analysis: Siamese CNN + VGG16 backbone (weight 60%) &nbsp;|&nbsp;
        Text analysis: Fine-tuned RoBERTa-base (weight 40%) &nbsp;|&nbsp;
        Explanations: Grad-CAM + SHAP
    </div>
    """)


# ─────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    demo.launch(share=False, css=CUSTOM_CSS)
