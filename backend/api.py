"""
Legal Document Authenticity Verifier — FastAPI Backend
Run:  uvicorn backend.api:app --reload --port 8000
      (from the legal_doc_verifier directory)
"""

import sys
import os
import io
import base64
import tempfile
import traceback

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import torch
from transformers import RobertaTokenizer, logging as hf_logging

# ── Silence noisy HuggingFace warnings ──────────────────────────
hf_logging.set_verbosity_error()

# ── Path setup ───────────────────────────────────────────────────
# Point to the backend directory so it can import models, agents, etc. natively
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.siamese_cnn   import SiameseNet
from models.roberta_nlp   import DeceptionClassifier
from agents.preprocessing  import PreprocessingAgent
from agents.signature_agent import SignatureAgent
from agents.text_agent      import TextAgent
from agents.supervisor      import SupervisorAgent
from xai.gradcam            import GradCAMAgent
from xai.shap_text          import SHAPTextAgent

# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title="Legal Document Authenticity Verifier API",
    description="Forensic-grade signature verification & text deception analysis",
    version="1.0.0"
)

# Allow React dev server (localhost:5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models ───────────────────────────────────────────────────
DEVICE = torch.device("cpu")
SAVED  = os.path.join(BASE_DIR, "models", "saved")

print("Loading models...")

siamese_model = SiameseNet(embedding_dim=128)
siamese_model.load_state_dict(
    torch.load(os.path.join(SAVED, "siamese_best.pt"), map_location=DEVICE)
)
siamese_model.eval()
print("[OK] Siamese CNN loaded")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
nlp_model  = DeceptionClassifier()
nlp_model.load_state_dict(
    torch.load(os.path.join(SAVED, "roberta_deception.pt"),
               map_location=DEVICE, weights_only=False)
)
nlp_model.eval()
print("[OK] RoBERTa loaded")

# ── Initialise agents ─────────────────────────────────────────────
prep_agent  = PreprocessingAgent()
sig_agent   = SignatureAgent(siamese_model, DEVICE)
txt_agent   = TextAgent(nlp_model, tokenizer, DEVICE)
sup_agent   = SupervisorAgent(sig_weight=0.6, text_weight=0.4, threshold=0.4)
cam_agent   = GradCAMAgent(siamese_model)
shap_agent  = SHAPTextAgent(nlp_model, tokenizer, DEVICE)

print("[OK] All agents ready\n")


# ── Image validation ──────────────────────────────────────────────
def is_valid_signature_image(img_path: str, label: str = "Image"):
    from PIL import Image as PILImage
    import numpy as np
    try:
        img = PILImage.open(img_path)
        w, h = img.size
        if w < 30 or h < 30:
            return False, f"{label} is too small ({w}x{h} px). Please upload a proper signature scan."
        if (w / h) < 0.3:
            return False, f"{label} looks like a portrait photo. Please upload a horizontal signature image."
        arr = np.array(img.convert("L"), dtype=float)
        if arr.std() < 5:
            return False, f"{label} appears blank or is a solid color. Please upload a signature image."
        return True, ""
    except Exception:
        return False, f"{label}: Could not read the file. Please upload a valid PNG or JPG image."


# ── PIL image → base64 string ─────────────────────────────────────
def pil_to_base64(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── /health ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}


# ── /verify ───────────────────────────────────────────────────────
@app.post("/verify")
async def verify(
    ref_signature:  UploadFile = File(..., description="Reference (known genuine) signature image"),
    test_signature: UploadFile = File(..., description="Test signature image under examination"),
    document_text:  str        = Form(..., description="Document text to analyse for deception"),
):
    # ── Validate text ────────────────────────────────────────────
    if not document_text or len(document_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Please enter more text (at least a full sentence).")

    # ── Save uploaded files to temp paths ────────────────────────
    tmp_dir = tempfile.mkdtemp()
    ref_path  = os.path.join(tmp_dir, "ref_sig.png")
    test_path = os.path.join(tmp_dir, "test_sig.png")

    with open(ref_path,  "wb") as f: f.write(await ref_signature.read())
    with open(test_path, "wb") as f: f.write(await test_signature.read())

    # ── Validate images ──────────────────────────────────────────
    ok, reason = is_valid_signature_image(ref_path, "Reference Signature")
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    ok, reason = is_valid_signature_image(test_path, "Test Signature")
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    # ── Run pipeline ─────────────────────────────────────────────
    try:
        ref_tensor  = prep_agent.prepare_signature(ref_path)
        test_tensor = prep_agent.prepare_signature(test_path)
        clean_text  = prep_agent.prepare_text(document_text)

        sig_score       = sig_agent.verify(ref_tensor, test_tensor)
        deception_score = txt_agent.analyze(clean_text)
        result          = sup_agent.decide(sig_score, deception_score)

        # Grad-CAM heatmap
        gradcam_img    = cam_agent.generate(test_tensor, test_path)
        heatmap_b64    = pil_to_base64(gradcam_img)

        # SHAP top words
        top_words = shap_agent.get_top_words(clean_text, n=5)
        shap_data = [{"word": w, "score": round(float(v), 4)} for w, v in top_words]

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        # Clean up temp files
        for p in [ref_path, test_path]:
            try: os.remove(p)
            except: pass
        try: os.rmdir(tmp_dir)
        except: pass

    return JSONResponse({
        "verdict":            result["verdict"],           # "AUTHENTIC" | "SUSPICIOUS"
        "signature_score":    round(result["signature_score"],   3),
        "deception_score":    round(result["deception_score"],   3),
        "combined_risk":      round(result["combined_risk"],     3),
        "signature_verdict":  result["signature_verdict"],  # "GENUINE" | "FORGED"
        "text_verdict":       result["text_verdict"],        # "TRUTHFUL" | "DECEPTIVE"
        "heatmap":            heatmap_b64,
        "shap_words":         shap_data,
        "thresholds": {
            "signature_genuine_min": 0.70,
            "text_deceptive_min":    0.45,
            "combined_suspicious_min": 0.40,
        }
    })
