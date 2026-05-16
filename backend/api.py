"""
Legal Document Authenticity Verifier — FastAPI Backend
Run:  cd backend && python -m uvicorn api:app --reload --port 8000
"""

import sys
import os
import io
import base64
import tempfile
import traceback
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import torch
from transformers import RobertaTokenizer, logging as hf_logging

# ── Silence noisy HuggingFace warnings ──────────────────────────
hf_logging.set_verbosity_error()

# ── Path setup ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.siamese_cnn    import SiameseNet
from models.roberta_nlp    import UnfairClauseClassifier
from agents.preprocessing  import PreprocessingAgent
from agents.signature_agent import SignatureAgent
from agents.text_agent      import TextAgent
from agents.supervisor      import SupervisorAgent
from xai.gradcam            import GradCAMAgent
from xai.shap_text          import SHAPTextAgent

# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title="Legal Document Authenticity Verifier API",
    description="Forensic-grade signature verification & unfair clause analysis",
    version="2.0.0"
)

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
nlp_model  = UnfairClauseClassifier()
nlp_model.load_state_dict(
    torch.load(os.path.join(SAVED, "roberta_unfair_clause.pt"),
               map_location=DEVICE, weights_only=False)
)
nlp_model.eval()
print("[OK] RoBERTa Unfair Clause Detector loaded")

# ── Initialise agents ─────────────────────────────────────────────
prep_agent  = PreprocessingAgent()
sig_agent   = SignatureAgent(siamese_model, DEVICE)
txt_agent   = TextAgent(nlp_model, tokenizer, DEVICE)
sup_agent   = SupervisorAgent(sig_weight=0.6, text_weight=0.4, threshold=0.4)
cam_agent   = GradCAMAgent(siamese_model)
shap_agent  = SHAPTextAgent(nlp_model, tokenizer, DEVICE)

print("[OK] All agents ready\n")


# ── Helpers ───────────────────────────────────────────────────────
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


def pil_to_base64(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── /health ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running", "version": "2.0.0"}


# ── /verify ── Supports 3 modes ───────────────────────────────────
#   mode = "signature_only"  → only CV pipeline runs
#   mode = "text_only"       → only NLP pipeline runs
#   mode = "combined"        → both pipelines run (default)
@app.post("/verify")
async def verify(
    analysis_mode:  str            = Form("combined"),
    ref_signature:  Optional[UploadFile] = File(None),
    test_signature: Optional[UploadFile] = File(None),
    document_text:  Optional[str]        = Form(None),
):
    VALID_MODES = {"signature_only", "text_only", "combined"}
    if analysis_mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Choose from: {VALID_MODES}")

    needs_sig  = analysis_mode in ("signature_only", "combined")
    needs_text = analysis_mode in ("text_only", "combined")

    # ── Validate inputs based on mode ────────────────────────────
    if needs_sig:
        if not ref_signature or not test_signature:
            raise HTTPException(status_code=400, detail="Please upload both signature images.")
    if needs_text:
        if not document_text or len(document_text.strip()) < 20:
            raise HTTPException(status_code=400, detail="Please enter at least one full sentence of text (20+ chars).")

    tmp_dir = tempfile.mkdtemp()
    ref_path = test_path = None

    try:
        # ── Save and validate images (if needed) ─────────────────
        if needs_sig:
            ref_path  = os.path.join(tmp_dir, "ref_sig.png")
            test_path = os.path.join(tmp_dir, "test_sig.png")
            with open(ref_path,  "wb") as f: f.write(await ref_signature.read())
            with open(test_path, "wb") as f: f.write(await test_signature.read())

            ok, reason = is_valid_signature_image(ref_path, "Reference Signature")
            if not ok:
                raise HTTPException(status_code=400, detail=reason)
            ok, reason = is_valid_signature_image(test_path, "Test Signature")
            if not ok:
                raise HTTPException(status_code=400, detail=reason)

        # ── Run selected pipeline ────────────────────────────────
        sig_score      = None
        sig_verdict    = None
        unfair_score   = None
        text_verdict   = None
        heatmap_b64    = None
        shap_data      = []
        combined_risk  = None
        verdict        = None

        if needs_sig:
            ref_tensor  = prep_agent.prepare_signature(ref_path)
            test_tensor = prep_agent.prepare_signature(test_path)
            sig_score   = sig_agent.verify(ref_tensor, test_tensor)
            sig_verdict = "GENUINE" if sig_score >= 0.70 else "FORGED"

            gradcam_img = cam_agent.generate(test_tensor, test_path)
            heatmap_b64 = pil_to_base64(gradcam_img)

        if needs_text:
            clean_text   = prep_agent.prepare_text(document_text)
            unfair_score = txt_agent.analyze(clean_text)
            text_verdict = "UNFAIR CLAUSES DETECTED" if unfair_score >= 0.45 else "SAFE"
            top_words    = shap_agent.get_top_words(clean_text, n=5)
            shap_data    = [{"word": w, "score": round(float(v), 4)} for w, v in top_words]

        # ── Supervisor decision ───────────────────────────────────
        if analysis_mode == "combined":
            result        = sup_agent.decide(sig_score, unfair_score)
            verdict       = result["verdict"]
            combined_risk = round(result["combined_risk"], 3)
        elif analysis_mode == "signature_only":
            verdict       = "AUTHENTIC" if sig_verdict == "GENUINE" else "SUSPICIOUS"
            combined_risk = round(1.0 - sig_score, 3)
        else:  # text_only
            verdict       = "AUTHENTIC" if text_verdict == "SAFE" else "SUSPICIOUS"
            combined_risk = round(unfair_score, 3)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        for p in [ref_path, test_path]:
            if p:
                try: os.remove(p)
                except: pass
        try: os.rmdir(tmp_dir)
        except: pass

    return JSONResponse({
        "verdict":           verdict,
        "analysis_mode":     analysis_mode,
        "signature_score":   round(sig_score,    3) if sig_score    is not None else None,
        "unfair_score":      round(unfair_score, 3) if unfair_score is not None else None,
        "combined_risk":     combined_risk,
        "signature_verdict": sig_verdict,
        "text_verdict":      text_verdict,
        "heatmap":           heatmap_b64,
        "shap_words":        shap_data,
        "thresholds": {
            "signature_genuine_min":    0.70,
            "unfair_clause_min":        0.45,
            "combined_suspicious_min":  0.40,
        }
    })
