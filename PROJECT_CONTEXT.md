# Legal Document Authenticity Verifier - Project Context

## Overview
This project is an AI-powered Legal Document Authenticity Verifier built as a Data Science final year project. It analyzes legal documents to ensure their integrity by checking two separate modalities: image (signature) and text (contract clauses).

## Current Architecture
The project is a production-grade full-stack web application:
*   **Frontend:** React (Vite) + Vanilla CSS. Features a premium glassmorphism aesthetic.
*   **Backend:** FastAPI running locally, serving PyTorch models.
*   **Communication:** Frontend sends multipart form data to FastAPI via Axios.

## Core Machine Learning Modules (Current State)

### 1. Signature Verification (Computer Vision)
*   **Model:** Siamese CNN with a VGG16 backbone.
*   **Training Data:** CEDAR Signature Dataset (genuine vs. forged pairs). Writer-independent split — trained on Signers 1–45, tested on Signers 46–55.
*   **Loss Function:** Contrastive Loss.
*   **Results:** Accuracy **80.21%** | AUC **0.9209**
*   **Explainability (XAI):** Uses **Grad-CAM** to generate a heatmap over the signature, highlighting which pen strokes triggered suspicion.
*   **Status:** Complete and deployed.

### 2. Text Analysis (NLP)
*   **Model:** RoBERTa-base — fully fine-tuned (all 125M parameters trainable).
*   **Training Data:** LexGLUE / UNFAIR-ToS dataset (`coastalcph/lex_glue`) — detects predatory and unfair legal clauses in Terms of Service contracts.
*   **Architecture:** RoBERTa-base + 3-layer MLP head with LayerNorm for gradient stability.
*   **Results:** Accuracy **95.83%** | Macro F1 **89.65%** | AUC **0.956**
*   **Explainability (XAI):** Uses **SHAP** token attribution to highlight words that influenced the prediction.
*   **Status:** Complete and deployed.

## Analysis Modes
The system supports three independent analysis modes:
1.  **Signature Verification Only** — checks if a signature is genuine or forged.
2.  **Contract Scan Only** — scans document text for unfair/predatory clauses.
3.  **Full Document Verification** — runs both pipelines and fuses results via the Supervisor Agent.

## Decision Fusion (Supervisor Agent)
```
sig_risk      = 1.0 - signature_similarity
text_risk     = unfair_clause_probability
combined_risk = (0.6 × sig_risk) + (0.4 × text_risk)

If combined_risk >= 0.40 → SUSPICIOUS
Else                     → AUTHENTIC
```
Signature gets **60% weight** as a forged physical signature is a stronger fraud indicator than text alone.

## Project Structure
```text
legal_doc_verifier/
├── frontend/             (React UI — Vite)
├── backend/              (FastAPI Server)
│   ├── api.py            (Main API entrypoint)
│   ├── models/           (Siamese CNN & RoBERTa classes & weights)
│   ├── agents/           (Inference pipeline & Supervisor Agent)
│   ├── xai/              (Grad-CAM & SHAP scripts)
│   ├── results/          (Training charts & test subjects)
│   └── requirements.txt  (Python dependencies)
└── colab_notebooks/      (Training scripts for Colab T4 GPU)
    ├── notebook1_siamese_training.py
    └── notebook2_unfair_clause_training.py
```
