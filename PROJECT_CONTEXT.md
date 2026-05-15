# Legal Document Authenticity Verifier - Project Context

## Overview
This project is an AI-powered Legal Document Authenticity Verifier built as a Data Science final year project. It analyzes legal documents to ensure their integrity by checking two separate modalities: image (signature) and text (contract clauses).

## Current Architecture
The project recently transitioned from a basic Gradio prototype to a production-grade full-stack web application:
*   **Frontend:** React (Vite) + Vanilla CSS. Features a premium "White-Gold/Dark-Navy" glassmorphism aesthetic.
*   **Backend:** FastAPI running locally, serving PyTorch models.
*   **Communication:** Frontend sends multipart form data to FastAPI via Axios.

## Core Machine Learning Modules (Current State)

### 1. Signature Verification (Computer Vision)
*   **Model:** Siamese CNN with a VGG16 backbone.
*   **Training Data:** CEDAR Signature Dataset (genuine vs. forged pairs).
*   **Loss Function:** Contrastive Loss.
*   **Explainability (XAI):** Uses **Grad-CAM** to generate a heatmap over the signature, highlighting exactly which pen strokes triggered suspicion.
*   **Status:** Excellent, keeping as is.

### 2. Text Analysis (NLP)
*   **Model:** RoBERTa-base (Sequence Classification).
*   **Current Training Data:** LIAR dataset (Political fact-checking).
*   **Explainability (XAI):** Uses **SHAP** token attribution to highlight words that influenced the prediction.
*   **Status:** *Pending replacement.* Using a political deception dataset on legal contracts is logically flawed.

## The "New Plan" (Planned Upgrades)
We are currently in the process of implementing two major upgrades to make the project logically sound and highly professional:

### Upgrade A: Swapping the NLP Dataset (LexGLUE)
Instead of political deception, the RoBERTa model will be retrained to detect **"Unfair / Predatory Legal Clauses."**
*   **New Dataset:** `coastalcph/lex_glue` (specifically the `unfair_tos` subset).
*   **How it works:** It reads the contract and flags toxic sentences (e.g., extreme liability waivers, hidden arbitration traps).
*   **Benefit:** Fits the theme perfectly. The CV model checks if the signature is fake; the NLP model acts as an AI lawyer checking if the contract terms are dangerous.

### Upgrade B: The 3-in-1 Modular Tool
Instead of forcing the user to upload both an image and text simultaneously, the system will be split into three modes:
1.  **Signature Verification Only:** (For checking checks/signed papers without long text).
2.  **Contract Scan Only:** (For checking a draft contract *before* signing it).
3.  **Full Document Verification:** (Checking both the signature and the text combined).

## Project Structure
```text
legal_doc_verifier/
├── frontend/             (React UI)
├── backend/              (FastAPI Server)
│   ├── api.py            (Main API entrypoint)
│   ├── models/           (Siamese CNN & RoBERTa classes & weights)
│   ├── agents/           (Inference pipeline & Supervisor Agent)
│   ├── xai/              (Grad-CAM & SHAP scripts)
│   └── requirements.txt  (Python dependencies)
└── colab_notebooks/      (Training scripts for Colab T4 GPU)
```

## Current Focus
The immediate next steps are to:
1. Modify the React UI to support the 3 modular tabs.
2. Update the FastAPI backend to handle partial requests (image-only or text-only).
3. Create a new Google Colab notebook to quickly fine-tune RoBERTa on the LexGLUE `unfair_tos` dataset.
