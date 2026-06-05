# Legal Document Authenticity Verifier

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev/)

A multi-modal AI system that verifies the authenticity of legal contracts by performing physical and semantic validation in parallel. The system utilizes computer vision for signature verification (detecting skilled forgeries) and natural language processing for contract clause analysis (detecting unfair or predatory terms), fusing the outputs through a supervisor decision model.

---

## System Overview

This application verifies legal documents across two distinct modalities:

1. **Signature Verification (Physical Modality)**
   * A **Siamese CNN with a VGG16 backbone** trained using contrastive loss on the CEDAR dataset.
   * Compares a scanned test signature against a known reference signature to identify skilled forgeries.
   * Leverages **Grad-CAM** spatial heatmaps to highlight suspicious pen strokes.

2. **Contract Analysis (Semantic Modality)**
   * A **RoBERTa-base classifier** fully fine-tuned on the LexGLUE UNFAIR-ToS benchmark.
   * Scans document text to flag predatory, unfair, or non-compliant clauses (e.g., unilateral termination, forced arbitration, hidden liability exclusions).
   * Leverages **SHAP** token attribution to highlight the exact words triggering the warning.

A **Supervisor Agent** fuses both modality scores into a unified risk rating, determining whether a document is **AUTHENTIC** or **SUSPICIOUS**.

---

## Technical Architecture

```
                       [Scanned Signature & Contract Text]
                                        |
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
        [Signature Images]                              [Contract Text]
                 |                                             |
     [Preprocessing Agent]                           [Preprocessing Agent]
                 |                                             |
       [Signature CNN Agent]                           [Text RoBERTa Agent]
      (Siamese VGG16 Backbone)                      (Fine-Tuned Classifier)
                 |                                             |
         similarity_score                             unfair_clause_score
                 |                                             |
                 └──────────────────────┬──────────────────────┘
                                        ▼
                               [Supervisor Agent]
                         Combined Risk Fusion Formula:
                 (0.6 * sig_risk) + (0.4 * unfair_clause_score)
                                        |
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
            [Grad-CAM]                                      [SHAP]
    (Signature Stroke Heatmap)                     (Text Attribution Weights)
                 └──────────────────────┬──────────────────────┘
                                        ▼
                            [Unified UI Dashboard]
```

---

## Core Model Benchmarks

The system achieves competitive, robust benchmarks on standard datasets:

| Pipeline | Model Architecture | Metric | Score |
| :--- | :--- | :--- | :--- |
| **Signature Verification** | Siamese VGG16 (Contrastive Loss) | Accuracy | **80.21%** |
| **Signature Verification** | Siamese VGG16 (Contrastive Loss) | AUC | **0.9209** |
| **Contract Analysis** | RoBERTa-base (Fine-Tuned Classifier) | Accuracy | **95.83%** |
| **Contract Analysis** | RoBERTa-base (Fine-Tuned Classifier) | Macro F1 | **89.65%** |
| **Contract Analysis** | RoBERTa-base (Fine-Tuned Classifier) | AUC | **0.956** |

---

## Repository Structure

```text
legal-doc-verifier/
├── backend/                  # FastAPI Application
│   ├── api.py                # REST API endpoints & server setup
│   ├── agents/               # Multi-agent inference pipeline
│   ├── models/               # Model architectures and saved weights
│   │   ├── siamese_cnn.py    # Siamese CNN for signature verification
│   │   ├── roberta_nlp.py    # RoBERTa-base classifier for text
│   │   └── saved/            # Production model weights (.pt files)
│   ├── xai/                  # Explainable AI scripts (Grad-CAM, SHAP)
│   ├── data/                 # Sample test datasets & evaluation data
│   ├── results/              # Model evaluation metrics & validation charts
│   └── requirements.txt      # Backend Python dependencies
├── frontend/                 # React UI Client
│   ├── src/                  # React components, styles, and hooks
│   ├── index.html            # Entrypoint HTML
│   └── package.json          # Frontend dependencies & scripts
├── colab_notebooks/          # Training pipelines
│   ├── notebook1_siamese_training.py       # Siamese CNN training notebook
│   └── notebook2_unfair_clause_training.py  # RoBERTa fine-tuning notebook
└── README.md
```

---

## Datasets

The model weights are trained on peer-reviewed research benchmarks:

* **CEDAR Signatures**: Benchmark dataset for offline signature verification containing genuine and forged signatures.
* **LexGLUE UNFAIR-ToS**: Standard benchmark subset (from `coastalcph/lex_glue`) containing clauses annotated by legal experts for fairness across terms-of-service documents.

---

## Step-by-Step Setup

### 1. Install Dependencies

**Backend Setup:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend Setup:**
```bash
cd frontend
npm install
```

### 2. Download and Place Model Weights
To run local inference, place the trained weights in the corresponding directory:
* Download `siamese_best.pt` and place it in: `backend/models/saved/`
* Download `roberta_unfair_clause.pt` and place it in: `backend/models/saved/`

*Note: You can train these models from scratch using the scripts provided in `colab_notebooks/`.*

### 3. Running the Application

Open two terminal sessions:

**Start Backend API (FastAPI):**
```bash
cd backend
python -m uvicorn api:app --reload --port 8000
```

**Start Frontend Client (React + Vite):**
```bash
cd frontend
npm run dev
```

Navigate to `http://localhost:5173` to access the interactive web interface.

---

## Analysis & Decision Logic

The frontend web interface supports three verification workflows:
1. **Signature Only**: Compares a reference signature image with a query signature image.
2. **Contract Scan**: Analyzes legal text to detect and highlight predatory clauses.
3. **Full Verification**: Performs both tasks and fuses the results.

### Risk Score Fusion Formula
The Supervisor Agent calculates a weighted risk profile:
```text
sig_risk      = 1.0 - signature_similarity_score
unfair_score  = unfair_clause_probability_score

combined_risk = (0.6 * sig_risk) + (0.4 * unfair_score)
```
* **Decision Boundary**: A document is flagged as **SUSPICIOUS** if `combined_risk >= 0.40`; otherwise, it is labeled **AUTHENTIC**.
* **Rationale**: Physical signature integrity is weighted higher (60%) since forgery is an absolute indicator of fraud, whereas unfair clauses (40%) represent high semantic risk but do not necessarily mean the document was forged.

---

## Explainable AI (XAI) Engine

A critical requirement of AI-powered document verification is transparency:
* **Signature Heatmaps**: Using Grad-CAM, the backend hooks into the final convolutional layers of the Siamese towers to extract gradients relative to the embedding distance. A colorized spatial overlay is generated where red areas show the exact regions of pen hesitation or deviation that influenced the forgery rating.
* **Text Attribution**: Using SHAP, the system runs local perturbations by masking input tokens, showing which terms and contract phrases contributed most toward the "Unfair" class classification.

---

## License & Disclaimer
This project is intended for research, education, and screening purposes. It is not a substitute for professional forensic document analysis or formal legal counsel.
