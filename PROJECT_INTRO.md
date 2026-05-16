# Legal Document Authenticity Verifier
## Project Introduction

---

### Project Title

**AI-Powered Legal Document Authenticity Verifier**
*A Multi-Modal Forensic Analysis System for Signature Verification and Unfair Clause Detection*

---

### One-Line Description

> An intelligent system that verifies whether a legal document is authentic by simultaneously checking if its signature is genuine and whether its text contains predatory or unfair clauses — using deep learning and explainable AI.

---

### What Problem Does This Solve?

Legal document fraud is a serious and growing problem. Two of the most common attack vectors are:

1. **Forged Signatures** — A fraudster traces or copies someone's signature onto a contract they never agreed to.
2. **Unfair / Hidden Clauses** — A document looks legitimate on the surface but contains predatory legal language buried in the fine print — unilateral termination, forced arbitration, liability waivers — that strip the signing party of their rights.

Traditional verification is slow, expensive, and requires human forensic experts. This system automates both checks simultaneously and provides an explainable report in seconds.

---

### What Does This Project Do?

The system takes two inputs from the user:
- **Two signature images** (a known reference + the signature to be verified)
- **The contract text** (pasted directly into the interface)

It then runs both through independent AI pipelines and produces a single unified verdict:

| Output | Description |
|---|---|
| **AUTHENTIC** | Signature is genuine AND no unfair clauses detected |
| **SUSPICIOUS** | Forged signature OR predatory clauses found (or both) |
| **Signature Score** | 0.0 (forged) to 1.0 (genuine) — with Grad-CAM heatmap |
| **Unfair Clause Score** | 0.0 (safe) to 1.0 (unfair) — with SHAP word highlights |

The user can also run each pipeline independently:
- **Signature Only** — just verify the signature
- **Contract Scan Only** — just scan the text for unfair clauses
- **Full Verification** — both together (default)

---

### How Does It Work? (Architecture Overview)

```
User Input
  |
  +-- Signature Images -----> PreprocessingAgent (resize, normalize)
  |                                  |
  |                           SignatureAgent (Siamese CNN)
  |                                  |
  |                           similarity score (0.0 - 1.0)
  |                                  |
  |                           GradCAMAgent (heatmap on suspicious strokes)
  |
  +-- Contract Text -------> PreprocessingAgent (clean, strip whitespace)
                                     |
                              TextAgent (RoBERTa NLP)
                                     |
                              unfair clause score (0.0 - 1.0)
                                     |
                              SHAPAgent (highlight risky words)
  |
  +-- Both scores ---------> SupervisorAgent
                                     |
                              combined_risk = (0.6 x sig_risk) + (0.4 x unfair_score)
                                     |
                              Final Verdict: AUTHENTIC or SUSPICIOUS
                                     |
                             React Frontend (full report displayed)
```

---

### The Two AI Models

#### Model 1 — Siamese CNN (Signature Verification)

| Property | Detail |
|---|---|
| Architecture | Siamese Network with VGG16 backbone (ImageNet pre-trained) |
| Training Data | CEDAR Signature Dataset — 55 writers, 24 genuine + 24 forged each |
| Training Split | Writer-independent: Signers 1-45 train, Signers 46-55 test |
| Loss Function | Contrastive Loss |
| How it works | Converts each signature into a 128-dimensional embedding vector. Genuine pairs have similar vectors; forged pairs have distant vectors. Score = exp(-distance). |
| Threshold | score >= 0.70 -> GENUINE, score < 0.70 -> FORGED |
| **Accuracy** | **80.21%** |
| **AUC** | **0.9209 (Outstanding)** |
| Explainability | Grad-CAM heatmap highlights suspicious pen strokes |

#### Model 2 — RoBERTa Unfair Clause Detector (NLP)

| Property | Detail |
|---|---|
| Architecture | RoBERTa-base (125M parameters) + 3-layer MLP classifier head |
| Training Data | LexGLUE UNFAIR-ToS (coastalcph/lex_glue on HuggingFace) |
| Task | Binary classification: SAFE (0) vs UNFAIR (1) |
| Fine-tuning | Full fine-tuning — all 125M parameters trainable |
| Handles imbalance | Weighted CrossEntropyLoss (~9:1 SAFE:UNFAIR ratio in dataset) |
| Threshold | unfair_score >= 0.45 -> UNFAIR CLAUSES DETECTED |
| **Accuracy** | **95.83%** |
| **Macro F1** | **89.65%** |
| **AUC** | **0.956** |
| Explainability | SHAP token attribution highlights the exact words that triggered the flag |

---

### Does This Project Make Sense?

**Yes — and here is why it is defensible:**

1. **Real-world need**: Document fraud costs billions annually. Law firms, banks, and notaries currently rely on slow manual verification. Automated screening tools have genuine market value.

2. **Technically sound**: Both models are built on peer-reviewed, published methodologies. The CEDAR dataset and LexGLUE benchmark are standard academic benchmarks. The results are comparable to published research.

3. **Multi-modal fusion is the right approach**: Neither model alone is sufficient. A document can have a genuine signature but predatory text (so text-only fails). It can have clean text but a forged signature (so text-only fails again). The Supervisor Agent fuses both into one robust verdict.

4. **Explainability is built-in**: The system does not just say "SUSPICIOUS" — it shows *where* in the signature and *which words* in the text triggered the flag. This is a requirement of the EU AI Act for high-risk AI systems.

5. **Both models are confirmed working**: 
   - RoBERTa: 8/10 (80%) on manual short clause tests, 5/5 on long clause tests
   - Siamese CNN: 53/60 (88.3%) on a writer-independent spot test (signers 46-55)

---

### Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite, Vanilla CSS (glassmorphism UI) |
| Backend API | FastAPI + Uvicorn (Python) |
| Signature Model | PyTorch — Siamese CNN (VGG16 backbone) |
| Text Model | HuggingFace Transformers — RoBERTa-base |
| Explainability | Grad-CAM (manual hooks) + SHAP |
| Training Platform | Google Colab (T4 GPU) |
| Communication | Axios multipart/form-data (frontend -> backend) |

---

### Where Are the Documents?

| File | Purpose |
|---|---|
| `README.md` | Setup guide, how to run, datasets, decision logic |
| `PROJECT_CONTEXT.md` | Quick technical reference — architecture, model specs, analysis modes |
| `PROJECT_DEFENSE.md` | Q&A answers prepared for the project defense presentation |
| `PROJECT_REPORT.md` | Full detailed technical report — datasets, architecture, results, evaluator Q&A |
| `PROJECT_INTRO.md` | **This file** — project overview, title, description, and high-level explanation |
| `backend/results/test_subjects/test_clauses.md` | RoBERTa manual test results on 10 clauses |
| `backend/results/test_subjects/siamese_inference_test.md` | Siamese CNN live inference test results |

---

### Project Status

| Component | Status |
|---|---|
| Siamese CNN training | Complete — siamese_best.pt saved |
| RoBERTa training | Complete — roberta_deception.pt saved |
| FastAPI backend | Complete — all agents wired up |
| React frontend | Complete — 3 analysis modes, glassmorphism UI |
| Grad-CAM XAI | Complete — heatmap overlay on signature |
| SHAP XAI | Complete — top-5 words highlighted |
| Model testing | Complete — both models verified working |
| Documentation | Complete — all 5 docs updated and accurate |

**The project is fully production-ready for the semester defense.**
