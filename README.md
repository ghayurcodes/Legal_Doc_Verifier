# Legal Document Authenticity Verifier
**Final Year Data Science Project**
Stack: Python · PyTorch · HuggingFace Transformers · React · FastAPI · Google Colab

---

## What This Project Does

This system takes a legal document and verifies two things simultaneously:

1. **Is the signature genuine or forged?**
   → Siamese CNN with VGG16 backbone trained using contrastive loss on the CEDAR dataset

2. **Does the contract contain unfair or predatory clauses?**
   → RoBERTa-base fine-tuned on the LexGLUE UNFAIR-ToS legal benchmark

A **Supervisor Agent** combines both verdicts into one authenticity report with:
- **Grad-CAM** heatmap showing suspicious signature strokes
- **SHAP** token attribution showing which words triggered the unfair clause flag

Everything runs behind a **React SPA** communicating with a **FastAPI backend**.

---

## Final Model Results

| Model | Metric | Score |
|---|---|---|
| Siamese CNN (Signature Verification) | Accuracy | **80.21%** |
| Siamese CNN (Signature Verification) | AUC | **0.9209** |
| RoBERTa (Unfair Clause Detection) | Accuracy | **95.83%** |
| RoBERTa (Unfair Clause Detection) | Macro F1 | **89.65%** |
| RoBERTa (Unfair Clause Detection) | AUC | **0.956** |

---

## Project Structure

```
legal_doc_verifier/
├── backend/
│   ├── api.py                 ← FastAPI server
│   ├── models/
│   │   ├── siamese_cnn.py     ← Siamese network architecture
│   │   ├── roberta_nlp.py     ← RoBERTa classifier architecture
│   │   └── saved/
│   │       ├── siamese_best.pt
│   │       └── roberta_unfair_clause.pt
│   ├── agents/                ← Inference pipeline agents
│   ├── xai/                   ← Explainability scripts (Grad-CAM, SHAP)
│   ├── data/                  ← Test datasets
│   ├── results/               ← Training charts & test clause subjects
│   └── requirements.txt       ← Backend dependencies
├── frontend/                  ← React user interface
│   ├── src/
│   ├── index.html
│   └── package.json
├── colab_notebooks/
│   ├── notebook1_siamese_training.py
│   └── notebook2_unfair_clause_training.py
└── README.md
```

---

## Datasets

| Dataset | Use | Source |
|---|---|---|
| CEDAR Signatures | Train Siamese CNN | Kaggle — search "CEDAR signature" |
| LexGLUE UNFAIR-ToS | Train RoBERTa | HuggingFace: `coastalcph/lex_glue` (unfair_tos subset) |

---

## Step-by-Step Setup

### Step 1 — Install Dependencies
**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### Step 2 — Upload CEDAR dataset to Google Drive
Upload the CEDAR signature zip file to `My Drive/ds_project/` in your Google Drive.

### Step 3 — Train Siamese CNN on Google Colab
1. Open [Google Colab](https://colab.research.google.com)
2. Set runtime: `Runtime > Change runtime type > T4 GPU`
3. Open `colab_notebooks/notebook1_siamese_training.py`
4. Create a new Colab notebook, copy each CELL block into a separate cell
5. Run cells top to bottom
6. After training — run the save cell to save to Drive
7. Download `siamese_best.pt` from Drive → put in `backend/models/saved/`

### Step 4 — Train RoBERTa on Google Colab
1. Start a new Colab session (fresh GPU)
2. Open `colab_notebooks/notebook2_unfair_clause_training.py`
3. The notebook auto-downloads the LexGLUE UNFAIR-ToS dataset from HuggingFace (no manual upload needed)
4. Same process — copy cells, run top to bottom
5. After training — run the save cell to save to Drive
6. Download `roberta_unfair_clause.pt` from Drive → put in `backend/models/saved/`

### Step 5 — Run the Application

You need two terminals running simultaneously.

**Terminal 1 — Backend:**
```bash
cd backend
python -m uvicorn api:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Open your browser to: `http://localhost:5173`

---

## How to Use the App

The web UI has **3 analysis modes**:

| Mode | What to provide |
|---|---|
| Signature Only | Upload reference (genuine) + test signature image (PNG/JPG) |
| Contract Scan | Paste contract text — the model flags unfair/predatory clauses |
| Full Document | Upload both signature images AND paste contract text |

Click **Verify** → see the full authenticity report.

---

## How the Decision is Made

```
sig_score     = Siamese CNN output  (0.0=forged → 1.0=genuine)
unfair_score  = RoBERTa output      (0.0=safe → 1.0=unfair)

sig_risk      = 1.0 - sig_score
combined_risk = (0.6 × sig_risk) + (0.4 × unfair_score)

If combined_risk >= 0.40  →  SUSPICIOUS
If combined_risk <  0.40  →  AUTHENTIC
```

Signature gets **60% weight** because a forged signature is a stronger fraud signal than predatory text alone.

---

## Colab Training Schedule

| Session | Notebook | GPU Time |
|---|---|---|
| Session 1 | Notebook 1 (Siamese CNN) | ~1.5–2 hours |
| Session 2 | Notebook 2 (RoBERTa UNFAIR-ToS) | ~45–60 mins |

> **Important:** Always run the save cell before your Colab session expires. Unsaved models are permanently lost when the session ends.

---

## What to Say to Your Professor

### 1. Siamese CNN with Contrastive Loss
> "Instead of a standard classifier, I built a Siamese network with two weight-sharing towers.
> The model learns an embedding space where genuine signature pairs cluster together and forged pairs are pushed apart.
> The contrastive loss function encodes this directly — it minimizes distance for genuine pairs and maximizes it for forged pairs."

### 2. RoBERTa for Legal Clause Detection
> "The RoBERTa model was fine-tuned on the LexGLUE UNFAIR-ToS dataset — a peer-reviewed legal NLP benchmark.
> It detects predatory clauses such as unilateral termination rights, liability waivers, and forced arbitration.
> The model achieves 95.83% accuracy and 89.65% Macro F1, which is competitive with published results on this benchmark."

### 3. Supervisor Agent with Dual XAI
> "The supervisor agent combines scores from two completely independent models using a weighted decision rule.
> Grad-CAM shows which stroke regions in the signature triggered the forgery flag.
> SHAP shows which words in the text drove the unfair clause score.
> Both are visible side by side in the web UI."

---

## Requirements

See `requirements.txt` for the full list. Key libraries:
- `torch`, `torchvision` — deep learning
- `transformers`, `datasets` — RoBERTa + HuggingFace data loading
- `shap` — text explainability
- `opencv-python` — Grad-CAM image processing
- `fastapi`, `uvicorn` — REST API backend
- `react`, `vite` — frontend framework
