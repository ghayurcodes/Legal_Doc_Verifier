# Legal Document Authenticity Verifier
**Semester Project — Data Science**
Stack: Python · PyTorch · HuggingFace Transformers · React · FastAPI · Google Colab

---

## What This Project Does

This system takes a legal document and verifies two things simultaneously:

1. **Is the signature genuine or forged?**
   → Siamese CNN with VGG16 backbone trained using contrastive loss

2. **Are the written claims truthful or deceptive?**
   → RoBERTa-base classifier fine-tuned on the LIAR dataset

A **Supervisor Agent** combines both verdicts into one authenticity report with:
- **Grad-CAM** heatmap showing suspicious signature strokes
- **SHAP** token attribution showing deceptive words

Everything runs behind a **React SPA** communicating with a **FastAPI backend**.

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
│   │       └── roberta_deception.pt
│   ├── agents/                ← Inference pipeline agents
│   ├── xai/                   ← Explainability scripts (Grad-CAM, SHAP)
│   ├── data/                  ← Test datasets
│   └── requirements.txt       ← Backend dependencies
├── frontend/                  ← React user interface
│   ├── src/
│   ├── index.html
│   └── package.json
├── colab_notebooks/
│   ├── notebook1_siamese_training.py
│   └── notebook2_roberta_training.py
└── README.md
```

---

## Datasets

| Dataset | Use | Source |
|---|---|---|
| CEDAR Signatures | Train Siamese CNN | Kaggle — search "CEDAR signature" |
| LIAR Dataset | Train RoBERTa | https://www.cs.ucsb.edu/~william/data/liar_dataset.zip |

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

### Step 2 — Upload datasets to Google Drive
Upload these two zip files to `My Drive/ds_project/` in your Google Drive:
- `archive.zip` (CEDAR signature dataset)
- `liar_dataset.zip` (LIAR text dataset)

### Step 3 — Train Siamese CNN on Google Colab
1. Open [Google Colab](https://colab.research.google.com)
2. Set runtime: `Runtime > Change runtime type > T4 GPU`
3. Open `colab_notebooks/notebook1_siamese_training.py`
4. Create a new Colab notebook
5. Copy each **CELL** block into a separate Colab cell
6. Run cells top to bottom
7. **After training** — run Cell 8 immediately to save to Drive
8. Download `siamese_best.pt` from Drive → put in `models/saved/`

### Step 4 — Train RoBERTa on Google Colab
1. Start a new Colab session (fresh GPU)
2. Open `colab_notebooks/notebook2_roberta_training.py`
3. Same process — copy cells, run top to bottom
4. **After training** — run Cell 8 immediately to save to Drive
5. Download `roberta_deception.pt` from Drive → put in `models/saved/`

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

The web UI has **3 inputs**:

| Input | What to provide |
|---|---|
| Reference Signature | A known genuine signature image (PNG/JPG) from `data/cedar/full_org/` |
| Test Signature | The signature to verify — use `full_org/` for genuine or `full_forg/` for forged |
| Document Text | Paste any text — type something deceptive or truthful |

Click **Verify Document Authenticity** → see the report.

---

## How the Decision is Made

```
sig_score       = Siamese CNN output  (0.0=forged → 1.0=genuine)
deception_score = RoBERTa output      (0.0=truthful → 1.0=deceptive)

sig_risk      = 1.0 - sig_score       (flip: genuine=low risk)
combined_risk = (0.6 × sig_risk) + (0.4 × deception_score)

If combined_risk >= 0.5  →  SUSPICIOUS
If combined_risk <  0.5  →  AUTHENTIC
```

Signature gets **60% weight** because a forged signature is a stronger fraud signal than deceptive text alone.

---

## Colab Training Schedule

| Session | Notebook | GPU Time |
|---|---|---|
| Session 1 | Notebook 1 (HPO + Siamese CNN) | ~1.5-2 hours |
| Session 2 | Notebook 2 (RoBERTa) | ~30-40 mins |

> **Important:** Always run the final save cell before your Colab session expires. Unsaved models are permanently lost when the session ends.

---

## What to Say to Your Professor

### 1. Siamese CNN with Contrastive Loss
> "Instead of a standard classifier, I built a Siamese network with two weight-sharing towers.
> The model learns an embedding space where genuine signature pairs cluster together and forged pairs are pushed apart.
> The contrastive loss function encodes this directly — it minimizes distance for genuine pairs and maximizes it for forged pairs."

### 2. Transfer Learning with Layer Freezing
> "I froze the first 24 layers of VGG16 which already know edges and textures from ImageNet training,
> and only trained the last conv block and my custom 3-layer head.
> This is the fine-tuning strategy that avoids overfitting on a small dataset."

### 3. Supervisor Agent with Dual XAI
> "The supervisor agent combines scores from two completely independent models using a weighted decision rule.
> Grad-CAM shows which stroke regions in the signature triggered the forgery flag.
> SHAP shows which words in the text drove the deception score.
> Both are visible side by side in the web UI."

---

## Requirements

See `requirements.txt` for the full list. Key libraries:
- `torch`, `torchvision` — deep learning
- `transformers` — RoBERTa
- `shap` — text explainability
- `opencv-python` — Grad-CAM image processing
- `optuna` — hyperparameter optimization (Colab only)
- `fastapi`, `uvicorn` — REST API backend
- `react`, `vite` — frontend framework
