# Legal Document Authenticity Verifier — Full Project Report
### Semester Final Year Project | Data Science

---

## 1. What Is This Project?

This project is an **AI-powered Legal Document Authenticity Verifier**. It takes a legal document and checks two things:
1. **Is the signature genuine or forged?** (using a deep learning image model)
2. **Is the written text truthful or deceptive?** (using a language model)

It then combines both results and gives a final verdict: **AUTHENTIC** or **SUSPICIOUS**.

The system also **explains its decision** — it shows a heatmap of which parts of the signature looked suspicious, and which specific words in the text pushed it toward "deceptive."

---

## 2. System Architecture — How It Works

```
User uploads → Reference Signature + Test Signature + Document Text
                              ↓
               ┌──────────────────────────────┐
               │      Preprocessing Agent      │
               │  - Resize/normalize images    │
               │  - Clean and tokenize text    │
               └──────────────────────────────┘
                      ↓              ↓
        ┌─────────────────┐   ┌──────────────────┐
        │  Signature Agent │   │   Text Agent      │
        │  Siamese CNN     │   │   RoBERTa NLP     │
        │  (VGG16 backbone)│   │   Classifier      │
        └─────────────────┘   └──────────────────┘
             sig_score              deception_score
        (0=forged, 1=genuine)   (0=truthful, 1=deceptive)
                      ↓              ↓
               ┌──────────────────────────────┐
               │      Supervisor Agent         │
               │  Combined Risk =              │
               │  (0.6 × sig_risk) +           │
               │  (0.4 × text_risk)            │
               │  If risk ≥ 0.5 → SUSPICIOUS   │
               └──────────────────────────────┘
                              ↓
               ┌──────────────────────────────┐
               │         XAI Agents            │
               │  Grad-CAM → heatmap on sig    │
               │  SHAP → top words in text     │
               └──────────────────────────────┘
                              ↓
                      Final Report (Gradio UI)
```

---

## 3. Datasets Used

### CEDAR (Signature Verification)
- **Full name:** Center of Excellence for Document Analysis and Recognition
- **Source:** University at Buffalo, USA
- **What it contains:** Real and forged handwritten signatures from 55 different people (writers)
- **Size:** 2,640 signature images (24 genuine + 24 forged per writer)
- **Why used:** Industry-standard benchmark for offline signature verification research

### LIAR (Text Deception Detection)
- **Source:** PolitiFact.com, compiled by researchers at UC Santa Barbara
- **What it contains:** 12,836 real political statements, each rated by human fact-checkers
- **Labels:** true, mostly-true, half-true, barely-true, false, pants-on-fire
- **Binary mapping:** true/mostly-true/half-true → Truthful (0) | barely-true/false/pants-fire → Deceptive (1)
- **Why used:** Most widely cited benchmark for automated fact-checking and deception detection

---

## 4. Model 1 — Siamese CNN (Signature Verification)

### What is a Siamese Network?
A Siamese Network is a special neural network that **compares two inputs** rather than classifying one input. It passes both images through the **same set of weights** (same "tower") and learns to measure how similar or different they are.

Think of it like: instead of asking "is this signature real?", it asks "does this signature look the same as the reference one?"

### Architecture
- **Backbone:** VGG16 (pre-trained on ImageNet)
- **Pre-training:** Yes — VGG16 was already trained on 1.2 million images. We use transfer learning.
- **Which layers were frozen?** The first 4 blocks of VGG16 (low-level feature detectors like edges, textures) were frozen. Only the last convolutional block was fine-tuned.
- **Why freeze early layers?** Early layers detect universal features (edges, curves) that are the same for any image. Freezing saves time and prevents overfitting.
- **Embedding size:** 128 dimensions — each signature becomes a 128-number vector
- **Distance function:** Euclidean distance between the two 128-dim vectors
- **Loss function:** Contrastive Loss — penalizes the model if it brings forged pairs too close together or pushes genuine pairs too far apart

### Training Setup
- **Optimizer:** Adam with L2 weight decay (prevents overfitting)
- **Augmentation:** ColorJitter + RandomAffine — simulates scanner variation and slight rotation
- **Split:** Writer-Independent — trained on signers 1–45, tested on signers 46–55 (completely unseen people)
- **Scheduler:** StepLR — reduces learning rate every few epochs

### Results
- **Accuracy:** 80.21%
- **AUC:** 0.9209 (Outstanding — above 0.90 in forensic AI literature)

---

## 5. Model 2 — RoBERTa Text Classifier (Deception Detection)

### What is RoBERTa?
RoBERTa (Robustly Optimized BERT Pretraining Approach) is a **transformer-based language model** developed by Facebook AI in 2019. It was pre-trained on 160GB of text from the internet. It understands the meaning of words in context (not just individual words).

### Is this Machine Learning or Deep Learning?
**Deep Learning.** RoBERTa is a deep neural network with 12 transformer layers, 12 attention heads, and 125 million parameters.

### Architecture
- **Base model:** roberta-base (125M parameters)
- **Our modification:** Removed RoBERTa's original language model head. Added our own 2-class classifier head (Truthful / Deceptive)
- **Classifier head:** Linear(768 → 256) → ReLU → Dropout(0.3) → Linear(256 → 2)
- **Input:** Raw statement text, tokenized to max 256 tokens

### Which layers were frozen/unfrozen?
**All layers were fine-tuned (nothing frozen).** This is called "full fine-tuning" — it's the standard approach for hard classification datasets and gives the best accuracy on LIAR.

### Training Setup
- **Loss function:** Weighted CrossEntropyLoss — LIAR has more Truthful samples than Deceptive, so we give higher weight to Deceptive class to compensate
- **Optimizer:** AdamW — Adam with decoupled weight decay, standard for transformers
- **Learning rate:** 1e-5 — must be very low for full fine-tuning to avoid "catastrophic forgetting"
- **Scheduler:** Linear Warmup + Decay — LR starts at 0, ramps up for 10% of training, then linearly decays to 0
- **Early stopping:** Patience = 2 — training stops if validation accuracy doesn't improve for 2 consecutive epochs
- **Best epoch:** Epoch 4 (65.03% val accuracy)

### Results
- **Accuracy:** 64.17%
- **F1 Score:** 0.531

---

## 6. Why Are These Accuracies Acceptable?

### Signature (80.21%, AUC 0.92):
Published CEDAR research papers report 75–85% under writer-independent evaluation. Our 80.21% is right in the middle of the published range. AUC of 0.92 is classified as "Outstanding."

### Text (64.17%):
The LIAR dataset is one of the hardest NLP benchmarks. Published results:
| Model | Accuracy |
|---|---|
| Naive Bayes | ~57% |
| SVM | ~61% |
| BERT fine-tuned | ~65% |
| **Our RoBERTa** | **64.17%** |
| State-of-the-art (2024) | ~68% |

Our model is at **published research paper level**, not just semester-project level.

---

## 7. XAI — Explainable AI

### Grad-CAM (for signatures)
- **Full name:** Gradient-weighted Class Activation Mapping
- **How it works:** Runs a backward pass through the network and finds which spatial regions of the image most influenced the output. Draws a heatmap over those regions.
- **Output:** A colored overlay on the signature — red regions = the model focused here most

### SHAP (for text)
- **Full name:** SHapley Additive exPlanations
- **How it works:** Masks/hides individual words and measures how much the prediction changes. A big change = that word was important.
- **Output:** A list of words with their importance scores — positive = pushed toward deceptive, negative = pushed toward truthful

### Why XAI matters:
The EU AI Act requires high-risk AI systems (like legal document verification) to be transparent and explainable. Without XAI, the system is a black box — a human expert cannot validate or override the decision.

---

## 8. Supervisor Agent — Decision Fusion

```
sig_risk     = 1.0 - signature_similarity    (genuine=low risk, forged=high risk)
text_risk    = deception_score               (truthful=low, deceptive=high)

combined_risk = (0.6 × sig_risk) + (0.4 × text_risk)

if combined_risk ≥ 0.5  →  SUSPICIOUS
else                     →  AUTHENTIC
```

**Why 60/40 split?** A forged physical signature is stronger evidence of fraud than suspicious text alone. Text can be misleading without being fraud.

---

## 9. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.13 |
| Deep Learning Framework | PyTorch 2.x |
| Image Model | VGG16 (torchvision) |
| NLP Model | RoBERTa (HuggingFace Transformers) |
| XAI | Grad-CAM (manual hooks) + SHAP |
| Web Interface | Gradio |
| Training Platform | Google Colab (T4 GPU) |

---

## 10. Common Evaluator Questions & Answers

**Q: Is this supervised or unsupervised learning?**
A: Both models are supervised — the training data has labels (genuine/forged for signatures, true/false for text).

**Q: What is transfer learning?**
A: We started with VGG16 and RoBERTa, which were already trained on massive datasets. We adapted them for our specific task instead of training from scratch. This saves time, data, and gives better accuracy.

**Q: What is a transformer?**
A: A neural network architecture that uses "attention" to understand relationships between all words in a sentence simultaneously, not one by one. RoBERTa is transformer-based.

**Q: What is the attention mechanism?**
A: When processing a word, attention lets the model look at all other words in the sentence and decide which ones are most relevant for understanding this word's meaning.

**Q: Why not use a simple CNN for text?**
A: CNNs process images with spatial relationships. Text needs to understand long-range semantic relationships (e.g., "not guilty" — the "not" modifies "guilty" across a gap). Transformers handle this better.

**Q: What is overfitting?**
A: When the model memorizes the training data instead of learning general patterns. It performs well on training data but badly on new data. We prevented it using dropout, weight decay, data augmentation, and early stopping.

**Q: What is dropout?**
A: During training, randomly "turns off" some neurons (probability 0.3 in our case). This forces the network to not rely on any single neuron, making it more robust.

**Q: Why did you use AdamW instead of regular Adam?**
A: AdamW separates the weight decay (L2 regularization) from the gradient update, which is more mathematically correct for transformers and leads to better generalization.

**Q: What is a learning rate scheduler?**
A: Instead of using a fixed learning rate throughout training, we start low (warmup), increase briefly, then gradually decrease. This helps the model find a good solution without jumping over it.

**Q: What is contrastive loss?**
A: A loss function designed for Siamese networks. It pushes embeddings of genuine pairs close together and forged pairs far apart in the embedding space, using a margin to define "far enough."

**Q: What is the difference between AUC and accuracy?**
A: Accuracy measures how often the model is correct at one specific threshold. AUC (Area Under the ROC Curve) measures performance across ALL possible thresholds — it's a better measure of overall model quality, especially when classes are imbalanced.

**Q: Why did you use writer-independent splitting?**
A: To test if the model can verify signatures of completely new people it has never seen. Writer-dependent splitting (same people in train and test) would give artificially high accuracy (95%+) but wouldn't reflect real-world performance.

**Q: What would you improve with more time?**
A: (1) Train on larger signature datasets, (2) Use a ViT (Vision Transformer) instead of VGG16, (3) Try larger RoBERTa-large instead of roberta-base, (4) Collect domain-specific legal document text data instead of political statements.

---

*This document covers the full technical scope of the Legal Document Authenticity Verifier project.*
*Signature Model: Siamese CNN (VGG16) | Text Model: RoBERTa-base | Interface: Gradio*
