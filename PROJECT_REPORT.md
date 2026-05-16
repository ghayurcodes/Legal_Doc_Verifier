# Legal Document Authenticity Verifier — Full Project Report
### Semester Final Year Project | Data Science

---

## 1. What Is This Project?

This project is an **AI-powered Legal Document Authenticity Verifier**. It takes a legal document and checks two things:
1. **Is the signature genuine or forged?** (using a deep learning image model)
2. **Does the contract contain unfair or predatory clauses?** (using a fine-tuned legal NLP model)

It then combines both results and gives a final verdict: **AUTHENTIC** or **SUSPICIOUS**.

The system also **explains its decision** — it shows a heatmap of which parts of the signature looked suspicious, and which specific words in the text pushed it toward "unfair."

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
              sig_score              unfair_score
         (0=forged, 1=genuine)   (0=safe, 1=unfair)
                      ↓              ↓
               ┌──────────────────────────────┐
               │      Supervisor Agent         │
                │  Combined Risk =              │
                │  (0.6 × sig_risk) +           │
                │  (0.4 × unfair_score)         │
                │  If risk ≥ 0.40 → SUSPICIOUS  │
               └──────────────────────────────┘
                              ↓
               ┌──────────────────────────────┐
               │         XAI Agents            │
               │  Grad-CAM → heatmap on sig    │
               │  SHAP → top words in text     │
               └──────────────────────────────┘
                              ↓
                      Final Report (React UI)
```

---

## 3. Datasets Used

### CEDAR (Signature Verification)
- **Full name:** Center of Excellence for Document Analysis and Recognition
- **Source:** University at Buffalo, USA
- **What it contains:** Real and forged handwritten signatures from 55 different people (writers)
- **Size:** 2,640 signature images (24 genuine + 24 forged per writer)
- **Why used:** Industry-standard benchmark for offline signature verification research

### LexGLUE UNFAIR-ToS (Unfair Clause Detection)
- **Source:** `coastalcph/lex_glue` on HuggingFace — created by researchers at the University of Copenhagen
- **What it contains:** Thousands of sentences extracted from real Terms of Service contracts across major platforms, annotated by legal experts into 8 unfair clause categories
- **Labels:** Binary — SAFE (no unfair annotations) vs. UNFAIR (any unfair annotation present)
- **Class imbalance:** Approximately 9:1 SAFE:UNFAIR ratio, handled using Weighted CrossEntropyLoss
- **Why used:** Peer-reviewed legal NLP benchmark — the standard dataset for contract clause fairness research

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
- **Similarity formula:** `score = exp(−distance)` — maps distance to 0.0–1.0 range:
  - Same signature uploaded twice → score ≈ **1.0**
  - Genuine pair (same person, different signature) → score ≈ **0.75–0.95**
  - Forged pair (different person) → score ≈ **0.30–0.65**
  - Threshold: score ≥ 0.70 → GENUINE, score < 0.70 → FORGED
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

## 5. Model 2 — RoBERTa Text Classifier (Unfair Clause Detection)

### What is RoBERTa?
RoBERTa (Robustly Optimized BERT Pretraining Approach) is a **transformer-based language model** developed by Facebook AI in 2019. It was pre-trained on 160GB of text from the internet. It understands the meaning of words in context (not just individual words).

### Is this Machine Learning or Deep Learning?
**Deep Learning.** RoBERTa is a deep neural network with 12 transformer layers, 12 attention heads, and 125 million parameters.

### Architecture
- **Base model:** roberta-base (125M parameters)
- **Our modification:** Removed RoBERTa's original language model head. Added our own 3-layer classifier head (SAFE / UNFAIR)
- **Classifier head:** Linear(768 → 512) → LayerNorm → ReLU → Dropout(0.3) → Linear(512 → 128) → LayerNorm → ReLU → Dropout(0.15) → Linear(128 → 2)
- **Input:** Contract/clause text, tokenized to max **256 tokens**
- **Inference threshold:** 0.45 (tuned for higher recall on the UNFAIR class)

### Which layers were frozen/unfrozen?
**All layers were fine-tuned (nothing frozen).** This is called "full fine-tuning" — all 125M RoBERTa parameters were trainable. This gives the best accuracy on legal domain text.

### Training Setup
- **Dataset:** LexGLUE UNFAIR-ToS (`coastalcph/lex_glue`) — ~9:1 SAFE:UNFAIR class imbalance
- **Loss function:** Weighted CrossEntropyLoss — higher weight given to the UNFAIR class to compensate for the severe imbalance
- **Optimizer:** AdamW — Adam with decoupled weight decay, standard for transformers
- **Learning rate:** 2e-5 — must be very low for full fine-tuning to avoid "catastrophic forgetting"
- **Scheduler:** Linear Warmup + Cosine Decay — LR warms up then smoothly decays
- **Epochs:** 10 — best checkpoint saved on peak Macro F1

### Results
- **Accuracy:** 95.83%
- **Macro F1:** 89.65%
- **AUC:** 0.956

---

## 6. Why Are These Accuracies Strong?

### Signature (80.21%, AUC 0.92):
Published CEDAR research papers report 75–85% under writer-independent evaluation. Our 80.21% is right in the middle of the published range. AUC of 0.92 is classified as "Outstanding" in the forensic AI literature.

### Text (95.83%, Macro F1 89.65%, AUC 0.956):
The LexGLUE UNFAIR-ToS benchmark is a peer-reviewed legal NLP dataset. Our results:
| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline (majority class) | ~90% | ~47% |
| BERT fine-tuned (published) | ~92% | ~82% |
| **Our RoBERTa (full fine-tune)** | **95.83%** | **89.65%** |

High accuracy alone is misleading due to class imbalance (~9:1). The **Macro F1 of 89.65%** is the real performance indicator — it weights both classes equally and confirms the model genuinely detects UNFAIR clauses, not just predicting SAFE every time.

---

## 7. XAI — Explainable AI

### Grad-CAM (for signatures)
- **Full name:** Gradient-weighted Class Activation Mapping
- **How it works — step by step:**
  1. Feed the test signature image through the VGG16 tower (forward pass)
  2. Compute a "score" — we use the magnitude (norm) of the 128-dim embedding vector
  3. Run backpropagation from that score — this asks: *"which neurons in the last conv layer caused this output?"*
  4. The gradients tell us how much each feature map channel contributed to the output
  5. Average gradients across spatial dimensions → one importance weight per channel
  6. Multiply those weights by the activation maps (feature maps at that layer)
  7. Sum across all channels → one 2D spatial importance map
  8. Apply ReLU — only keep positive contributions (negative = irrelevant)
  9. Resize the map back to match the original image size
  10. Apply COLORMAP_JET: blue = low attention, yellow = medium, red = high attention
  11. Blend with original image: 60% original + 40% heatmap
- **Output:** A colored overlay — **red regions = the model focused here most** when analysing this signature
- **Why it sometimes looks unclear or flat:**
  - The Siamese network is not a classifier — it computes similarity, not a class probability. We backpropagate through the embedding norm, which is an approximation. If the model is very confident or the signature is very uniform, gradients flatten out and the heatmap looks evenly coloured.
  - This is a known limitation of applying Grad-CAM to Siamese/metric learning networks. It still provides useful visual guidance for most signatures.

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
text_risk    = unfair_clause_probability     (safe=low, unfair=high)

combined_risk = (0.6 × sig_risk) + (0.4 × text_risk)

if signature_similarity < 0.70  →  individually labelled FORGED
if text_risk            ≥ 0.45  →  individually labelled UNFAIR CLAUSES DETECTED
if combined_risk        ≥ 0.40  →  final verdict SUSPICIOUS
else                             →  AUTHENTIC
```

**Why 60/40 split?** A forged physical signature is a stronger indicator of document fraud than unfair text clauses alone. A contract can contain harsh-but-legal clauses without being fraudulent.

---

## 9. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.13 |
| Deep Learning Framework | PyTorch 2.x |
| Image Model | VGG16 (torchvision) |
| NLP Model | RoBERTa (HuggingFace Transformers) |
| XAI | Grad-CAM (manual hooks) + SHAP |
| Web Interface | React + Vite (Frontend) |
| API Server | FastAPI (Backend) |
| Training Platform | Google Colab (T4 GPU) |

---

## 10. Common Evaluator Questions & Answers

**Q: Is this supervised or unsupervised learning?**
A: Both models are supervised — the training data has labels (genuine/forged for signatures, SAFE/UNFAIR for text clauses).

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
A: (1) Train on larger signature datasets, (2) Use a ViT (Vision Transformer) instead of VGG16, (3) Try larger RoBERTa-large instead of roberta-base, (4) Expand training to more legal clause categories beyond SAFE/UNFAIR binary.

**Q: What about skilled forgeries — if a forger copies a signature well, won't it score high and fool the system?**
A: This is a real and well-known challenge in the field. A forger copies what they *see* — the overall shape. But the Siamese CNN compares 128-dimensional embedding vectors that encode micro-level stroke patterns invisible to the human eye: stroke curvature at the pixel level, pen pressure distribution, how strokes begin and end, relative proportions between letters. A forger produces different micro-patterns even when the overall shape looks the same. The CEDAR dataset specifically uses *skilled* forgeries — forgers who practiced before the final attempt. The model achieves 80.21% accuracy on those skilled forgeries. The remaining ~20% that fool the system is why this is a screening tool, not a final verdict — a human forensic document examiner investigates further.

**Q: How does the Grad-CAM heatmap work?**
A: Grad-CAM (Gradient-weighted Class Activation Mapping) runs a backward pass through the network. It asks: *which spatial regions of the image caused the strongest gradient signal in the last convolutional layer?* Those regions are highlighted in red. It blends the heatmap (40%) with the original image (60%) so you can see exactly which pen strokes the model focused on. Sometimes the heatmap looks flat — this happens because Siamese networks compute similarity, not classification, so backpropagating through the embedding norm is an approximation.

**Q: Why does the text model sometimes miss unfair clauses?**
A: The model misses approximately 16% of unfair clauses (28 false negatives out of 172 total UNFAIR clauses in the test set). These are typically **short, subtle clauses** where predatory language is embedded in otherwise neutral-sounding sentences. The inference threshold is set at 0.45 (instead of 0.5) to increase recall on the UNFAIR class, accepting slightly more false positives. The Macro F1 of 89.65% reflects this trade-off accurately.

**Q: What does the similarity score mean? Why are the values close together?**
A: The score is computed as `exp(−euclidean_distance)` between the two signature embeddings. Identical signatures score ≈1.0. Genuine pairs from the same person score ~0.75–0.95. Forged pairs score ~0.30–0.65. The system uses a 0.70 threshold — anything below that is classified as FORGED.

**Q: Can you modify the model architecture files (.py) to change the output?**
A: No. The `.py` files only define the layer structure (blueprint). The trained weights — the actual learned knowledge — are in the `.pt` files. Changing the architecture without retraining would cause a shape mismatch crash. Only the decision thresholds in the supervisor can be tuned post-training.

---

*This document covers the full technical scope of the Legal Document Authenticity Verifier project.*
*Signature Model: Siamese CNN (VGG16) — CEDAR Dataset | Text Model: RoBERTa-base — LexGLUE UNFAIR-ToS | Interface: React + FastAPI*
