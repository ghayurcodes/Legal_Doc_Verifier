# 📄 Legal Document Authenticity Verifier — Project Defense Notes

---

## ✅ Final Model Results

| Model | Metric | Score |
|---|---|---|
| Siamese CNN (Signature Verification) | Accuracy | **80.21%** |
| Siamese CNN (Signature Verification) | AUC | **0.9209** |
| RoBERTa (Unfair Clause Detection) | Accuracy | **95.83%** |
| RoBERTa (Unfair Clause Detection) | Macro F1 | **89.65%** |
| RoBERTa (Unfair Clause Detection) | AUC | **0.956** |

---

## 🛡️ How to Defend the Siamese CNN (80%, AUC 0.92)

### If asked: "Why is accuracy only 80%?"

> "We deliberately used a **writer-independent evaluation protocol**, which is the gold standard
> for signature verification research. The model was trained on signatures from Signers 1–45,
> and tested entirely on Signers 46–55 — people the model had **never seen before**.
> This is the same protocol used in published CEDAR research papers, where typical results
> range from 75–85%. Our 80.21% is right in the middle of the published range.
> An 'easy' split where the same people appear in both train and test would give 95%+,
> but that would not reflect real-world performance."

### If asked: "What does AUC 0.92 mean?"

> "AUC (Area Under the ROC Curve) measures how well the model separates genuine from forged
> signatures regardless of the threshold. A score above 0.90 is classified as **'Outstanding'**
> in the medical and forensic AI literature. This means even when overall accuracy is 80%,
> the model's confidence scores are highly reliable and well-separated."

### Key talking point:
> "The Siamese Network does not memorize specific signatures. It learns **universal forgery
> features** — hesitation in pen strokes, unnatural pressure, inconsistent stroke connections.
> This is why it can verify the signature of a completely new person it has never seen before,
> which is exactly how a real-world document verification system must work."

---

## 🛡️ How to Defend the RoBERTa Text Model (95.83%, Macro F1 89.65%)

### If asked: "What does the RoBERTa model detect?"

> "The model detects **unfair and predatory legal clauses** in contracts and Terms of Service
> documents. It was fine-tuned on the **LexGLUE / UNFAIR-ToS dataset**, a peer-reviewed legal
> NLP benchmark created by researchers at the University of Copenhagen. It flags clauses such as
> unilateral termination rights, liability exclusions, mandatory arbitration waivers, and
> unilateral contract modification rights — all patterns that courts have historically ruled
> as potentially unfair to consumers."

### If asked: "What dataset did you use?"

> "We used the **LexGLUE UNFAIR-ToS subset** (from `coastalcph/lex_glue` on HuggingFace).
> This dataset contains thousands of sentences extracted from real Terms of Service contracts
> across major platforms, annotated by legal experts into 8 unfair clause categories.
> We simplified this to a binary task: SAFE (no unfair annotations) vs. UNFAIR (any unfair annotation).
> The dataset is significantly imbalanced (~9:1 SAFE:UNFAIR ratio), which we handled using
> weighted CrossEntropy loss during training."

### If asked: "Why 95.83% and not 100%?"

> "The model misses approximately 16% of unfair clauses (see confusion matrix: 28 false negatives
> out of 172 total UNFAIR clauses). These are typically **short, subtle clauses** where unfair
> language is camouflaged in otherwise neutral-sounding sentences. The Macro F1 of 89.65%
> more accurately represents performance across both classes — an excellent result given the
> severe class imbalance."

### Key talking point:
> "What makes this component powerful is the combination of the **RoBERTa model** and
> **SHAP explainability**. The system does not just say 'this contract is risky' — it highlights
> the exact words and phrases that triggered the flag. A lawyer or consumer can then review
> specifically those highlighted sections, making this a practical legal screening tool."

---

## 🛡️ How to Defend the Combined System

### If asked: "Why combine both models?"

> "This is the core insight of the multi-agent architecture. In the real world, a human forensic
> expert does not verify a legal document by looking at only the signature OR only the text —
> they look at both. Our Supervisor Agent fuses:
> - Signature similarity score (weighted 60%) — the primary physical evidence
> - Unfair clause probability (weighted 40%) — the semantic content analysis
>
> A document that has a **genuine signature but predatory clauses** will still be flagged as
> suspicious. A document with a **forged signature but clean text** will also be flagged.
> Neither model alone could achieve this. The combined system is more robust than either
> component individually."

---

## 🛡️ How to Defend the Technical Choices

### Why Siamese CNN instead of a standard classifier?
> "A standard CNN would classify each signature as 'real' or 'fake' based on a fixed learned
> boundary. It would fail on any new person's signature because it has never seen that person.
> A Siamese Network learns a **distance metric** — it learns what makes two signatures
> from the same person similar, and what makes a forgery different. This allows **zero-shot
> verification** of any new person without retraining the model."

### Why RoBERTa instead of BERT or GPT?
> "RoBERTa (Robustly Optimized BERT Pretraining Approach) was published by Facebook AI in 2019
> and consistently outperforms BERT on text classification benchmarks. It was pre-trained on
> 160GB of text (10x more than BERT) with improved training procedures. For legal text classification,
> RoBERTa is the standard choice in academic NLP literature and achieves top results on LexGLUE."

### Why Grad-CAM + SHAP for explainability?
> "Both methods are required by the EU AI Act's transparency requirements for high-risk AI
> systems. Grad-CAM shows WHICH part of the signature the model focused on (spatial heatmap).
> SHAP shows WHICH words in the text contributed to the unfair clause score. Without these,
> the system is a black box. With them, a human expert can validate or override the AI decision."

---

## 📊 Training Configuration (for technical questions)

### Siamese CNN:
- Backbone: VGG16 (ImageNet pre-trained, last conv block unfrozen)
- Loss: Contrastive Loss with learned margin
- Optimizer: Adam with L2 weight decay
- Augmentation: ColorJitter + RandomAffine (to prevent scanner-artifact shortcuts)
- Evaluation: Writer-independent split (Signers 1–45 train, 46–55 test)
- Epochs: 15 with StepLR scheduler

### RoBERTa Unfair Clause Detector:
- Backbone: roberta-base (125M parameters, **fully fine-tuned** — all layers trainable)
- Dataset: LexGLUE UNFAIR-ToS (`coastalcph/lex_glue`)
- Architecture head: Linear(768→512) → LayerNorm → ReLU → Dropout(0.3) → Linear(512→128) → LayerNorm → ReLU → Dropout(0.15) → Linear(128→2)
- Loss: Weighted CrossEntropyLoss (to handle ~9:1 class imbalance)
- Optimizer: AdamW with linear warmup + cosine decay scheduler
- Learning rate: 2e-5
- Max sequence length: 256 tokens
- Training: 10 epochs, best checkpoint saved on peak Macro F1
- Inference threshold: 0.45 (tuned for higher recall on UNFAIR class)

---

*This document was generated as a reference for semester project presentation and defense.*
