# Legal Document Authenticity Verifier — System Design & Technical Report

This document provides a comprehensive technical overview of the design, architecture, model specifications, evaluation benchmarks, and explainability mechanisms implemented in the Legal Document Authenticity Verifier system.

---

## 1. Introduction & Objectives

Autonomous legal document verification presents a unique challenge: verification cannot rely solely on visual inspection (e.g., verifying a signature) or semantic analysis (e.g., scanning the contract text) in isolation. A forged signature renders an otherwise standard contract invalid, while a genuine signature on a predatory contract containing unfair clauses presents severe legal and financial risks.

To address these vulnerabilities, this project implements a **multi-modal forensic analysis system** that evaluates document authenticity across two primary modalities:
1. **Signature Verification (Visual Modality)**: Employs metric learning via a Siamese CNN to determine if a query signature matches a reference signature, detecting skilled forgeries.
2. **Unfair Clause Detection (Semantic Modality)**: Employs a transformer-based language model to classify contract text, flagging predatory clauses like unilateral termination, class-action waivers, or hidden liability exclusions.

The system is designed as a modular multi-agent pipeline orchestrated by a Supervisor Agent, with built-in Explainable AI (XAI) models providing human-interpretable rationales for all decisions.

---

## 2. System Architecture & Data Flow

The architecture consists of a FastAPI backend serving PyTorch models, a React frontend dashboard, and an inference pipeline managed by specialized agents:

```
User Input (Reference Signature + Query Signature + Contract Text)
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Preprocessing Agent │
                    │ - Normalizes images │
                    │ - Tokenizes text    │
                    └──────────┬──────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
     ┌──────────────────┐            ┌──────────────────┐
     │ Signature Agent  │            │    Text Agent    │
     │ Siamese CNN      │            │   RoBERTa NLP    │
     │ (VGG16 Backbone) │            │ (Fully Fine-Tuned)│
     └─────────┬────────┘            └─────────┬────────┘
               │ similarity score              │ probability score
               │ (0.0 to 1.0)                  │ (0.0 to 1.0)
               ▼                               ▼
     ┌──────────────────────────────────────────────────┐
     │                 Supervisor Agent                 │
     │  Risk Fusion: (0.6 * sig_risk) + (0.4 * nlp_risk)│
     │  Combined Risk Threshold: >= 0.40 -> SUSPICIOUS  │
     └─────────────────────────┬────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
     ┌──────────────────┐            ┌──────────────────┐
     │    XAI Agent     │            │    XAI Agent     │
     │    (Grad-CAM)    │            │      (SHAP)      │
     │ Generate Heatmap │            │ Token Attribution│
     └─────────┬────────┘            └─────────┬────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
                     [UI Dashboard Output]
```

---

## 3. Datasets & Benchmarks

### CEDAR Signature Dataset (Visual Modality)
* **Description**: A public signature database compiled by the Center of Excellence for Document Analysis and Recognition.
* **Content**: Signatures from 55 writers. Each writer contributes 24 genuine signatures and 24 skilled forgeries (forgeries produced by individuals who practiced the target signature beforehand).
* **Format**: Grayscale images of varying resolutions, normalized and binarized during preprocessing.
* **Usage**: Used to train the metric-learning Siamese CNN. To evaluate real-world generalization, we implement a **writer-independent split**: the model is trained on Signers 1–45 and evaluated on Signers 46–55 (completely unseen writers).

### LexGLUE UNFAIR-ToS (Semantic Modality)
* **Description**: The Unfair Terms of Service subset of the LexGLUE (Legal Evaluation Benchmark for General Understanding of English) dataset.
* **Content**: Contract sentences annotated by legal experts for unfair clauses based on European consumer law.
* **Categories**: Detects unilateral termination rights, liability exclusions, unilateral contract modifications, and forced arbitration waivers.
* **Class Imbalance**: Approximately 9:1 ratio of SAFE to UNFAIR clauses. Handled using class-weighted loss functions during training.

---

## 4. Computer Vision Pipeline: Siamese CNN

Offline signature verification is framed as a **metric learning task** using a Siamese network architecture.

### Model Architecture
* **Backbone**: VGG16 pre-trained on ImageNet.
* **Feature Extraction**: The network utilizes weight-sharing twin towers to process the reference image ($I_{ref}$) and the query image ($I_{query}$).
* **Fine-Tuning Protocol**: The first four convolutional blocks of VGG16 are frozen to retain general low-level edge detectors. The final convolutional block and the fully connected layers are fine-tuned.
* **Embedding Projection**: Output feature maps are flattened and projected through a dense layer stack to yield a $128$-dimensional embedding vector ($v \in \mathbb{R}^{128}$).

### Similarity Metrics & Loss Function
* **Distance Measure**: Euclidean distance ($d$) between the two feature embeddings:
  $$d(v_{ref}, v_{query}) = \|v_{ref} - v_{query}\|_2$$
* **Similarity Output**: Mapped using an exponential decay function to yield a score $s \in [0.0, 1.0]$:
  $$s = \exp(-d)$$
* **Training Objective**: Trained using **Contrastive Loss** to pull embeddings of genuine pairs close together ($d \to 0$) and push forged pairs apart beyond a specified margin ($m = 2.0$):
  $$\mathcal{L} = (1 - Y) \frac{1}{2} d^2 + Y \frac{1}{2} \max(0, m - d)^2$$
  *Where $Y = 0$ for genuine pairs and $Y = 1$ for forged pairs.*

### Signature Verification Performance
* **Accuracy**: **80.21%** (Writer-independent evaluation)
* **Area Under ROC (AUC)**: **0.9209**

---

## 5. Natural Language Processing Pipeline: RoBERTa

Contract clause analysis is framed as a supervised binary classification task.

### Model Architecture
* **Base Transformer**: `roberta-base` (125 million parameters) pre-trained on 160GB of diverse text corpora.
* **Classifier Head**: Custom 3-layer MLP attached to the pooler output:
  $$\text{Linear}(768 \to 512) \to \text{LayerNorm} \to \text{ReLU} \to \text{Dropout}(0.3)$$
  $$\to \text{Linear}(512 \to 128) \to \text{LayerNorm} \to \text{ReLU} \to \text{Dropout}(0.15)$$
  $$\to \text{Linear}(128 \to 2)$$
* **Context Window**: Configured for inputs up to $256$ tokens.

### Fine-Tuning Strategy
* **Optimization**: AdamW optimizer (decoupled weight decay at $0.01$).
* **Learning Rate Schedule**: Initial learning rate of $2 \times 10^{-5}$ with a linear warmup for the first $10\%$ of steps followed by cosine learning rate decay.
* **Imbalance Treatment**: Weighted Cross-Entropy Loss applied during backpropagation:
  $$w_{\text{unfair}} \approx 9.0, \quad w_{\text{safe}} \approx 1.0$$
* **Decision Boundary**: Set at $0.45$ to prioritize recall on unfair clauses.

### Contract Analysis Performance
* **Accuracy**: **95.83%**
* **Macro F1-Score**: **89.65%**
* **Area Under ROC (AUC)**: **0.956**

---

## 6. Supervisor Agent & Decision Fusion

The Supervisor Agent acts as an orchestrator, fusing the independent classification metrics of the visual and semantic models into a final security verdict.

### Risk Fusion Math
1. **Signature Risk Calculation**:
   $$\text{sig\_risk} = 1.0 - s$$
2. **Text Risk Calculation**:
   $$\text{text\_risk} = P(\text{Unfair} \mid \text{clause\_text})$$
3. **Fused Risk Score**:
   $$\text{combined\_risk} = (0.6 \times \text{sig\_risk}) + (0.4 \times \text{text\_risk})$$

### Rationale
* **Weighting Scheme (60/40)**: Physical signature forgery is weighted higher because signature verification is a direct indicator of unauthorized document execution (fraud). A contract with high semantic risk (unfair clauses) is still legally binding and authentic in terms of execution, whereas a forged signature invalidates the entire agreement immediately.
* **Thresholding**: If $\text{combined\_risk} \geq 0.40$, the Supervisor flags the document as **SUSPICIOUS**. Otherwise, the document is classified as **AUTHENTIC**.

---

## 7. Explainable AI (XAI) Architecture

To ensure auditability, the system avoids "black-box" predictions by generating visual and textual explanations.

### Visual Modality: Grad-CAM on Siamese CNN
* **Methodology**: Gradient-weighted Class Activation Mapping (Grad-CAM) extracts activation maps from the final convolutional layer of the VGG16 backbone.
* **Gradient Backpropagation**: Because the Siamese model uses distance metric learning rather than direct classification, gradients are computed relative to the magnitude (L2 norm) of the projected embedding vectors.
* **Overlay Generation**: Gradients are average-pooled to obtain channel weights, which scale the feature maps. After applying a ReLU activation to focus on positive contributions, the map is upsampled and blended with the original image using OpenCV (`COLORMAP_JET`).
* **Interpretation**: Highlights the specific pen strokes (e.g., regions of tremor, tracing hesitation) that influenced the distance score.

### Semantic Modality: SHAP Token Attribution
* **Methodology**: SHapley Additive exPlanations (SHAP) is utilized to measure the marginal contribution of individual tokens to the model's output probability.
* **Perturbation Model**: Tokens are masked iteratively, and the change in prediction is measured to approximate Shapley values.
* **Interpretation**: Highlights the exact words or phrases (e.g., *"unilateral"*, *"sole discretion"*, *"waives the right"*) that triggered the unfair clause classification.

---

## 8. System Design & Technical FAQ

### Q1: How does a Siamese network verify signatures of unseen signers without retraining?
Siamese networks do not learn to classify signatures as belonging to specific individuals. Instead, they learn a **metric embedding space** where the distance between vector representations corresponds to signature similarity. The network learns universal visual characteristics of handwritings and forgeries (tremors, pressure variations, stroke connections). During inference, the query signature is compared to a reference signature in this embedding space. This allows zero-shot verification for any new individual immediately.

### Q2: Why is a writer-independent evaluation protocol crucial for this system?
In signature verification, a writer-dependent split (where signatures of the same individuals are present in both training and test sets) produces artificially high accuracies (often $>95\%$). However, such models fail in production because they memorize writer-specific styles. By utilizing a writer-independent split (training on Signers 1-45 and testing on Signers 46-55), we ensure the model's metrics reflect its ability to generalize to novel writers, mimicking a real-world deployment scenario.

### Q3: Why is the Macro F1-score reported for contract analysis instead of accuracy alone?
The LexGLUE UNFAIR-ToS dataset has a severe class imbalance (~90% SAFE, ~10% UNFAIR). If a model predicted "SAFE" for every input, it would achieve 90% accuracy while failing to identify any predatory clauses. The Macro F1-score calculates the harmonic mean of precision and recall for both classes independently, averaging them equally. Achieving a Macro F1 of 89.65% verifies that the model is highly precise and sensitive in identifying both SAFE and UNFAIR clauses.

### Q4: How is Grad-CAM adapted for a Siamese network where there is no class probability score?
Standard Grad-CAM backpropagates the gradient of a target class logit. For a Siamese CNN, we do not have class logits; we have an embedding distance. We backpropagate the gradient of the L2 norm of the embedding difference. This represents how much the spatial features of the query signature contribute to pushing the query embedding away from the reference embedding. The resulting heatmap identifies the visual strokes responsible for the vector distance.

### Q5: What is the benefit of using AdamW over standard Adam for fine-tuning the transformer model?
Standard Adam implements weight decay by adding it directly to the gradient update, which links it with the moving averages of the gradients. For transformer models like RoBERTa, this leads to suboptimal regularization. AdamW decoupled weight decay from the gradient update step, applying decay directly to the weights. This yields better generalization, prevents catastrophic forgetting, and stabilizes training when full fine-tuning is performed.

### Q6: Why does the system use a weighted Cross-Entropy Loss for training RoBERTa?
Due to the 9:1 imbalance in the text dataset, standard Cross-Entropy loss would bias the model toward predicting the majority class (SAFE) because errors on the minority class (UNFAIR) contribute very little to the overall loss. Weighted Cross-Entropy scales the loss of the minority class by a factor proportional to its under-representation. This forces the optimizer to penalize false negatives heavily, optimizing the decision boundary for higher recall.

### Q7: What are the main limitations of the current architecture?
1. **Skilled Forgery Limits**: Extremely high-quality forgeries can bypass the Siamese CNN, as the model relies on static offline images where dynamic signature data (e.g., pressure profiles, velocity vectors) is lost.
2. **Short Clause Ambiguity**: The NLP model exhibits lower recall on very short, single-sentence clauses where predatory intent is obfuscated using standard commercial language.
3. **Metric Learning Grad-CAM Variance**: Because we backpropagate through the embedding norm, the resulting heatmaps can occasionally appear flat if the embeddings are highly converged.

### Q8: What are the recommended directions for scaling this project?
* **Transition to Vision Transformers (ViT)**: Replacing the CNN backbone with a Vision Transformer to capture global attention patterns across signature strokes.
* **Multi-Class NLP Labels**: Expanding the NLP model's classifier head to output multi-class ratings mapping directly to the 8 LexGLUE categories, rather than a binary output.
* **Retrieval-Augmented Verification (RAG)**: Integrating a vector store containing legal precedents to allow the text model to cross-reference flagged clauses with actual judicial rulings.
