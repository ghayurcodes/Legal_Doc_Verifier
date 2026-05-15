# =============================================================================
#  UNFAIR-ToS Legal Clause Detector — RoBERTa Fine-Tuning
#  Dataset  : LexGLUE / UNFAIR-ToS  (HuggingFace coastalcph/lex_glue)
#  Model    : roberta-base + improved MLP head (binary classifier)
#  Backend  : Drop-in replacement — class name & .pt filename unchanged
#
#  HOW TO USE IN GOOGLE COLAB:
#  Each "# ── CELL N" section below is one Colab cell.
#  Copy everything between two "# ── CELL" markers into that cell.
# =============================================================================


# ── CELL 1 ── Install dependencies ───────────────────────────────────────────
# (Run once at the start of your Colab session)

!pip install transformers datasets accelerate --quiet
!pip install torch --quiet
!pip install scikit-learn matplotlib seaborn --quiet

print("✅ All libraries installed!")


# ── CELL 2 ── Mount Google Drive ─────────────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted!")


# ── CELL 3 ── Load the LexGLUE UNFAIR-ToS dataset ───────────────────────────

from datasets import load_dataset

print("Loading LexGLUE UNFAIR-ToS dataset from Hugging Face...")
raw_dataset = load_dataset("coastalcph/lex_glue", "unfair_tos")

print(f"\n✅ Dataset loaded!")
print(f"   Train   : {len(raw_dataset['train'])} samples")
print(f"   Val     : {len(raw_dataset['validation'])} samples")
print(f"   Test    : {len(raw_dataset['test'])} samples")
print("\n--- Sample from train set ---")
print(raw_dataset['train'][0])
print("Labels key: [] = SAFE clause | non-empty list = UNFAIR clause")


# ── CELL 4 ── Config, Seed, Dataset class, Model definition ──────────────────

import random, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel

# ── Global constants ──────────────────────────────────────────────────────────
SEED                = 42
MAX_LEN             = 256    # covers virtually all UNFAIR-ToS sentences
BATCH_SIZE          = 32     # base batch; effective = BATCH_SIZE × ACCUM_STEPS
INFERENCE_THRESHOLD = 0.45   # aligned with backend/api.py (was 0.50 — mismatch!)

# ── Reproducibility ───────────────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    """Seed all RNGs so every training run gives the same result."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False  # small speed trade-off for determinism

seed_everything(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
print(f"GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only ⚠️'}")

# ── Label helper ──────────────────────────────────────────────────────────────
def to_binary(labels_list):
    """[] → 0 (SAFE),  any label → 1 (UNFAIR)."""
    return 1 if len(labels_list) > 0 else 0

# ── Dataset ───────────────────────────────────────────────────────────────────
class UnfairClauseDataset(Dataset):
    """Wraps the HuggingFace LexGLUE split for PyTorch binary classification."""

    def __init__(self, hf_split, tokenizer, max_len: int = MAX_LEN):
        self.texts     = [item['text']            for item in hf_split]
        self.labels    = [to_binary(item['labels']) for item in hf_split]
        self.tokenizer = tokenizer
        self.max_len   = max_len

        safe   = self.labels.count(0)
        unfair = self.labels.count(1)
        ratio  = safe / unfair if unfair else float('inf')
        print(f"   SAFE: {safe:>5d} | UNFAIR: {unfair:>4d} | imbalance ratio ≈ {ratio:.1f}:1")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Model ─────────────────────────────────────────────────────────────────────
class DeceptionClassifier(nn.Module):
    """
    RoBERTa-base + 3-layer MLP head for binary legal clause classification.

    Class name kept as 'DeceptionClassifier' for backend/api.py compatibility.
    Filename roberta_deception.pt is also unchanged — pure drop-in replacement.

    Architecture improvements vs original:
      • nn.LayerNorm after each hidden layer  → stable gradient flow
      • dropout / 2 on 2nd hidden layer       → prevents over-regularisation
      • All RoBERTa weights are trainable     → full fine-tuning
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        for param in self.roberta.parameters():
            param.requires_grad = True                       # full fine-tuning

        hidden = self.roberta.config.hidden_size             # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.LayerNorm(512),                               # NEW — gradient stability
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),                               # NEW — gradient stability
            nn.ReLU(),
            nn.Dropout(dropout / 2),                         # NEW — lighter on 2nd layer
            nn.Linear(128, 2),                               # 0 = SAFE, 1 = UNFAIR
        )

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]                 # [CLS] token
        return self.classifier(cls)

    def predict_deception_score(self, text: str, tokenizer, device) -> float:
        """
        Returns P(UNFAIR) in [0.0, 1.0].
          0.0 → clearly SAFE clause
          1.0 → clearly UNFAIR / predatory clause

        Method name kept as predict_deception_score for backend/api.py compatibility.
        Use INFERENCE_THRESHOLD (0.45) — not 0.50 — to match the backend.
        """
        self.eval()
        enc = tokenizer(
            text, return_tensors='pt',
            max_length=MAX_LEN, truncation=True, padding='max_length',
        )
        enc = {k: v.to(device) for k, v in enc.items()
               if k in ('input_ids', 'attention_mask')}
        with torch.no_grad():
            logits = self.forward(enc['input_ids'], enc['attention_mask'])
        return torch.softmax(logits, dim=1)[0][1].item()   # P(UNFAIR)

print("\n✅ Config, seed, Dataset class, and Model defined!")


# ── CELL 5 ── Tokenizer + DataLoaders ────────────────────────────────────────

print("Loading RoBERTa tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

print("\nBuilding dataset splits:")
print("  train  :", end=" "); train_ds = UnfairClauseDataset(raw_dataset['train'],      tokenizer)
print("  val    :", end=" "); val_ds   = UnfairClauseDataset(raw_dataset['validation'], tokenizer)
print("  test   :", end=" "); test_ds  = UnfairClauseDataset(raw_dataset['test'],       tokenizer)

# persistent_workers=True avoids re-spawning workers every epoch (faster on Colab)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)

print(f"\n✅ DataLoaders ready!")
print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


# ── CELL 6 ── Training ────────────────────────────────────────────────────────

import torch.optim as optim
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score as sk_f1
import matplotlib.pyplot as plt

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS       = 10      # more room; early stopping exits when ready
LR_BACKBONE  = 1e-5    # conservative LR for pretrained RoBERTa weights
LR_HEAD      = 5e-5    # higher LR for the newly-initialised classifier head
DROPOUT      = 0.3
PATIENCE     = 4       # stop after 4 epochs with no macro-F1 improvement
ACCUM_STEPS  = 2       # gradient accumulation → effective batch = 32 × 2 = 64
SAVE_PATH    = '/content/roberta_deception.pt'

# ── Initialise model (with seed for reproducibility) ─────────────────────────
seed_everything(SEED)
model = DeceptionClassifier(dropout=DROPOUT).to(DEVICE)

# ── Class weights — sklearn balanced ─────────────────────────────────────────
# n_samples / (n_classes × class_count) for each class.
# Gives the UNFAIR class ~7.8× more weight to compensate for imbalance.
all_train_labels = [to_binary(item['labels']) for item in raw_dataset['train']]
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=all_train_labels)
class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)
print(f"Class weights → SAFE: {class_weights[0]:.4f} | UNFAIR: {class_weights[1]:.4f}")

# label_smoothing=0.05 prevents the model from becoming overconfident on SAFE
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

# ── Discriminative learning rates ─────────────────────────────────────────────
# Pretrained backbone: low LR (avoid destroying learned representations)
# Classifier head   : high LR (it starts from random init, needs to learn fast)
no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']

optimizer = optim.AdamW([
    {'params': [p for n, p in model.roberta.named_parameters()
                if not any(nd in n for nd in no_decay)],
     'lr': LR_BACKBONE, 'weight_decay': 0.01},
    {'params': [p for n, p in model.roberta.named_parameters()
                if     any(nd in n for nd in no_decay)],
     'lr': LR_BACKBONE, 'weight_decay': 0.0},
    {'params': [p for n, p in model.classifier.named_parameters()
                if not any(nd in n for nd in no_decay)],
     'lr': LR_HEAD, 'weight_decay': 0.01},
    {'params': [p for n, p in model.classifier.named_parameters()
                if     any(nd in n for nd in no_decay)],
     'lr': LR_HEAD, 'weight_decay': 0.0},
])

# ── Cosine schedule with warmup ───────────────────────────────────────────────
effective_steps_per_epoch = len(train_loader) // ACCUM_STEPS
total_steps  = effective_steps_per_epoch * EPOCHS
warmup_steps = total_steps // 10
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# ── Mixed-precision scaler ────────────────────────────────────────────────────
AMP_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = GradScaler(AMP_DEVICE)

# ── Training loop ─────────────────────────────────────────────────────────────
train_losses, val_accs, val_macro_f1s, val_binary_f1s = [], [], [], []
best_val_f1 = 0.0
no_improve  = 0

print(f"\nTraining for up to {EPOCHS} epochs  |  Patience = {PATIENCE}")
print(f"Backbone LR = {LR_BACKBONE}  |  Head LR = {LR_HEAD}  |  "
      f"Effective batch = {BATCH_SIZE * ACCUM_STEPS}  |  Warmup = {warmup_steps} steps")
print(f"⚠  Model saved on MACRO F1 (not accuracy) — critical fix for imbalanced data")
print("─" * 80)

for epoch in range(1, EPOCHS + 1):

    # ── Training phase ────────────────────────────────────────────────────────
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['label'].to(DEVICE)

        with autocast(AMP_DEVICE):
            logits = model(ids, mask)
            loss   = criterion(logits, lbls) / ACCUM_STEPS   # scale for accumulation

        scaler.scale(loss).backward()
        total_loss += loss.item() * ACCUM_STEPS               # un-scale for logging

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ── Validation phase ──────────────────────────────────────────────────────
    model.eval()
    preds_v, labels_v = [], []

    with torch.no_grad():
        for batch in val_loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbls = batch['label'].to(DEVICE)
            with autocast(AMP_DEVICE):
                logits = model(ids, mask)
            preds_v.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels_v.extend(lbls.cpu().tolist())

    val_acc      = 100.0 * sum(p == l for p, l in zip(preds_v, labels_v)) / len(labels_v)
    val_macro_f1 = 100.0 * sk_f1(labels_v, preds_v, average='macro',  zero_division=0)
    val_bin_f1   = 100.0 * sk_f1(labels_v, preds_v, average='binary', zero_division=0)

    val_accs.append(val_acc)
    val_macro_f1s.append(val_macro_f1)
    val_binary_f1s.append(val_bin_f1)

    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch:2d}/{EPOCHS}  │  Loss: {avg_loss:.4f}  │  "
          f"Acc: {val_acc:.2f}%  │  Macro F1: {val_macro_f1:.2f}%  │  "
          f"UNFAIR F1: {val_bin_f1:.2f}%  │  LR: {current_lr:.2e}")

    # Save checkpoint when macro F1 improves
    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        no_improve  = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Best model saved! (macro F1: {best_val_f1:.2f}%)")
    else:
        no_improve += 1
        print(f"  − No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            print(f"  ⏹ Early stopping at epoch {epoch}  |  Best macro F1: {best_val_f1:.2f}%")
            break

# ── Training curves ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(train_losses, color='#c9973a', marker='o', linewidth=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('CrossEntropy Loss')
ax1.set_title('Training Loss')

ax2.plot(val_accs,       color='#4a90d9', marker='s', linewidth=2, label='Val Accuracy')
ax2.plot(val_macro_f1s,  color='#e84c3d', marker='^', linewidth=2, label='Val Macro F1')
ax2.plot(val_binary_f1s, color='#2ecc71', marker='D', linewidth=2, label='Val UNFAIR F1')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Score (%)')
ax2.set_title('Validation Metrics'); ax2.legend()

plt.tight_layout()
plt.savefig('/content/roberta_training_curves.png', dpi=150)
plt.show()
print("\n✅ Training complete!")


# ── CELL 7 ── Evaluation on the test set ─────────────────────────────────────

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score,
    matthews_corrcoef, roc_curve, precision_recall_curve,
)
import seaborn as sns

# ── Load the best checkpoint ──────────────────────────────────────────────────
model.load_state_dict(
    torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()

# ── Collect predictions + probabilities ──────────────────────────────────────
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['label'].to(DEVICE)
        with autocast(AMP_DEVICE):
            logits = model(ids, mask)
        probs = torch.softmax(logits, dim=1)[:, 1]            # P(UNFAIR)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(lbls.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

# ── Metrics ───────────────────────────────────────────────────────────────────
acc       = accuracy_score(all_labels, all_preds)
macro_f1  = f1_score(all_labels, all_preds, average='macro',  zero_division=0)
binary_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
roc_auc   = roc_auc_score(all_labels, all_probs)
avg_prec  = average_precision_score(all_labels, all_probs)
mcc       = matthews_corrcoef(all_labels, all_preds)

print("╔══════════════════════════════════════════════════════╗")
print("║      Unfair Clause Detector — Test Set Results        ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Accuracy              : {acc*100:6.2f}%                     ║")
print(f"║  Macro F1              : {macro_f1*100:6.2f}%                     ║")
print(f"║  UNFAIR F1 (binary)    : {binary_f1*100:6.2f}%                     ║")
print(f"║  ROC-AUC               : {roc_auc:.4f}                     ║")
print(f"║  Avg Precision (AP)    : {avg_prec:.4f}                     ║")
print(f"║  Matthews CC (MCC)     : {mcc:.4f}                     ║")
print("╚══════════════════════════════════════════════════════╝")
print()
print(classification_report(all_labels, all_preds,
                             target_names=['SAFE', 'UNFAIR'], zero_division=0))

# ── Sanity check — uses INFERENCE_THRESHOLD (0.45) to match backend ───────────
print(f"\n🧪 Sanity Check  (threshold = {INFERENCE_THRESHOLD}  ← aligned with backend/api.py):")
sanity_cases = [
    # (clause_text, expected_label)   0 = SAFE, 1 = UNFAIR
    ("This agreement takes effect on January 1st, 2024.",                                0),
    ("Payment is due within 30 days of the invoice date.",                               0),
    ("You may cancel your subscription at any time for any reason.",                     0),
    ("The company may terminate your account at any time without notice or refund.",     1),
    ("User waives all rights to file a lawsuit or class action under any circumstances.", 1),
    ("We may modify these terms at any time without prior notice to you.",               1),
    ("The company is not liable for any damages, including loss of data or profits.",    1),
    ("By continuing to use the service you agree to the updated terms.",                 1),
]

passed = 0
for text, expected in sanity_cases:
    score  = model.predict_deception_score(text, tokenizer, DEVICE)
    pred   = 1 if score > INFERENCE_THRESHOLD else 0
    ok     = "✓" if pred == expected else "✗"
    label  = "🚨 UNFAIR" if score > INFERENCE_THRESHOLD else "✅ SAFE  "
    passed += (pred == expected)
    print(f"  [{ok}] {label} ({score:.2f}) — {text[:72]}")

print(f"\n  Result: {passed}/{len(sanity_cases)} correct")
if passed == len(sanity_cases):
    print("  🎉 All sanity checks passed!")
else:
    print("  ⚠  Some cases missed — consider running more epochs or adjusting the threshold.")

# ── Evaluation plots ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1) Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=['SAFE', 'UNFAIR'], yticklabels=['SAFE', 'UNFAIR'])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

# 2) ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
axes[1].plot(fpr, tpr, color='#c9973a', lw=2, label=f'AUC = {roc_auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve'); axes[1].legend()

# 3) Precision-Recall curve (more informative than ROC for imbalanced data)
prec, rec, _ = precision_recall_curve(all_labels, all_probs)
axes[2].plot(rec, prec, color='#4a90d9', lw=2, label=f'AP = {avg_prec:.3f}')
axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
axes[2].set_title('Precision-Recall Curve'); axes[2].legend()

plt.tight_layout()
plt.savefig('/content/roberta_evaluation.png', dpi=150)
plt.show()


# ── CELL 8 ── Save everything to Google Drive ─────────────────────────────────

import shutil, os

DRIVE_FOLDER = '/content/drive/MyDrive/Colab Notebooks/sem_project_231570/models'
os.makedirs(DRIVE_FOLDER, exist_ok=True)

# Model weights — same filename as before, drop-in replacement for backend
shutil.copy('/content/roberta_deception.pt',         f'{DRIVE_FOLDER}/roberta_deception.pt')
shutil.copy('/content/roberta_training_curves.png',  f'{DRIVE_FOLDER}/roberta_training_curves.png')
shutil.copy('/content/roberta_evaluation.png',       f'{DRIVE_FOLDER}/roberta_evaluation.png')

print("✅ All files saved to Google Drive!")
print(f"   Location: {DRIVE_FOLDER}/")
print("\nNext steps:")
print("  1. Download roberta_deception.pt from Drive to your PC")
print("  2. Put it in: legal_doc_verifier/backend/models/saved/roberta_deception.pt")
print("     (REPLACE the existing file — same filename, drop-in replacement!)")
print("  3. Restart the backend: python -m uvicorn api:app --reload --port 8000")
print("  4. Frontend will now show improved UNFAIR CLAUSE RISK scores.")
