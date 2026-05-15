# ╔══════════════════════════════════════════════════════════════════════╗
# ║   COLAB NOTEBOOK 2 (NEW) — RoBERTa Unfair Clause Detector           ║
# ║   Dataset:   LexGLUE → UNFAIR-ToS (Hugging Face, no download needed)║
# ║   Runtime:   GPU (T4) | Estimated time: ~20-30 minutes              ║
# ║   Output:    roberta_deception.pt (same filename, drop-in replace)   ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# HOW TO USE IN COLAB:
#   1. Open Google Colab → New Notebook
#   2. Set Runtime > Change runtime type > T4 GPU
#   3. Copy each CELL block below into a separate Colab cell
#   4. Run cells ONE BY ONE from top to bottom
#   5. Run CELL 8 IMMEDIATELY after training finishes
#
# WHAT THIS MODEL DOES:
#   - Reads a sentence from a legal contract
#   - Predicts: SAFE (0) or UNFAIR/PREDATORY (1)
#   - Example: "User waives all rights to compensation" → UNFAIR


# ════════════════════════════════════════════════════════════════
# CELL 1 — Install Required Libraries
# ════════════════════════════════════════════════════════════════

!pip install transformers datasets --quiet
!pip install torch --quiet
!pip install scikit-learn matplotlib seaborn --quiet

print("✅ All libraries installed!")


# ════════════════════════════════════════════════════════════════
# CELL 2 — Mount Google Drive
# ════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

print("✅ Google Drive mounted!")


# ════════════════════════════════════════════════════════════════
# CELL 3 — Load the LexGLUE UNFAIR-ToS Dataset (No download needed!)
# ════════════════════════════════════════════════════════════════

from datasets import load_dataset

print("Loading LexGLUE UNFAIR-ToS dataset from Hugging Face...")
print("(This streams directly — no zip file needed!)")

raw_dataset = load_dataset("coastalcph/lex_glue", "unfair_tos")

print(f"\n✅ Dataset loaded!")
print(f"   Train samples : {len(raw_dataset['train'])}")
print(f"   Val samples   : {len(raw_dataset['validation'])}")
print(f"   Test samples  : {len(raw_dataset['test'])}")

# Show what a sample looks like
print("\n--- Sample from train set ---")
print(raw_dataset['train'][0])
print("Labels key: empty list [] = SAFE clause, non-empty list = UNFAIR clause")


# ════════════════════════════════════════════════════════════════
# CELL 4 — Define Model Architecture and Dataset Class
# ════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only ⚠️'}")


# Convert multi-label to binary:
# labels == [] (empty)  → 0 = SAFE
# labels has any value  → 1 = UNFAIR/PREDATORY
def to_binary(labels_list):
    return 1 if len(labels_list) > 0 else 0


class UnfairClauseDataset(Dataset):
    """Wraps the HuggingFace LexGLUE dataset for PyTorch."""
    def __init__(self, hf_split, tokenizer, max_len=256):
        self.texts = [item['text'] for item in hf_split]
        self.labels = [to_binary(item['labels']) for item in hf_split]
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Print class distribution
        safe = self.labels.count(0)
        unfair = self.labels.count(1)
        print(f"   Class distribution → SAFE: {safe} | UNFAIR: {unfair}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ──────────────────────────────────────────────────────────────
# IMPORTANT: We keep the class name "DeceptionClassifier" and
# the same architecture as before. This means the saved .pt file
# is a drop-in replacement — no changes needed in backend/api.py
# ──────────────────────────────────────────────────────────────
class DeceptionClassifier(nn.Module):
    """
    RoBERTa-based binary classifier.
    Repurposed from 'Deceptive Text' to 'Unfair Legal Clause' detection.
    Architecture is identical — only the training data changes.
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        # Full fine-tuning — all RoBERTa layers trainable
        for param in self.roberta.parameters():
            param.requires_grad = True

        hidden_size = self.roberta.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)   # 2 classes: 0=Safe, 1=Unfair
        )

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = out.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_embedding)

    def predict_deception_score(self, text, tokenizer, device):
        """
        Returns a score from 0.0 to 1.0.
        0.0 = completely SAFE clause
        1.0 = clearly UNFAIR/PREDATORY clause
        (Named predict_deception_score to stay compatible with backend/api.py)
        """
        self.eval()
        enc = tokenizer(text, return_tensors='pt', max_length=256,
                        truncation=True, padding='max_length')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.forward(enc['input_ids'], enc['attention_mask'])
        probs = torch.softmax(logits, dim=1)
        return probs[0][1].item()  # probability of class 1 (UNFAIR)


print("\n✅ Model classes defined!")


# ════════════════════════════════════════════════════════════════
# CELL 5 — Load Tokenizer and Create DataLoaders
# ════════════════════════════════════════════════════════════════

print("Loading RoBERTa tokenizer... (downloads ~500MB on first run)")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

BATCH_SIZE = 32  # Larger batch = faster training on small dataset

print("\nBuilding train split:")
train_ds = UnfairClauseDataset(raw_dataset['train'],      tokenizer)
print("Building validation split:")
val_ds   = UnfairClauseDataset(raw_dataset['validation'], tokenizer)
print("Building test split:")
test_ds  = UnfairClauseDataset(raw_dataset['test'],       tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"\n✅ DataLoaders ready!")
print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


# ════════════════════════════════════════════════════════════════
# CELL 6 — Train the Model
# ════════════════════════════════════════════════════════════════

import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup

# ── Hyperparameters (aligned with official LexGLUE paper) ────
# Paper: batch_size=8, grad_accumulation=8 → effective batch = 64
EPOCHS             = 8
LR                 = 2e-5
DROPOUT            = 0.3
PATIENCE           = 3
ACCUM_STEPS        = 8    # ← gradient accumulation (paper-accurate)
SAVE_PATH          = '/content/roberta_deception.pt'

model = DeceptionClassifier(dropout=DROPOUT).to(DEVICE)

# ── Weighted loss to handle class imbalance ───────────────────
# UNFAIR-ToS has ~70% SAFE, ~30% UNFAIR → weight UNFAIR class higher
all_train_labels = [to_binary(item['labels']) for item in raw_dataset['train']]
class_counts = np.bincount(all_train_labels)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(DEVICE)
class_weights = class_weights / class_weights.sum()
print(f"Class distribution → SAFE: {class_counts[0]} | UNFAIR: {class_counts[1]}")
print(f"Class weights      → SAFE: {class_weights[0]:.4f} | UNFAIR: {class_weights[1]:.4f}")
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=0.01
)

# Effective steps account for gradient accumulation
total_steps  = (len(train_loader) // ACCUM_STEPS) * EPOCHS
warmup_steps = total_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

train_losses = []
best_val_f1  = 0.0    # Saving based on Macro F1 to handle class imbalance
no_improve   = 0

print(f"\nStarting training for up to {EPOCHS} epochs...")
print(f"LR={LR} | Batch={BATCH_SIZE} | Grad Accum={ACCUM_STEPS} | Effective Batch=64")
print(f"Warmup steps={warmup_steps} | Patience={PATIENCE}")
print("─" * 65)

from sklearn.metrics import f1_score as sk_f1

for epoch in range(1, EPOCHS + 1):
    # ── Training with gradient accumulation ───────────────────
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['label'].to(DEVICE)

        logits = model(ids, mask)
        loss   = criterion(logits, lbls)
        loss   = loss / ACCUM_STEPS          # scale loss by accum steps
        loss.backward()
        total_loss += loss.item() * ACCUM_STEPS

        # Only step optimizer every ACCUM_STEPS batches
        if (i + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ── Validation (report BOTH accuracy and F1) ──────────────
    model.eval()
    val_preds_all, val_labels_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbls = batch['label'].to(DEVICE)
            preds = torch.argmax(model(ids, mask), dim=1)
            val_preds_all.extend(preds.cpu().tolist())
            val_labels_all.extend(lbls.cpu().tolist())

    val_acc = 100 * sum(p == l for p, l in zip(val_preds_all, val_labels_all)) / len(val_labels_all)
    val_f1_macro  = sk_f1(val_labels_all, val_preds_all, average='macro', zero_division=0)
    val_f1_binary = sk_f1(val_labels_all, val_preds_all, average='binary', zero_division=0)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}/{EPOCHS} │ Loss: {avg_loss:.4f} │ Val Acc: {val_acc:.2f}% │ Val Macro F1: {val_f1_macro:.4f} │ LR: {current_lr:.2e}")

    # ── Save based on Macro F1 (handles class imbalance best) ─────────
    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        no_improve  = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Best model saved! (val Macro F1: {best_val_f1:.4f})")
    else:
        no_improve += 1
        print(f"  − No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

# Plot loss curve
plt.figure(figsize=(8, 4))
plt.plot(train_losses, color='#c9973a', marker='o', linewidth=2)
plt.xlabel('Epoch'); plt.ylabel('CrossEntropy Loss')
plt.title('RoBERTa Unfair Clause Detector — Training Loss')
plt.tight_layout()
plt.savefig('/content/roberta_loss_curve.png', dpi=150)
plt.show()
print("✅ Training complete!")


# ════════════════════════════════════════════════════════════════
# CELL 7 — Evaluate on Test Set
# ════════════════════════════════════════════════════════════════

from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)
import seaborn as sns

# Load best model
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['label'].to(DEVICE)
        preds = torch.argmax(model(ids, mask), dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(lbls.cpu().tolist())

acc = accuracy_score(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
f1_binary = f1_score(all_labels, all_preds, average='binary', zero_division=0)

print(f"\n📊 Unfair Clause Detector — Test Results:")
print(f"   Accuracy      : {acc*100:.2f}%")
print(f"   Macro F1      : {f1_macro:.4f}  (Overall performance across both classes)")
print(f"   Binary F1     : {f1_binary:.4f}  (Performance specifically on UNFAIR class)")
print(f"\n{classification_report(all_labels, all_preds, target_names=['SAFE','UNFAIR'])}")

# Quick sanity test
print("\n🧪 Sanity Check:")
test_sentences = [
    "This agreement takes effect on January 1st, 2024.",
    "The company may terminate your account at any time without notice or refund.",
    "Payment is due within 30 days of the invoice date.",
    "User waives all rights to file a lawsuit or class action under any circumstances.",
]
for s in test_sentences:
    score = model.predict_deception_score(s, tokenizer, DEVICE)
    label = "🚨 UNFAIR" if score >= 0.45 else "✅ SAFE"
    print(f"  {label} ({score:.2f}) — {s[:65]}...")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['SAFE', 'UNFAIR'],
            yticklabels=['SAFE', 'UNFAIR'])
plt.title('RoBERTa Unfair Clause Detector — Confusion Matrix')
plt.tight_layout()
plt.savefig('/content/roberta_confusion_matrix.png', dpi=150)
plt.show()


# ════════════════════════════════════════════════════════════════
# CELL 8 — !! SAVE TO DRIVE IMMEDIATELY — Run this right away !!
# ════════════════════════════════════════════════════════════════

import shutil, os

DRIVE_FOLDER = '/content/drive/MyDrive/Colab Notebooks/sem_project_231570/models'
os.makedirs(DRIVE_FOLDER, exist_ok=True)

# Save model weights — same filename as before for drop-in replacement
shutil.copy('/content/roberta_deception.pt',
            f'{DRIVE_FOLDER}/roberta_deception.pt')

# Save evaluation plots
shutil.copy('/content/roberta_loss_curve.png',       f'{DRIVE_FOLDER}/roberta_loss_curve.png')
shutil.copy('/content/roberta_confusion_matrix.png', f'{DRIVE_FOLDER}/roberta_confusion_matrix.png')

print("✅ All files saved to Google Drive!")
print(f"   Location: {DRIVE_FOLDER}/")
print("\nNext steps:")
print("  1. Download roberta_deception.pt from Drive to your PC")
print("  2. Put it in: legal_doc_verifier/backend/models/saved/roberta_deception.pt")
print("     (REPLACE the existing file — same filename!)")
print("  3. Restart the backend server: python -m uvicorn api:app --reload --port 8000")
print("  4. The frontend will now show UNFAIR CLAUSE RISK instead of Deception Score")
