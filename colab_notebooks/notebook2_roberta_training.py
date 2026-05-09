# ╔══════════════════════════════════════════════════════════════╗
# ║   COLAB NOTEBOOK 2 — RoBERTa Deception Classifier Training  ║
# ║   Copy each CELL block into a separate Colab cell            ║
# ║   Runtime: GPU (T4) | Estimated time: ~30-40 minutes        ║
# ╚══════════════════════════════════════════════════════════════╝
#
# BEFORE RUNNING:
#   1. Upload liar_dataset.zip to Google Drive > Colab Notebooks > sem_project_231570/
#   2. Set Runtime > Change runtime type > T4 GPU
#   3. Run cells ONE BY ONE from top to bottom



# ════════════════════════════════════════════════════════════════
# CELL 1 — Install Required Libraries
# ════════════════════════════════════════════════════════════════

!pip install transformers --quiet
!pip install torch --quiet
!pip install scikit-learn matplotlib seaborn pandas --quiet

print("✅ All libraries installed!")


# ════════════════════════════════════════════════════════════════
# CELL 2 — Mount Google Drive
# ════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

print("✅ Google Drive mounted!")


# ════════════════════════════════════════════════════════════════
# CELL 3 — Extract LIAR Dataset from Drive
# ════════════════════════════════════════════════════════════════

import zipfile
import os

# liar_dataset.zip is already in your Drive at: Colab Notebooks > sem_project_231570
LIAR_ZIP   = '/content/drive/MyDrive/Colab Notebooks/sem_project_231570/liar_dataset.zip'
EXTRACT_TO = '/content/liar_data'

print("Extracting LIAR dataset...")
with zipfile.ZipFile(LIAR_ZIP, 'r') as z:
    z.extractall(EXTRACT_TO)

# Auto-detect where the TSV files actually landed
# (the zip might extract directly or into a subfolder)
def find_tsv(base_dir, filename):
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

TRAIN_PATH = find_tsv(EXTRACT_TO, 'train.tsv')
TEST_PATH  = find_tsv(EXTRACT_TO, 'test.tsv')
VALID_PATH = find_tsv(EXTRACT_TO, 'valid.tsv')

# Verify files found
for name, path in [('train.tsv', TRAIN_PATH), ('test.tsv', TEST_PATH), ('valid.tsv', VALID_PATH)]:
    if path:
        print(f"✅ {name} found at: {path} — {os.path.getsize(path)/1024:.1f} KB")
    else:
        print(f"❌ {name} NOT FOUND — check your zip file!")


# ════════════════════════════════════════════════════════════════
# CELL 4 — Paste Model Code (RoBERTa Classifier + Dataset)
# ════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

# Label mapping — simplify 6 LIAR labels to binary
LABEL_MAP = {
    'true':        0,
    'mostly-true': 0,
    'half-true':   0,
    'barely-true': 1,
    'false':       1,
    'pants-fire':  1
}


class LIARDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_len=256):
        df = pd.read_csv(tsv_path, sep='\t', header=None, on_bad_lines='skip')
        self.texts  = df[2].fillna('').astype(str).tolist()
        self.labels = [LABEL_MAP.get(str(l).strip(), 1) for l in df[1].tolist()]
        self.tokenizer = tokenizer
        self.max_len   = max_len

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


class DeceptionClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        # Full fine-tuning — unfreeze ALL RoBERTa layers
        # This is what research shows gives best results on hard text datasets
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
            nn.Linear(128, 2)   # 2 classes: truthful vs deceptive
        )

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = out.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_embedding)

    def predict_deception_score(self, text, tokenizer, device):
        self.eval()
        enc = tokenizer(text, return_tensors='pt', max_length=128,
                        truncation=True, padding='max_length')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.forward(enc['input_ids'], enc['attention_mask'])
        probs = torch.softmax(logits, dim=1)
        return probs[0][1].item()


print("✅ Model classes defined!")


# ════════════════════════════════════════════════════════════════
# CELL 5 — Load Tokenizer and Datasets
# ════════════════════════════════════════════════════════════════

print("Loading RoBERTa tokenizer... (downloads ~500MB on first run)")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

BATCH_SIZE = 16

train_ds = LIARDataset(TRAIN_PATH, tokenizer)
test_ds  = LIARDataset(TEST_PATH,  tokenizer)
valid_ds = LIARDataset(VALID_PATH, tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ Train: {len(train_ds)} samples")
print(f"✅ Test : {len(test_ds)}  samples")
print(f"✅ Valid: {len(valid_ds)} samples")


# ════════════════════════════════════════════════════════════════
# CELL 6 — Train RoBERTa Classifier
# ════════════════════════════════════════════════════════════════

import torch.optim as optim
import matplotlib.pyplot as plt

EPOCHS    = 6
LR        = 1e-5   # Lower LR required for full fine-tuning of all RoBERTa layers
DROPOUT   = 0.3
PATIENCE  = 2
SAVE_PATH = '/content/roberta_deception.pt'

model     = DeceptionClassifier(dropout=DROPOUT).to(DEVICE)

# Weighted loss to fix class imbalance (LIAR has more Truthful than Deceptive)
# Count class distribution in training set
import numpy as np
train_labels = [LABEL_MAP.get(str(l).strip(), 1) for l in pd.read_csv(TRAIN_PATH, sep='\t', header=None, on_bad_lines='skip')[1].tolist()]
class_counts = np.bincount(train_labels)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(DEVICE)
class_weights = class_weights / class_weights.sum()  # normalize
print(f"Class weights — Truthful: {class_weights[0]:.3f}, Deceptive: {class_weights[1]:.3f}")
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Train the classifier head AND the last 2 layers of RoBERTa
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# LR Warmup Scheduler — standard practice for transformer fine-tuning
# LR starts at 0, ramps up for 10% of training, then linearly decays to 0
from transformers import get_linear_schedule_with_warmup
total_steps   = len(train_loader) * EPOCHS
warmup_steps  = total_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

train_losses = []
best_val_acc = 0.0
no_improve   = 0   # early stopping counter

print(f"Training RoBERTa for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...")
print(f"Full fine-tuning ALL RoBERTa layers | max_len=256 | LR={LR}")
print(f"LR warmup for {warmup_steps} steps, then linear decay")
print("-" * 60)

for epoch in range(1, EPOCHS + 1):
    # ── Training ──────────────────────────────────────────────
    model.train()
    total_loss = 0

    for batch in train_loader:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, lbls)
        loss.backward()
        optimizer.step()
        scheduler.step()   # ← step the warmup scheduler each batch
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ── Validation accuracy ────────────────────────────────────
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in valid_loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbls = batch['label'].to(DEVICE)
            logits = model(ids, mask)
            preds  = torch.argmax(logits, dim=1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)

    val_acc = 100 * correct / total
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}")

    # Save best model + early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve   = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Best model saved (val acc: {best_val_acc:.2f}%)")
    else:
        no_improve += 1
        print(f"  − No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            print(f"  ⏹ Early stopping triggered at epoch {epoch}")
            break

# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(train_losses, color='coral', marker='o')
plt.xlabel('Epoch'); plt.ylabel('CrossEntropy Loss')
plt.title('RoBERTa Training Loss')
plt.tight_layout()
plt.savefig('/content/roberta_loss_curve.png', dpi=150)
plt.show()
print("✅ Training complete!")


# ════════════════════════════════════════════════════════════════
# CELL 7 — Evaluate on Test Set (Accuracy, F1, Confusion Matrix)
# ════════════════════════════════════════════════════════════════

from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)
import seaborn as sns
import numpy as np

# Load best model
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['label'].to(DEVICE)
        logits = model(ids, mask)
        preds  = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(lbls.cpu().tolist())

acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds, zero_division=0)

print(f"\n📊 RoBERTa Deception Classifier Results:")
print(f"   Accuracy : {acc*100:.2f}%")
print(f"   F1 Score : {f1:.4f}")
print(f"\n{classification_report(all_labels, all_preds, target_names=['Truthful','Deceptive'])}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Truthful','Deceptive'],
            yticklabels=['Truthful','Deceptive'])
plt.title('RoBERTa — Confusion Matrix')
plt.tight_layout()
plt.savefig('/content/roberta_confusion_matrix.png', dpi=150)
plt.show()


# ════════════════════════════════════════════════════════════════
# CELL 8 — Save Everything to Google Drive
# ════════════════════════════════════════════════════════════════

# !! RUN THIS CELL IMMEDIATELY AFTER TRAINING — before session expires !!

import shutil

DRIVE_FOLDER = '/content/drive/MyDrive/Colab Notebooks/sem_project_231570/models'
os.makedirs(DRIVE_FOLDER, exist_ok=True)

# Save model weights
shutil.copy('/content/roberta_deception.pt',
            f'{DRIVE_FOLDER}/roberta_deception.pt')

# Save plots
shutil.copy('/content/roberta_loss_curve.png',      f'{DRIVE_FOLDER}/roberta_loss_curve.png')
shutil.copy('/content/roberta_confusion_matrix.png', f'{DRIVE_FOLDER}/roberta_confusion_matrix.png')

print("✅ All files saved to Google Drive!")
print(f"   Location: {DRIVE_FOLDER}")
print("\nNext steps:")
print("  1. Download roberta_deception.pt from Drive to your PC")
print("  2. Put it in: legal_doc_verifier/models/saved/")
print("  3. Run the Gradio demo: python demo/app.py")
