# ╔══════════════════════════════════════════════════════════════╗
# ║   COLAB NOTEBOOK 1 — HPO Agent + Siamese CNN Training       ║
# ║   Copy each CELL block into a separate Colab cell            ║
# ║   Runtime: GPU (T4) | Estimated time: ~1.5 - 2 hours        ║
# ╚══════════════════════════════════════════════════════════════╝
#
# BEFORE RUNNING:
#   1. Upload archive.zip (CEDAR) to Google Drive > Colab Notebooks > sem_project_231570/
#   2. Set Runtime > Change runtime type > T4 GPU
#   3. Run cells ONE BY ONE from top to bottom



# ════════════════════════════════════════════════════════════════
# CELL 1 — Install Required Libraries
# ════════════════════════════════════════════════════════════════

# Paste this into Cell 1 and run it first

!pip install torch torchvision --quiet
!pip install optuna --quiet
!pip install scikit-learn matplotlib seaborn --quiet
!pip install Pillow opencv-python --quiet

print("✅ All libraries installed!")


# ════════════════════════════════════════════════════════════════
# CELL 2 — Mount Google Drive
# ════════════════════════════════════════════════════════════════

# This connects your Google Drive to Colab
# A popup will ask you to sign in — click Allow

from google.colab import drive
drive.mount('/content/drive')

# After mounting, your Drive is at: /content/drive/MyDrive/
print("✅ Google Drive mounted!")


# ════════════════════════════════════════════════════════════════
# CELL 3 — Extract CEDAR Dataset from Drive
# ════════════════════════════════════════════════════════════════

import zipfile
import os

# Path to your zip file in Google Drive
# Make sure archive.zip is uploaded to: Colab Notebooks > sem_project_231570
CEDAR_ZIP  = '/content/drive/MyDrive/Colab Notebooks/sem_project_231570/archive.zip'
EXTRACT_TO = '/content/cedar_data'

print("Extracting CEDAR signatures... (may take 1-2 minutes)")
with zipfile.ZipFile(CEDAR_ZIP, 'r') as z:
    z.extractall(EXTRACT_TO)

# Find the full_org and full_forg directories
GENUINE_DIR = '/content/cedar_data/signatures/full_org'
FORGED_DIR  = '/content/cedar_data/signatures/full_forg'

# Verify
genuine_count = len(os.listdir(GENUINE_DIR))
forged_count  = len(os.listdir(FORGED_DIR))
print(f"✅ Genuine images: {genuine_count}")
print(f"✅ Forged  images: {forged_count}")


# ════════════════════════════════════════════════════════════════
# CELL 4 — Paste Model Code (SiameseNet + Dataset + Loss)
# ════════════════════════════════════════════════════════════════

# We paste the model code directly here so Colab has it available
# (no need to upload .py files)

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")


class SiameseDataset(Dataset):
    def __init__(self, genuine_imgs, forged_imgs, transform=None):
        import re, collections
        def get_id(p):
            m = re.search(r'_(\d+)_', os.path.basename(p))
            return int(m.group(1)) if m else 1

        # Group images by Signer ID
        self.gen_dict = collections.defaultdict(list)
        gen_list = genuine_imgs if isinstance(genuine_imgs, list) else [os.path.join(genuine_imgs, f) for f in os.listdir(genuine_imgs) if f.endswith(('.png', '.jpg', '.PNG'))]
        for p in gen_list: self.gen_dict[get_id(p)].append(p)

        self.forg_dict = collections.defaultdict(list)
        forg_list = forged_imgs if isinstance(forged_imgs, list) else [os.path.join(forged_imgs, f) for f in os.listdir(forged_imgs) if f.endswith(('.png', '.jpg', '.PNG'))]
        for p in forg_list: self.forg_dict[get_id(p)].append(p)

        self.signers = list(self.gen_dict.keys())
        self.length = sum(len(v) for v in self.gen_dict.values()) * 2

        self.transform = transform or T.Compose([
            T.Resize((128, 256)),
            T.Grayscale(num_output_channels=3),
            # Strong augmentation to prevent the model from memorizing 
            # background color or scanner artifacts (a known issue in CEDAR)
            T.ColorJitter(brightness=0.4, contrast=0.4),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Pick a random signer from the ones available in this split
        signer = random.choice(self.signers)
        
        # Image 1 is ALWAYS a genuine signature from this signer
        img1_path = random.choice(self.gen_dict[signer])
        
        if idx % 2 == 0:
            # Match: Image 2 is ANOTHER genuine signature from the SAME signer
            img2_path = random.choice(self.gen_dict[signer])
            # Prevent picking the exact same image
            while img2_path == img1_path and len(self.gen_dict[signer]) > 1:
                img2_path = random.choice(self.gen_dict[signer])
            label = torch.tensor(0.0)
        else:
            # Mismatch: Image 2 is a FORGED signature from the SAME signer
            img2_path = random.choice(self.forg_dict[signer])
            label = torch.tensor(1.0)
            
        img1 = self.transform(Image.open(img1_path).convert('RGB'))
        img2 = self.transform(Image.open(img2_path).convert('RGB'))
        return img1, img2, label


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128, dropout=0.5):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for i, param in enumerate(vgg.features.parameters()):
            if i < 24:
                param.requires_grad = False
        self.features = vgg.features
        self.avgpool  = vgg.avgpool
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.head(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=128, dropout=0.5):
        super().__init__()
        self.tower = EmbeddingNet(embedding_dim, dropout)

    def forward(self, img1, img2):
        return self.tower(img1), self.tower(img2)

    def get_similarity_score(self, img1, img2):
        self.eval()
        with torch.no_grad():
            emb1, emb2 = self.forward(img1, img2)
            dist = torch.nn.functional.pairwise_distance(emb1, emb2)
            return torch.sigmoid(-dist + 1).item()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        dist = torch.nn.functional.pairwise_distance(emb1, emb2)
        loss_genuine = (1 - label) * dist.pow(2)
        loss_forged  = label * torch.clamp(self.margin - dist, min=0.0).pow(2)
        return (loss_genuine + loss_forged).mean()


print("✅ Model classes defined!")


# ════════════════════════════════════════════════════════════════
# CELL 5 — HPO Agent (Optuna — finds best hyperparameters)
# ════════════════════════════════════════════════════════════════

# This runs 10 quick trials (3 epochs each) to find the best settings
# Takes about 30-45 minutes — watch the trial results print below

import optuna
import torch.optim as optim

N_TRIALS = 10   # increase to 20 if you have extra time

def objective(trial):
    # Optuna picks values from these ranges
    lr            = trial.suggest_float('lr',            1e-5, 1e-3, log=True)
    dropout       = trial.suggest_float('dropout',       0.2,  0.6)
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
    margin        = trial.suggest_float('margin',        0.5,  2.0)
    batch_size    = trial.suggest_categorical('batch_size', [16, 32])

    # Use the same Writer-Independent split for HPO to be accurate
    all_gen = sorted([os.path.join(GENUINE_DIR, f) for f in os.listdir(GENUINE_DIR) if f.endswith(('.png','.jpg','.PNG'))])
    all_forg = sorted([os.path.join(FORGED_DIR, f) for f in os.listdir(FORGED_DIR) if f.endswith(('.png','.jpg','.PNG'))])
    
    import re
    def get_sid(p):
        m = re.search(r'_(\d+)_', os.path.basename(p))
        return int(m.group(1)) if m else 1
        
    train_gen = [f for f in all_gen if get_sid(f) <= 45]
    val_gen   = [f for f in all_gen if get_sid(f) > 45]
    train_forg= [f for f in all_forg if get_sid(f) <= 45]
    val_forg  = [f for f in all_forg if get_sid(f) > 45]

    train_ds = SiameseDataset(train_gen, train_forg)
    val_ds   = SiameseDataset(val_gen, val_forg)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model     = SiameseNet(embedding_dim=embedding_dim, dropout=dropout).to(DEVICE)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Quick 3-epoch training to judge the hyperparameters
    for epoch in range(3):
        model.train()
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, label)
            loss.backward()
            optimizer.step()

    # Measure validation loss — this is what Optuna minimises
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            emb1, emb2 = model(img1, img2)
            val_loss += criterion(emb1, emb2, label).item()
    return val_loss / len(val_loader)


print(f"Starting HPO with {N_TRIALS} trials...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS)

BEST_PARAMS = study.best_params
print("\n✅ HPO Complete! Best hyperparameters:")
for k, v in BEST_PARAMS.items():
    print(f"   {k}: {v}")


# ════════════════════════════════════════════════════════════════
# CELL 6 — Full Siamese CNN Training with Best Hyperparameters
# ════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

# Use best params found by HPO (or set manually if you skipped HPO)
LR            = BEST_PARAMS.get('lr',            1e-4)
DROPOUT       = BEST_PARAMS.get('dropout',       0.5)
EMBEDDING_DIM = BEST_PARAMS.get('embedding_dim', 128)
MARGIN        = BEST_PARAMS.get('margin',        1.0)
BATCH_SIZE    = BEST_PARAMS.get('batch_size',    32)
EPOCHS        = 15

print(f"Training with: lr={LR:.5f}, dropout={DROPOUT}, emb_dim={EMBEDDING_DIM}, margin={MARGIN}, batch={BATCH_SIZE}")

# ── Writer-Independent Split ──────────────────────────────────
# Split by SIGNER ID extracted from the filename.
# Signers 1-45 → Training   (model learns general forgery patterns)
# Signers 46-55 → Validation (model sees completely unseen people)
# This is the gold standard evaluation for CEDAR signature datasets.
import re

def get_signer_id(filepath):
    # CEDAR filenames: "original_46_2.png" → signer ID = 46
    match = re.search(r'_(\d+)_', os.path.basename(filepath))
    return int(match.group(1)) if match else 1

all_genuine = sorted([os.path.join(GENUINE_DIR, f) for f in os.listdir(GENUINE_DIR) if f.endswith(('.png','.jpg','.PNG'))])
all_forged  = sorted([os.path.join(FORGED_DIR,  f) for f in os.listdir(FORGED_DIR)  if f.endswith(('.png','.jpg','.PNG'))])

train_genuine = [f for f in all_genuine if get_signer_id(f) <= 45]
val_genuine   = [f for f in all_genuine if get_signer_id(f) >  45]
train_forged  = [f for f in all_forged  if get_signer_id(f) <= 45]
val_forged    = [f for f in all_forged  if get_signer_id(f) >  45]

print(f"Train: {len(train_genuine)} genuine + {len(train_forged)} forged  (Signers 1–45)")
print(f"Val  : {len(val_genuine)}  genuine + {len(val_forged)}  forged  (Signers 46–55)")
print("Writer-Independent split active ✅ — zero signer overlap")

train_ds = SiameseDataset(train_genuine, train_forged)
val_ds   = SiameseDataset(val_genuine,   val_forged)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Build model
model     = SiameseNet(embedding_dim=EMBEDDING_DIM, dropout=DROPOUT).to(DEVICE)
criterion = ContrastiveLoss(margin=MARGIN)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

train_losses, val_losses = [], []
best_val_loss = float('inf')
SAVE_PATH = '/content/siamese_best.pt'

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    total_train = 0
    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, label)
        loss.backward()
        optimizer.step()
        total_train += loss.item()

    # Validation
    model.eval()
    total_val = 0
    correct, total = 0, 0
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, label)
            total_val += loss.item()
            dist = torch.nn.functional.pairwise_distance(emb1, emb2)
            pred = (dist > 0.5).float()
            correct += (pred == label).sum().item()
            total   += label.size(0)

    avg_train = total_train / len(train_loader)
    avg_val   = total_val   / len(val_loader)
    val_acc   = 100 * correct / total
    train_losses.append(avg_train)
    val_losses.append(avg_val)
    scheduler.step()

    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {val_acc:.2f}%")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Best model saved (val loss: {best_val_loss:.4f})")

# Plot loss curves
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss', color='steelblue')
plt.plot(val_losses,   label='Val Loss',   color='coral')
plt.xlabel('Epoch'); plt.ylabel('Contrastive Loss')
plt.title('Siamese CNN — Training Curves')
plt.legend(); plt.tight_layout()
plt.savefig('/content/siamese_loss_curve.png', dpi=150)
plt.show()
print("✅ Training complete!")


# ════════════════════════════════════════════════════════════════
# CELL 7 — Evaluate Model (Accuracy, F1, Confusion Matrix, ROC)
# ════════════════════════════════════════════════════════════════

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import numpy as np

# Load best saved model
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval()

all_preds, all_labels, all_scores = [], [], []

with torch.no_grad():
    for img1, img2, label in val_loader:
        img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
        emb1, emb2 = model(img1, img2)
        dist = torch.nn.functional.pairwise_distance(emb1, emb2)
        pred = (dist > 0.5).float()
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(label.cpu().tolist())
        all_scores.extend(dist.cpu().tolist())

acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_scores)

print(f"\n📊 Siamese CNN Results:")
print(f"   Accuracy : {acc*100:.2f}%")
print(f"   F1 Score : {f1:.4f}")
print(f"   AUC      : {auc:.4f}")
print(f"\n{classification_report(all_labels, all_preds, target_names=['Genuine','Forged'])}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Genuine','Forged'], yticklabels=['Genuine','Forged'])
plt.title('Siamese CNN — Confusion Matrix')
plt.tight_layout()
plt.savefig('/content/siamese_confusion_matrix.png', dpi=150)
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_scores)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='steelblue', label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Siamese CNN — ROC Curve')
plt.legend(); plt.tight_layout()
plt.savefig('/content/siamese_roc_curve.png', dpi=150)
plt.show()


# ════════════════════════════════════════════════════════════════
# CELL 8 — Save Everything to Google Drive
# ════════════════════════════════════════════════════════════════

# !! RUN THIS CELL IMMEDIATELY AFTER TRAINING — before session expires !!

import shutil, os

# Create folder in Drive
DRIVE_FOLDER = '/content/drive/MyDrive/Colab Notebooks/sem_project_231570/models'
os.makedirs(DRIVE_FOLDER, exist_ok=True)

# Save model weights
shutil.copy('/content/siamese_best.pt',
            f'{DRIVE_FOLDER}/siamese_best.pt')

# Save plots
shutil.copy('/content/siamese_loss_curve.png',      f'{DRIVE_FOLDER}/siamese_loss_curve.png')
shutil.copy('/content/siamese_confusion_matrix.png', f'{DRIVE_FOLDER}/siamese_confusion_matrix.png')
shutil.copy('/content/siamese_roc_curve.png',        f'{DRIVE_FOLDER}/siamese_roc_curve.png')

print("✅ All files saved to Google Drive!")
print(f"   Location: {DRIVE_FOLDER}")
print("\nNext steps:")
print("  1. Download siamese_best.pt from Drive to your PC")
print("  2. Put it in: legal_doc_verifier/models/saved/")
print("  3. Open Colab Notebook 2 for RoBERTa training")
