import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import os, random


# ─────────────────────────────────────────────────────────────
# PART 1: DATASET
# Loads pairs of signature images and labels them:
#   label = 0 → genuine pair (both signatures are real)
#   label = 1 → forged pair (one real, one fake)
# ─────────────────────────────────────────────────────────────

class SiameseDataset(Dataset):

    def __init__(self, genuine_imgs, forged_imgs, transform=None):
        """
        genuine_imgs: either a folder path (str) OR a pre-split list of image paths
        forged_imgs : either a folder path (str) OR a pre-split list of image paths

        Accepting lists allows proper image-level train/val splitting so
        the same image never appears in both train and val sets.
        """
        # Accept either a directory path or a pre-split list
        if isinstance(genuine_imgs, str):
            self.genuine = sorted([
                os.path.join(genuine_imgs, f)
                for f in os.listdir(genuine_imgs)
                if f.endswith(('.png', '.jpg', '.PNG'))
            ])
        else:
            self.genuine = genuine_imgs   # already a list — use directly

        if isinstance(forged_imgs, str):
            self.forged = sorted([
                os.path.join(forged_imgs, f)
                for f in os.listdir(forged_imgs)
                if f.endswith(('.png', '.jpg', '.PNG'))
            ])
        else:
            self.forged = forged_imgs     # already a list — use directly

        # How to preprocess every image before feeding to model
        self.transform = transform or T.Compose([
            T.Resize((128, 256)),               # resize to fixed size
            T.Grayscale(num_output_channels=3), # keep 3 channels (VGG needs RGB)
            T.ToTensor(),                       # convert image to numbers (0-1)
            T.Normalize(mean=[0.5, 0.5, 0.5],  # normalize so values are centered
                        std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        # Dataset size = twice the genuine images (half genuine pairs, half forged)
        return len(self.genuine) * 2

    def __getitem__(self, idx):
        if idx % 2 == 0:
            # Even index → genuine pair (label = 0, similar)
            i = idx // 2 % len(self.genuine)
            j = random.randint(0, len(self.genuine) - 1)
            img1 = self.transform(Image.open(self.genuine[i]).convert('RGB'))
            img2 = self.transform(Image.open(self.genuine[j]).convert('RGB'))
            label = torch.tensor(0.0)
        else:
            # Odd index → forged pair (label = 1, dissimilar)
            i = idx // 2 % len(self.genuine)
            j = random.randint(0, len(self.forged) - 1)
            img1 = self.transform(Image.open(self.genuine[i]).convert('RGB'))
            img2 = self.transform(Image.open(self.forged[j]).convert('RGB'))
            label = torch.tensor(1.0)

        return img1, img2, label


# ─────────────────────────────────────────────────────────────
# PART 2: EMBEDDING NETWORK (One Tower)
# This is ONE branch of the Siamese network.
# It takes an image and converts it into a list of numbers
# called an "embedding" — a compact representation of the signature.
# ─────────────────────────────────────────────────────────────

class EmbeddingNet(nn.Module):

    def __init__(self, embedding_dim=128):
        super().__init__()

        # Load VGG16 pretrained on ImageNet — it already knows edges, textures
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Freeze early layers (they already learned basic features)
        # Only train the last conv block (layer 24+)
        for i, param in enumerate(vgg.features.parameters()):
            if i < 24:
                param.requires_grad = False

        self.features = vgg.features   # convolutional layers
        self.avgpool  = vgg.avgpool    # average pooling

        # Our custom trainable head on top of VGG
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),  # flatten → 1024 neurons
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),             # randomly drop 50% neurons (prevents overfitting)
            nn.Linear(1024, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.3),
            nn.Linear(256, embedding_dim)  # final output: embedding vector
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.head(x)
        return x


# ─────────────────────────────────────────────────────────────
# PART 3: SIAMESE NETWORK (Two Towers, Shared Weights)
# Both towers are the SAME EmbeddingNet with SHARED weights.
# Image 1 → Tower → Embedding 1
# Image 2 → Tower → Embedding 2
# Then we compare the two embeddings.
# ─────────────────────────────────────────────────────────────

class SiameseNet(nn.Module):

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.tower = EmbeddingNet(embedding_dim)  # one tower, shared by both images

    def forward(self, img1, img2):
        emb1 = self.tower(img1)   # embedding for image 1
        emb2 = self.tower(img2)   # embedding for image 2
        return emb1, emb2

    def get_similarity_score(self, img1, img2):
        """
        Returns a score: 0.0 = completely different, 1.0 = identical.
        Uses exp(-dist) instead of sigmoid for a wider, more readable score range:
          - Same signature    → score close to 1.0
          - Genuine pair      → score ~0.75–0.95
          - Forged pair       → score ~0.30–0.65
        """
        self.eval()
        with torch.no_grad():
            emb1, emb2 = self.forward(img1, img2)
            dist = torch.nn.functional.pairwise_distance(emb1, emb2)
            similarity = torch.exp(-dist)   # wider range than sigmoid
        return similarity.item()


# ─────────────────────────────────────────────────────────────
# PART 4: CONTRASTIVE LOSS
# Custom loss function for Siamese training.
# Genuine pair (label=0): push embeddings CLOSER together
# Forged  pair (label=1): push embeddings FURTHER apart
# ─────────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin  # minimum distance we want for forged pairs

    def forward(self, emb1, emb2, label):
        dist = torch.nn.functional.pairwise_distance(emb1, emb2)

        # Genuine pair loss: minimize distance
        loss_genuine = (1 - label) * dist.pow(2)

        # Forged pair loss: maximize distance (up to margin)
        loss_forged  = label * torch.clamp(self.margin - dist, min=0.0).pow(2)

        return (loss_genuine + loss_forged).mean()
