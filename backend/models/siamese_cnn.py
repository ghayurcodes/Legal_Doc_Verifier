import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
# ─────────────────────────────────────────────────────────────
# PART 1: EMBEDDING NETWORK (One Tower)
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
# PART 2: SIAMESE NETWORK (Two Towers, Shared Weights)
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

