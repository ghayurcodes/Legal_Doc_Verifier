import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd


# ─────────────────────────────────────────────────────────────
# LABEL MAP
# LexGLUE UNFAIR-ToS dataset — simplified to binary:
#   SAFE   (0): clause has no unfair annotations
#   UNFAIR (1): clause has at least one unfair annotation
# ─────────────────────────────────────────────────────────────

LABEL_MAP = {
    'safe':   0,
    'unfair': 1,
}


# ─────────────────────────────────────────────────────────────
# PART 1: DATASET
# Reads the LexGLUE UNFAIR-ToS data and prepares text + labels.
# Each row contains one contract/ToS clause and its binary label:
#   0 = SAFE, 1 = UNFAIR
# Note: This class is used during training (in Colab notebooks).
# At inference time the model is loaded directly from the .pt file.
# ─────────────────────────────────────────────────────────────

class LIARDataset(Dataset):

    def __init__(self, tsv_path, tokenizer, max_len=128):
        # Load the TSV file — no header row, skip bad lines
        df = pd.read_csv(tsv_path, sep='\t', header=None, on_bad_lines='skip')

        self.texts  = df[2].astype(str).tolist()   # column 2 = statement text
        self.labels = [LABEL_MAP.get(str(l).strip(), 1) for l in df[1].tolist()]  # column 1 = label

        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the text — convert words into numbers RoBERTa understands
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',   # pad short sentences to max_len
            truncation=True,        # cut long sentences at max_len
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),       # token IDs
            'attention_mask': enc['attention_mask'].squeeze(0),  # 1=real token, 0=padding
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ─────────────────────────────────────────────────────────────
# PART 2: UNFAIR CLAUSE CLASSIFIER
# RoBERTa is a powerful language model pretrained on billions of words.
# We FULLY FINE-TUNE all its weights on the LexGLUE UNFAIR-ToS dataset.
# All 125M parameters are trainable — this is called "full fine-tuning".
# The custom 3-layer MLP head outputs SAFE (0) or UNFAIR (1).
# ─────────────────────────────────────────────────────────────

class UnfairClauseClassifier(nn.Module):
    """
    RoBERTa-base + 3-layer MLP head for binary legal clause classification.
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

    def predict_unfair_score(self, text: str, tokenizer, device) -> float:
        self.eval()
        enc = tokenizer(
            text, return_tensors='pt',
            max_length=256, truncation=True, padding='max_length',
        )
        enc = {k: v.to(device) for k, v in enc.items()
               if k in ('input_ids', 'attention_mask')}
        with torch.no_grad():
            logits = self.forward(enc['input_ids'], enc['attention_mask'])
        return torch.softmax(logits, dim=1)[0][1].item()   # P(UNFAIR)
