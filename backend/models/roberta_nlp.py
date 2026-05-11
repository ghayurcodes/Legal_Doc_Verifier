import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd


# ─────────────────────────────────────────────────────────────
# LABEL MAP
# LIAR dataset has 6 labels — we simplify to binary:
#   Truthful  (0): true, mostly-true, half-true
#   Deceptive (1): barely-true, false, pants-fire
# ─────────────────────────────────────────────────────────────

LABEL_MAP = {
    'true':        0,
    'mostly-true': 0,
    'half-true':   0,
    'barely-true': 1,
    'false':       1,
    'pants-fire':  1
}


# ─────────────────────────────────────────────────────────────
# PART 1: DATASET
# Reads the LIAR .tsv files and prepares text + labels.
# Each row in the TSV is one political statement with a label.
# Column 1 = label (e.g. "false"), Column 2 = the statement text
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
# PART 2: DECEPTION CLASSIFIER
# RoBERTa is a powerful language model pretrained on billions of words.
# We FREEZE all its weights — it already understands language.
# We only TRAIN our custom head on top of it.
# This is called "feature extraction" — RoBERTa extracts features,
# our head makes the final decision.
# ─────────────────────────────────────────────────────────────

class DeceptionClassifier(nn.Module):

    def __init__(self, dropout=0.3):
        super().__init__()

        # Load pretrained RoBERTa — knows English language deeply
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        # Freeze ALL RoBERTa weights — we do NOT retrain it
        for param in self.roberta.parameters():
            param.requires_grad = False

        hidden_size = self.roberta.config.hidden_size  # 768 (RoBERTa output size)

        # Our custom trainable classification head
        # Input: 768 numbers from RoBERTa's [CLS] token
        # Output: 2 numbers (scores for truthful vs deceptive)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),  # 768 → 512
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),          # 512 → 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)             # 128 → 2 (truthful or deceptive)
        )

    def forward(self, input_ids, attention_mask):
        # Pass text through RoBERTa
        out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Take the [CLS] token output — it represents the whole sentence
        cls_embedding = out.last_hidden_state[:, 0, :]  # shape: (batch, 768)

        # Pass through our classifier head
        logits = self.classifier(cls_embedding)  # shape: (batch, 2)
        return logits

    def predict_deception_score(self, text, tokenizer, device):
        """
        Takes a plain text string.
        Returns a float: 0.0 = very truthful, 1.0 = very deceptive
        """
        self.eval()
        enc = tokenizer(
            text,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding='max_length'
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.forward(enc['input_ids'], enc['attention_mask'])

        # Convert raw scores to probabilities (they sum to 1.0)
        probs = torch.softmax(logits, dim=1)

        # Return probability of class 1 (deceptive)
        return probs[0][1].item()
