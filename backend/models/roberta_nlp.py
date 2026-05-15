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

    def predict_deception_score(self, text: str, tokenizer, device) -> float:
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
