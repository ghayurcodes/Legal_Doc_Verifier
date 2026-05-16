import torch
import torch.nn as nn
from transformers import RobertaModel
# ─────────────────────────────────────────────────────────────
# UNFAIR CLAUSE CLASSIFIER
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
