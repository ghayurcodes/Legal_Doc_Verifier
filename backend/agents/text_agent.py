import torch


# ─────────────────────────────────────────────────────────────
# TEXT AGENT
# This agent is responsible for ONE thing only:
# Take a cleaned text string → run through RoBERTa → return score.
#
# Score: 0.0 = truthful, 1.0 = deceptive
# ─────────────────────────────────────────────────────────────

class TextAgent:

    def __init__(self, model, tokenizer, device):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.device    = device

    def analyze(self, text):
        """
        text: a cleaned string from PreprocessingAgent

        Returns unfair clause probability between 0.0 and 1.0.
        The closer to 1.0, the more likely the text contains deceptive claims.
        """
        score = self.model.predict_unfair_score(text, self.tokenizer, self.device)
        return score
