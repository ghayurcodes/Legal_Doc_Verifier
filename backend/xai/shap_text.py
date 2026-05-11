import shap
import torch
import numpy as np


# ─────────────────────────────────────────────────────────────
# SHAP TEXT AGENT
# SHAP = SHapley Additive exPlanations
#
# In plain English:
#   1. We take the document text
#   2. We mask/hide different words one by one
#   3. We ask: "how much did the prediction CHANGE when this word was hidden?"
#   4. A big change = that word was very important
#
# For text: it shows which words pushed the score toward "deceptive"
# Example output:
#   "guaranteed" → +0.12  (pushed toward deceptive)
#   "payment"    → -0.05  (pushed toward truthful)
# ─────────────────────────────────────────────────────────────

class SHAPTextAgent:

    def __init__(self, model, tokenizer, device):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device

    def predict_fn(self, texts):
        """
        SHAP needs a function that:
          - Takes a list of text strings
          - Returns an array of probabilities for each

        This wrapper converts our RoBERTa model into that format.
        """
        self.model.eval()
        results = []

        for text in texts:
            enc = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=256,
                truncation=True,
                padding='max_length'
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(enc['input_ids'], enc['attention_mask'])

            # Convert to probabilities [prob_truthful, prob_deceptive]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            results.append(probs)

        return np.array(results)

    def explain(self, text, n_samples=100):
        """
        Runs SHAP on the given text.
        n_samples: how many word-masking experiments to run
                   (more = more accurate but slower)

        Returns SHAP values object containing per-token attributions.
        """
        # Text masker: SHAP will mask words using the tokenizer
        masker    = shap.maskers.Text(self.tokenizer)
        explainer = shap.Explainer(self.predict_fn, masker)

        shap_values = explainer([text], max_evals=n_samples)
        return shap_values

    def get_top_words(self, text, n=5):
        """
        Returns the top N most influential words for the deception score.

        Positive SHAP value → word pushed prediction toward DECEPTIVE
        Negative SHAP value → word pushed prediction toward TRUTHFUL
        """
        shap_values = self.explain(text)

        tokens = shap_values.data[0]              # the words/tokens
        values = shap_values.values[0, :, 1]      # SHAP values for class 1 (deceptive)

        # Filter out punctuation, single chars, and whitespace tokens
        STOPWORDS = {'the','a','an','is','it','in','of','to','and','or','for',
                     'on','at','by','as','be','are','was','were','has','have',
                     'that','this','with','from','not','but','so','if','he',
                     'she','they','we','you','i','my','his','her','their','our'}

        word_scores = sorted(
            [
                (w, v) for w, v in zip(tokens, values)
                if len(w.strip()) > 1               # skip single chars like "."
                and w.strip().isalpha()             # skip punctuation/numbers
                and w.strip().lower() not in STOPWORDS
            ],
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return word_scores[:n]   # return top N words
