import torch


# ─────────────────────────────────────────────────────────────
# XAI AGENT
# XAI = Explainable AI
# This agent coordinates BOTH explainability methods:
#   - Grad-CAM  → for the signature image (which strokes look suspicious?)
#   - SHAP      → for the text (which words triggered deception flag?)
#
# It is a thin wrapper — the actual logic lives in:
#   xai/gradcam.py    → GradCAMAgent
#   xai/shap_text.py  → SHAPTextAgent
# ─────────────────────────────────────────────────────────────

class XAIAgent:

    def __init__(self, gradcam_agent, shap_agent):
        self.gradcam_agent = gradcam_agent   # handles image explanation
        self.shap_agent    = shap_agent      # handles text explanation

    def explain_signature(self, img_tensor, img_path):
        """
        Runs Grad-CAM on the test signature image.
        Returns a PIL Image with a heatmap overlay showing
        which parts of the signature the model focused on.
        """
        return self.gradcam_agent.generate(img_tensor, img_path)

    def explain_text(self, text, n=5):
        """
        Runs SHAP on the document text.
        Returns the top N words that most influenced the deception score.
        Positive score = pushed toward deceptive.
        Negative score = pushed toward truthful.
        """
        return self.shap_agent.get_top_words(text, n=n)
