import torch


# ─────────────────────────────────────────────────────────────
# SIGNATURE AGENT
# This agent is responsible for ONE thing only:
# Take two signature tensors → run through Siamese CNN → return score.
#
# Score: 0.0 = forged, 1.0 = genuine
# ─────────────────────────────────────────────────────────────

class SignatureAgent:

    def __init__(self, model, device):
        self.model  = model.to(device)
        self.device = device

    def verify(self, ref_tensor, test_tensor):
        """
        ref_tensor  : tensor of the KNOWN genuine signature (reference)
        test_tensor : tensor of the signature being tested

        Returns similarity score between 0.0 and 1.0.
        The closer to 1.0, the more likely the test signature is genuine.
        """
        ref_tensor  = ref_tensor.to(self.device)
        test_tensor = test_tensor.to(self.device)

        score = self.model.get_similarity_score(ref_tensor, test_tensor)
        return score
