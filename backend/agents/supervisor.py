# ─────────────────────────────────────────────────────────────
# SUPERVISOR AGENT
# This is the BRAIN of the entire system.
# It receives scores from both models and makes the final decision.
#
# Flow:
#   SignatureAgent → sig_score (0=forged, 1=genuine)
#   TextAgent      -> unfair_score (0=safe, 1=unfair)
#   SupervisorAgent → combines both → AUTHENTIC or SUSPICIOUS
# ─────────────────────────────────────────────────────────────

class SupervisorAgent:

    def __init__(self, sig_weight=0.6, text_weight=0.4, threshold=0.5):
        """
        sig_weight  : how much the signature score matters (60%)
        text_weight : how much the text score matters (40%)
        threshold   : combined risk above this = SUSPICIOUS

        Signature gets more weight (0.6) because a forged signature
        is a stronger indicator of fraud than deceptive text alone.
        """
        self.sig_weight  = sig_weight
        self.text_weight = text_weight
        self.threshold   = threshold

    def decide(self, sig_score, text_unfair_score):
        """
        sig_score            : 0.0 (forged) → 1.0 (genuine)
        text_unfair_score : 0.0 (safe) -> 1.0 (unfair)

        Step 1: Convert sig_score to risk (flip it — genuine=low risk)
        Step 2: Combine both risks with weights
        Step 3: Compare combined risk to threshold
        Step 4: Return full report dictionary
        """

        # Convert signature similarity to RISK (higher = more suspicious)
        sig_risk  = 1.0 - sig_score           # genuine(1.0) → risk(0.0)
        text_risk = text_unfair_score          # unfair(1.0) -> risk(1.0)

        # Weighted combination
        combined_risk = (self.sig_weight * sig_risk) + (self.text_weight * text_risk)

        # Final verdict
        verdict = "SUSPICIOUS" if combined_risk >= self.threshold else "AUTHENTIC"

        # Confidence: how far from the 0.5 decision boundary (max 100%)
        confidence = round(abs(combined_risk - 0.5) * 200, 1)

        return {
            'verdict':           verdict,
            'combined_risk':     round(combined_risk, 3),
            'signature_score':   round(sig_score, 3),
            'unfair_score':   round(text_unfair_score, 3),
            'signature_verdict': 'FORGED'    if sig_score  < 0.70 else 'GENUINE',
            'text_verdict':      'DECEPTIVE' if text_risk  >= 0.45 else 'TRUTHFUL',
            'confidence':        confidence
        }
