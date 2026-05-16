"""
Siamese CNN -- Local Inference Test
Tests the trained model on CEDAR test signers (46-55).
These signers were NEVER seen during training (writer-independent protocol).

Test types:
  Genuine pairs : same signer, two different genuine signatures -> score >= 0.70
  Forged pairs  : genuine ref vs forged attempt for same signer -> score <  0.70

Run:  cd backend && python test_siamese.py
"""

import os
import sys
import random
import torch

# -- Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.siamese_cnn import SiameseNet
from agents.preprocessing import PreprocessingAgent

# -- Load model
DEVICE = torch.device("cpu")
SAVED  = os.path.join(BASE_DIR, "models", "saved")

print("Loading Siamese CNN model...")
model = SiameseNet(embedding_dim=128)
model.load_state_dict(
    torch.load(os.path.join(SAVED, "siamese_best.pt"), map_location=DEVICE)
)
model.eval()
print("[OK] Siamese CNN loaded\n")

prep = PreprocessingAgent()

# -- Data paths
GENUINE_DIR = os.path.join(BASE_DIR, "data", "cedar", "full_org")
FORGED_DIR  = os.path.join(BASE_DIR, "data", "cedar", "full_forg")

# -- Config
TEST_SIGNERS = list(range(46, 56))
THRESHOLD    = 0.70

print("=" * 72)
print("SIAMESE CNN - WRITER-INDEPENDENT EVALUATION (Test Signers 46-55)")
print("=" * 72)
print("Threshold: score >= %.2f -> GENUINE | score < %.2f -> FORGED" % (THRESHOLD, THRESHOLD))
print()

results = []

for signer in TEST_SIGNERS:
    genuine_imgs = sorted([
        os.path.join(GENUINE_DIR, f)
        for f in os.listdir(GENUINE_DIR)
        if f.startswith("original_%d_" % signer) and f.endswith(".png")
    ])
    forged_imgs = sorted([
        os.path.join(FORGED_DIR, f)
        for f in os.listdir(FORGED_DIR)
        if f.startswith("forgeries_%d_" % signer) and f.endswith(".png")
    ])

    if len(genuine_imgs) < 2 or len(forged_imgs) < 1:
        print("  [SKIP] Signer %d: not enough images" % signer)
        continue

    print("--- Signer %d (%d genuine, %d forged) ---" % (signer, len(genuine_imgs), len(forged_imgs)))

    # Test 1: Genuine Pairs
    num_genuine_tests = min(3, len(genuine_imgs) // 2)
    for t in range(num_genuine_tests):
        i, j = random.sample(range(len(genuine_imgs)), 2)
        ref_path  = genuine_imgs[i]
        test_path = genuine_imgs[j]

        ref_t  = prep.prepare_signature(ref_path)
        test_t = prep.prepare_signature(test_path)
        score  = model.get_similarity_score(ref_t, test_t)

        verdict = "GENUINE" if score >= THRESHOLD else "FORGED"
        correct = verdict == "GENUINE"
        marker  = "OK" if correct else "MISS"

        results.append({"signer": signer, "type": "genuine", "score": score, "correct": correct})
        print("  [Genuine] %s vs %s: score=%.4f -> %s [%s]" % (
            os.path.basename(ref_path), os.path.basename(test_path), score, verdict, marker))

    # Test 2: Forged Pairs
    ref_path = genuine_imgs[0]
    num_forged_tests = min(3, len(forged_imgs))
    test_forgeries = random.sample(forged_imgs, num_forged_tests)

    for forg_path in test_forgeries:
        ref_t  = prep.prepare_signature(ref_path)
        test_t = prep.prepare_signature(forg_path)
        score  = model.get_similarity_score(ref_t, test_t)

        verdict = "GENUINE" if score >= THRESHOLD else "FORGED"
        correct = verdict == "FORGED"
        marker  = "OK" if correct else "MISS"

        results.append({"signer": signer, "type": "forged", "score": score, "correct": correct})
        print("  [Forged]  %s vs %s: score=%.4f -> %s [%s]" % (
            os.path.basename(ref_path), os.path.basename(forg_path), score, verdict, marker))

    print()

# -- Summary
genuine_res = [r for r in results if r["type"] == "genuine"]
forged_res  = [r for r in results if r["type"] == "forged"]

g_correct = sum(1 for r in genuine_res if r["correct"])
f_correct = sum(1 for r in forged_res  if r["correct"])
total_correct = g_correct + f_correct
total = len(results)

g_scores = [r["score"] for r in genuine_res]
f_scores = [r["score"] for r in forged_res]

print("=" * 72)
print("SUMMARY")
print("=" * 72)
print("Total tests       : %d" % total)
print("Overall accuracy  : %d/%d (%.1f%%)" % (total_correct, total, 100*total_correct/total))
print()
print("Genuine pairs     : %d/%d correct" % (g_correct, len(genuine_res)))
print("  Avg score       : %.4f" % (sum(g_scores)/len(g_scores)))
print("  Min / Max       : %.4f / %.4f" % (min(g_scores), max(g_scores)))
print()
print("Forged pairs      : %d/%d correct" % (f_correct, len(forged_res)))
print("  Avg score       : %.4f" % (sum(f_scores)/len(f_scores)))
print("  Min / Max       : %.4f / %.4f" % (min(f_scores), max(f_scores)))
print()
print("Score separation  : genuine_avg=%.4f  forged_avg=%.4f  gap=%.4f" % (
    sum(g_scores)/len(g_scores),
    sum(f_scores)/len(f_scores),
    sum(g_scores)/len(g_scores) - sum(f_scores)/len(f_scores)
))
print("=" * 72)
