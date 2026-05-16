"""
Siamese CNN -- Inference Test Evaluation Charts
Generates and saves result images matching the style of the existing
roberta_evaluation.png and siamese training charts.

Run:  cd backend && python generate_siamese_inference_charts.py
"""

import os
import sys
import random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.siamese_cnn import SiameseNet
from agents.preprocessing import PreprocessingAgent

RESULTS_DIR = os.path.join(BASE_DIR, "results")
GENUINE_DIR = os.path.join(BASE_DIR, "data", "cedar", "full_org")
FORGED_DIR  = os.path.join(BASE_DIR, "data", "cedar", "full_forg")
DEVICE      = torch.device("cpu")
THRESHOLD   = 0.70
TEST_SIGNERS = list(range(46, 56))  # never-seen signers

# ── Load model ──────────────────────────────────────────────────
print("Loading Siamese CNN...")
model = SiameseNet(embedding_dim=128)
model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "models", "saved", "siamese_best.pt"), map_location=DEVICE)
)
model.eval()
prep = PreprocessingAgent()
print("[OK] Model loaded\n")

# ── Collect scores across all test signers ───────────────────────
print("Running inference on test signers 46-55...")
all_scores  = []
all_labels  = []   # 0 = genuine pair, 1 = forged pair
signer_accs = {}   # per-signer accuracy for bar chart

random.seed(42)

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
        continue

    correct = 0
    total   = 0

    # Genuine pairs (sample up to 6)
    for _ in range(min(6, len(genuine_imgs) // 2)):
        i, j = random.sample(range(len(genuine_imgs)), 2)
        t1 = prep.prepare_signature(genuine_imgs[i])
        t2 = prep.prepare_signature(genuine_imgs[j])
        score = model.get_similarity_score(t1, t2)
        all_scores.append(score)
        all_labels.append(0)  # genuine pair label
        correct += 1 if score >= THRESHOLD else 0
        total   += 1

    # Forged pairs (sample up to 6)
    ref_path = genuine_imgs[0]
    for forg_path in random.sample(forged_imgs, min(6, len(forged_imgs))):
        t1 = prep.prepare_signature(ref_path)
        t2 = prep.prepare_signature(forg_path)
        score = model.get_similarity_score(t1, t2)
        all_scores.append(score)
        all_labels.append(1)  # forged pair label
        correct += 1 if score < THRESHOLD else 0
        total   += 1

    signer_accs[signer] = 100.0 * correct / total
    print("  Signer %d: %d/%d correct (%.0f%%)" % (signer, correct, total, signer_accs[signer]))

scores = np.array(all_scores)
labels = np.array(all_labels)

# For ROC/PR: predict score as "forged risk" = 1 - score
risk_scores = 1.0 - scores

genuine_scores = scores[labels == 0]
forged_scores  = scores[labels == 1]

overall_preds  = (scores < THRESHOLD).astype(int)  # 1=forged prediction
overall_correct = np.sum(overall_preds == labels)
overall_acc    = 100.0 * overall_correct / len(labels)

print("\nTotal pairs: %d | Overall accuracy: %.1f%%" % (len(labels), overall_acc))

# ── CHART 1: Inference Evaluation (confusion matrix + ROC + score dist) ──
print("\nGenerating siamese_inference_evaluation.png ...")

fig = plt.figure(figsize=(15, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# --- Subplot 1: Confusion Matrix ---
ax1 = fig.add_subplot(gs[0])
preds = (scores < THRESHOLD).astype(int)   # 1=forged
cm = confusion_matrix(labels, preds)
# labels: 0=genuine pair, 1=forged pair
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['GENUINE', 'FORGED'],
            yticklabels=['GENUINE', 'FORGED'],
            linewidths=0.5, linecolor='white',
            annot_kws={"size": 14, "weight": "bold"})
ax1.set_xlabel('Predicted', fontsize=12)
ax1.set_ylabel('Actual', fontsize=12)
ax1.set_title('Confusion Matrix\n(Inference Test, Signers 46-55)', fontsize=12, fontweight='bold')

# --- Subplot 2: ROC Curve ---
ax2 = fig.add_subplot(gs[1])
fpr, tpr, _ = roc_curve(labels, risk_scores)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color='steelblue', lw=2, label='AUC = %.3f' % roc_auc)
ax2.plot([0, 1], [0, 1], 'k--', lw=1)
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve\n(Inference Test)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=11)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.02])

# --- Subplot 3: Score Distribution ---
ax3 = fig.add_subplot(gs[2])
ax3.hist(genuine_scores, bins=15, alpha=0.7, color='steelblue',  label='Genuine pairs (n=%d)' % len(genuine_scores), edgecolor='white')
ax3.hist(forged_scores,  bins=15, alpha=0.7, color='tomato',     label='Forged pairs  (n=%d)' % len(forged_scores),  edgecolor='white')
ax3.axvline(x=THRESHOLD, color='black', linestyle='--', lw=1.5, label='Threshold = %.2f' % THRESHOLD)
ax3.set_xlabel('Similarity Score', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Score Distribution\nGenuine vs Forged', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)

fig.suptitle(
    'Siamese CNN -- Inference Evaluation  |  Accuracy: %.1f%%  |  AUC: %.3f' % (overall_acc, roc_auc),
    fontsize=13, fontweight='bold', y=1.02
)

out_path = os.path.join(RESULTS_DIR, "siamese_inference_evaluation.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] %s" % out_path)


# ── CHART 2: Per-Signer Accuracy Bar Chart ──────────────────────
print("Generating siamese_per_signer_accuracy.png ...")

fig, ax = plt.subplots(figsize=(10, 5))

signers = list(signer_accs.keys())
accs    = [signer_accs[s] for s in signers]
colors  = ['#2ecc71' if a >= 80 else '#e74c3c' for a in accs]

bars = ax.bar([str(s) for s in signers], accs, color=colors, edgecolor='white', linewidth=0.8)

# value labels on bars
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.5,
            '%.0f%%' % acc, ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(y=overall_acc, color='steelblue', linestyle='--', lw=1.5,
           label='Overall avg: %.1f%%' % overall_acc)
ax.axhline(y=80.21, color='orange', linestyle=':', lw=1.5,
           label='Training accuracy: 80.21%')

ax.set_xlabel('Signer ID (Test Set — Never Seen During Training)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Siamese CNN -- Per-Signer Accuracy\nWriter-Independent Evaluation (Signers 46-55)', fontsize=13, fontweight='bold')
ax.set_ylim([0, 115])
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

out_path = os.path.join(RESULTS_DIR, "siamese_per_signer_accuracy.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] %s" % out_path)


# ── CHART 3: Score Separation Box Plot ──────────────────────────
print("Generating siamese_score_separation.png ...")

fig, ax = plt.subplots(figsize=(7, 5))

data   = [genuine_scores.tolist(), forged_scores.tolist()]
labels_box = ['Genuine Pairs', 'Forged Pairs']
colors_box = ['steelblue', 'tomato']

bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

# overlay individual points
for i, (d, c) in enumerate(zip(data, colors_box), start=1):
    jitter = np.random.uniform(-0.08, 0.08, size=len(d))
    ax.scatter([i + j for j in jitter], d, alpha=0.5, s=20, color=c, zorder=3)

ax.axhline(y=THRESHOLD, color='black', linestyle='--', lw=1.5,
           label='Decision threshold = %.2f' % THRESHOLD)
ax.set_xticks([1, 2])
ax.set_xticklabels(labels_box, fontsize=12)
ax.set_ylabel('Similarity Score', fontsize=12)
ax.set_title('Siamese CNN -- Score Separation\nGenuine vs Forged (Inference Test)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# stat annotations
ax.text(1, genuine_scores.max() + 0.02, 'avg=%.3f' % genuine_scores.mean(),
        ha='center', fontsize=9, color='steelblue', fontweight='bold')
ax.text(2, forged_scores.max() + 0.02, 'avg=%.3f' % forged_scores.mean(),
        ha='center', fontsize=9, color='tomato', fontweight='bold')

out_path = os.path.join(RESULTS_DIR, "siamese_score_separation.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] %s" % out_path)

print("\nAll charts saved to: %s" % RESULTS_DIR)
print("Files generated:")
print("  siamese_inference_evaluation.png  (confusion matrix + ROC + score dist)")
print("  siamese_per_signer_accuracy.png   (per-signer bar chart)")
print("  siamese_score_separation.png      (box plot: genuine vs forged)")
