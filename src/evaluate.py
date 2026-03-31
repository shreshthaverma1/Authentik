import torch
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import get_dataloaders
from src.model import get_model

def evaluate():

    #load data ──────────────────
    _, _, test_loader = get_dataloaders()

    model = get_model()
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location = config.DEVICE )
    )
    
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs>0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

# ── Calculate Metrics ─────

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc     = roc_auc_score(all_labels, all_probs)             # AUC score
    report  = classification_report(                           # full report
            all_labels, all_preds,
            target_names=["REAL", "FAKE"]
        )
    
    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score      : {f1:.4f}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"{'='*50}")
    print(f"\nClassification Report:\n{report}")

# ── Confusion Matrix ────────

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot = True,
        fmt = 'd',
        cmap = 'Blues',
        xticklabels = ['REAL', 'FAKE'],
        yticklabels = ['REAL', 'FAKE']
    )

    plt.title("Confusion Matrix — SynthGuard")                 # title
    plt.ylabel("True Label")                                   # y axis
    plt.xlabel("Predicted Label")                              # x axis
    plt.tight_layout()                                         # clean layout
    plt.savefig(                                               # save to outputs/plots/
        os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
    )
    plt.show()
    print(f"Confusion matrix saved to {config.PLOTS_DIR}")


if __name__ == "__main__":
    evaluate()
