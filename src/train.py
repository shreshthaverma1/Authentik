import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import get_dataloaders
from src.model import get_model


def train():

    # ── Load Data ─────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders()

    # ── Load Model ────────────────────────────────────────────
    model = get_model()

    # ── Loss Function ─────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR_HEAD,
        weight_decay=config.WEIGHT_DECAY
    )

    # ── Scheduler ─────────────────────────────────────────────
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS
    )

    # ── Tracking ──────────────────────────────────────────────
    best_val_acc  = 0.0
    patience_wait = 0

    # ── Training Loop ─────────────────────────────────────────
    for epoch in range(1, config.EPOCHS + 1):

        # ── Unfreeze top layers at UNFREEZE_EPOCH ─────────────
        if epoch == config.UNFREEZE_EPOCH:
            model.unfreeze_top_layers()
            optimizer = AdamW([
                {'params': model.backbone.features.parameters(),   'lr': config.LR_BACKBONE},
                {'params': model.backbone.classifier.parameters(), 'lr': config.LR_HEAD}
            ], weight_decay=config.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.EPOCHS - config.UNFREEZE_EPOCH
            )

        # ── Train Phase ───────────────────────────────────────
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for images, labels in train_loader:
            images = images.to(config.DEVICE)
            labels = labels.float().to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item()
            preds          = (torch.sigmoid(outputs) > 0.5).long()
            train_correct += (preds == labels.long()).sum().item()
            train_total   += labels.size(0)

        # ── Validation Phase ──────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images  = images.to(config.DEVICE)
                labels  = labels.float().to(config.DEVICE)

                outputs = model(images).squeeze(1)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item()
                preds        = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total   += labels.size(0)

        # ── Metrics ───────────────────────────────────────────
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        train_acc      = train_correct / train_total
        val_acc        = val_correct   / val_total

        print(f"Epoch [{epoch:02d}/{config.EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step()

        # ── Save Best Model ───────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f" Best model saved — Val Acc: {val_acc:.4f}")
            patience_wait = 0
        else:
            patience_wait += 1
            print(f"  No improvement — patience {patience_wait}/{config.PATIENCE}")

        # ── Early Stopping ────────────────────────────────────
        if patience_wait >= config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # ── Save Last Model ───────────────────────────────────────
    torch.save(model.state_dict(), config.LAST_MODEL_PATH)
    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")

    return model


if __name__ == "__main__":
    train()