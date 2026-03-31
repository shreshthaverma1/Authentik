import torch                                                    # core PyTorch library
import torch.nn as nn                                          # neural network building blocks
import torchvision.models as models                            # pretrained models including EfficientNet
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add root to path
import config                                                  # our hyperparameters


class SynthGuard(nn.Module):                                   # our model class inherits from nn.Module
    def __init__(self):
        super(SynthGuard, self).__init__()                     # fixed: self() → self

        # ── Load Pretrained Backbone ──────────────────────────
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT     # use ImageNet pretrained weights
        )

        # ── Replace Classifier Head ───────────────────────────
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT),                      # randomly zero 30% of neurons
            nn.Linear(1280, config.NUM_CLASSES)                # 1280 features → 1 output score
        )

    def forward(self, x):                                      # defines how data flows through model
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.parameters():               # loop through every parameter
            param.requires_grad = False                        # freeze — no gradient updates
        for param in self.backbone.classifier.parameters():   # unfreeze our head specifically
            param.requires_grad = True                         # head must always train
        print("Backbone frozen — training head only")

    def unfreeze_top_layers(self):
        for name, param in self.backbone.named_parameters():
            if "features.7" in name or "features.8" in name:  # last two blocks
                param.requires_grad = True                     # allow gradient updates
        print("Top layers unfrozen — fine-tuning backbone")

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True                         # full fine-tuning
        print("All layers unfrozen")


def get_model():                                               # fixed: moved OUTSIDE the class
    model = SynthGuard()                                       # create model instance
    model = model.to(config.DEVICE)                           # move to GPU/CPU
    model.freeze_backbone()                                    # start with frozen backbone

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total      = sum(p.numel() for p in model.parameters())
    print(f"Trainable params : {trainable:,}")
    print(f"Total params     : {total:,}")

    return model