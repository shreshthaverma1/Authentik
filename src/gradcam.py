import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_gradcam(model, image_tensor, original_image):

    # ── Target Layer ──────────────────────────────────────────
    # Use the last conv layer inside EfficientNet features
    target_layer = [model.backbone.features[-1][0]]            # fixed target layer

    # ── Create GradCAM ────────────────────────────────────────
    cam = GradCAM(
        model=model,
        target_layers=target_layer
    )

    # ── Generate Heatmap ──────────────────────────────────────
    targets = [ClassifierOutputTarget(0)]                      # explain FAKE class

    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets
    )

    grayscale_cam = grayscale_cam[0]                           # remove batch dim

    # ── Overlay ───────────────────────────────────────────────
    visualization = show_cam_on_image(
        original_image.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    return visualization


def preprocess_for_gradcam(pil_image):
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    image_tensor = transform(pil_image)                        # apply transforms
    image_tensor = image_tensor.unsqueeze(0)                   # add batch dim
    image_tensor = image_tensor.to(config.DEVICE)             # move to device

    # ── Original image for overlay ────────────────────────────
    original = pil_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    original = np.array(original).astype(np.float32) / 255.0  # normalize to 0-1

    return image_tensor, original