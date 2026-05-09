import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


# ─────────────────────────────────────────────────────────────
# GRAD-CAM AGENT
# Grad-CAM = Gradient-weighted Class Activation Mapping
#
# In plain English:
#   1. We run the image through the model (forward pass)
#   2. We ask: "which neurons fired the most for this prediction?"
#   3. We trace those neurons back to the last conv layer
#   4. We draw a heatmap showing which IMAGE REGIONS caused the output
#
# For signatures: it highlights which pen strokes looked suspicious
# ─────────────────────────────────────────────────────────────

class GradCAMAgent:

    def __init__(self, model, target_layer_name='features.28'):
        """
        model            : the SiameseNet (we use one tower for Grad-CAM)
        target_layer_name: which conv layer to visualize
                           'features.28' = last conv layer in VGG16
                           This is the best layer — it has the highest-level features
        """
        self.model       = model
        self.gradients   = None   # will store gradients during backward pass
        self.activations = None   # will store feature maps during forward pass

        # Register hooks on the target layer
        # Hooks are like "listeners" — they capture values during forward/backward
        target_layer = dict(model.tower.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Called automatically during forward pass — saves feature maps"""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Called automatically during backward pass — saves gradients"""
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, img_path_original):
        """
        img_tensor       : preprocessed image tensor (1, 3, 128, 256)
        img_path_original: path to the ORIGINAL image file (for overlay)

        Returns: PIL Image with colored heatmap overlaid on the signature
        """
        import torch.nn as nn

        # VGG16 uses inplace=True ReLU by default, which crashes with
        # PyTorch backward hooks. Disable inplace on ALL ReLU layers.
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        self.model.eval()
        img_tensor = img_tensor.clone().requires_grad_(True)

        # Forward pass through ONE tower of the Siamese network
        embedding = self.model.tower(img_tensor)

        # Use the norm of the embedding as the "score" to backpropagate
        score = embedding.norm()
        self.model.zero_grad()
        score.backward()   # backward pass — this triggers _save_gradient

        # ── Compute Grad-CAM ──────────────────────────────────
        # Average the gradients across spatial dimensions → importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU: only keep positive activations (negative = unimportant)
        cam = F.relu(cam)

        # Resize CAM to match original image size
        cam = F.interpolate(cam, size=(128, 256), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0-1 range
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # ── Overlay heatmap on original image ─────────────────
        original = np.array(Image.open(img_path_original).resize((256, 128)))

        # If grayscale image, convert to RGB for overlay
        if original.ndim == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

        # Create colored heatmap (blue=low attention, red=high attention)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend original image (60%) with heatmap (40%)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        return Image.fromarray(overlay)
