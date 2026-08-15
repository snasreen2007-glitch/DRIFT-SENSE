"""
cnn_feature_extractor.py
=========================
Implements Document Section 3.1 (Reference Feature Extraction) and
Section 3.3 (Deep Feature Maps).

Role in the DRIFT-SENSE pipeline
---------------------------------
Both the reference wafer patch and every level of the search-image
pyramid are pushed through the SAME CNN backbone. This gives:

  * F_R   -> a single embedding vector for the reference region
  * F_S^l -> a dense feature map for each pyramid level l of the search image

These are the "F_R = f_theta(I_R)" and "F_S^l = f_theta(I_S^l)" terms
in the paper.

Why a small custom CNN instead of a huge pretrained backbone?
---------------------------------------------------------------
A full ImageNet-pretrained ResNet/VGG works too (see `use_pretrained=True`
below) and will usually give better semantic robustness. But for wafer
images (mostly geometric patterns, not natural-image content) a compact
CNN trained/initialized from scratch is:
  * much faster to run on CPU (important during dataset-scale evaluation)
  * avoids a large network download
  * easy to fine-tune later on real wafer data if you get access to it

Swap `backbone="resnet18"` for the ImageNet path any time; the rest of
the pipeline (matching, registration, subpixel, confidence) does not
change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DriftSenseCNN(nn.Module):
    """A compact fully-convolutional feature extractor.

    Output stride is 8 (i.e. a 256x256 input produces a 32x32 feature
    map), which is a good trade-off between spatial localization
    accuracy (for later subpixel refinement) and coarse-matching speed.
    """

    def __init__(self, out_channels: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, padding_mode="replicate"),   # /2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),  # /4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),  # /8
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)


class CNNFeatureExtractor:
    """Wraps DriftSenseCNN (or a pretrained torchvision backbone) with
    the pre/post-processing DRIFT-SENSE needs."""

    def __init__(self, backbone: str = "driftsense", device: str = None,
                 weights_path: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_name = backbone

        if backbone == "driftsense":
            self.model = DriftSenseCNN()
            self.out_channels = 128
            self.stride = 8
        elif backbone == "resnet18":
            import torchvision.models as models
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # keep up to layer2 -> stride 8, 128 channels
            self.model = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=1),  # gray -> 3ch for ImageNet weights
                net.conv1, net.bn1, net.relu, net.maxpool,
                net.layer1, net.layer2,
            )
            self.out_channels = 128
            self.stride = 8
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_map(self, image_gray_uint8: np.ndarray) -> torch.Tensor:
        """image -> dense feature map, shape (C, H', W')."""
        t = torch.from_numpy(image_gray_uint8).float().unsqueeze(0).unsqueeze(0) / 255.0
        t = t.to(self.device)
        fmap = self.model(t)[0]  # (C,H',W')
        return fmap

    @torch.no_grad()
    def extract_embedding(self, patch_gray_uint8: np.ndarray) -> torch.Tensor:
        """Reference patch -> single global-average-pooled embedding, shape (C,).
        This is F_R in Section 3.1."""
        fmap = self.extract_map(patch_gray_uint8)          # (C,H',W')
        emb = fmap.mean(dim=(1, 2))                         # (C,)
        emb = F.normalize(emb, dim=0)
        return emb


if __name__ == "__main__":
    # quick smoke test
    extractor = CNNFeatureExtractor(backbone="driftsense")
    dummy_ref = (np.random.rand(128, 128) * 255).astype(np.uint8)
    dummy_search = (np.random.rand(512, 512) * 255).astype(np.uint8)
    emb = extractor.extract_embedding(dummy_ref)
    fmap = extractor.extract_map(dummy_search)
    print("Reference embedding shape:", emb.shape)
    print("Search feature map shape:", fmap.shape)
