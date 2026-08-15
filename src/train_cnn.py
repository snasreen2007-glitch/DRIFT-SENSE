"""
train_cnn.py
==============
Trains DriftSenseCNN so the CNN branch actually contributes to
matching, instead of running on random weights.

Approach: triplet loss.
  anchor   = embedding of the reference patch
  positive = embedding of the TRUE location patch, cropped from the
             search image at the ground-truth coordinate
  negative = embedding of a RANDOM wrong-location patch from the same
             search image

Loss pulls anchor/positive together and pushes anchor/negative apart.
This only needs (reference, search, ground_truth) triples -- exactly
what dataset_generator.py already produces. No manual labeling needed.

Usage:
    # 1. generate a LARGE training set (hundreds-thousands of samples;
    #    30 is only enough for evaluation, not training)
    python src/dataset_generator.py --n 800 --out data_train --seed 1

    # 2. train
    python src/train_cnn.py --data_dir data_train --epochs 15 --out weights/driftsense_cnn.pt

    # 3. point the detector at the trained weights (see drift_sense_pipeline.py
    #    or pass weights_path= when constructing CNNFeatureExtractor)
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from cnn_feature_extractor import DriftSenseCNN


class TripletWaferDataset(Dataset):
    def __init__(self, data_dir: str, patch_size: int = 64):
        with open(os.path.join(data_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.manifest)

    def _crop(self, img, cx, cy):
        h, w = img.shape
        half = self.patch_size // 2
        cx = int(np.clip(cx, half, w - half))
        cy = int(np.clip(cy, half, h - half))
        return img[cy - half:cy + half, cx - half:cx + half]

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        ref = cv2.imread(entry["reference"], cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(entry["search"], cv2.IMREAD_GRAYSCALE)
        with open(entry["ground_truth_file"]) as f:
            gt = json.load(f)["ground_truth"]

        positive = self._crop(search, gt["x"], gt["y"])

        h, w = search.shape
        half = self.patch_size // 2
        for _ in range(10):
            nx = random.randint(half, w - half)
            ny = random.randint(half, h - half)
            if np.hypot(nx - gt["x"], ny - gt["y"]) > 3 * self.patch_size:
                break
        negative = self._crop(search, nx, ny)

        def to_tensor(img):
            if img.shape != (self.patch_size, self.patch_size):
                img = cv2.resize(img, (self.patch_size, self.patch_size))
            return torch.from_numpy(img).float().unsqueeze(0) / 255.0

        return to_tensor(ref), to_tensor(positive), to_tensor(negative)


def train(data_dir: str, epochs: int, batch_size: int, lr: float, out_path: str,
          margin: float = 0.3, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TripletWaferDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DriftSenseCNN().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def embed(batch):
        fmap = model(batch)              # (B,C,H',W')
        emb = fmap.mean(dim=(2, 3))       # global average pool -> (B,C)
        return F.normalize(emb, dim=1)

    print(f"Training on {len(dataset)} triplets, device={device}")
    for epoch in range(1, epochs + 1):
        total_loss, n_batches = 0.0, 0
        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            emb_a = embed(anchor)
            emb_p = embed(positive)
            emb_n = embed(negative)

            d_pos = (emb_a - emb_p).pow(2).sum(dim=1)
            d_neg = (emb_a - emb_n).pow(2).sum(dim=1)
            loss = F.relu(d_pos - d_neg + margin).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch:03d}/{epochs}  avg_triplet_loss={avg_loss:.4f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved trained weights to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_train")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="weights/driftsense_cnn.pt")
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.out)
