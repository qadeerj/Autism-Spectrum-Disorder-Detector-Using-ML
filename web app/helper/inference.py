# inference.py
import os, uuid, torch, numpy as np
import torch.nn as nn
from torchvision.models import resnet18

# -----------------------------------------          ---------------------------------
# CONFIG – edit if your paths differ
# --------------------------------------------------------------------------
SLICE_DIR   = "data/smri_slices"          # where 15 .npy slices live
WEIGHTS_PTH = "models/best_smri_expert.pth"      # trained weights file
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SLICES  = 15
MAJORITY_N  = NUM_SLICES // 2 + 1         # 8 for 15 slices


# --------------------------------------------------------------------------
# Model definition (identical to training)
# --------------------------------------------------------------------------
class SMRIExpertClassifier(nn.Module):
    def __init__(self, embedding_dim=512, dropout_prob=0.5):
        super().__init__()
        backbone = resnet18(pretrained=False)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.embed = nn.Linear(feat_dim, embedding_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x):
        if x.shape[1] == 1:                 # gray -> 3-channel
            x = x.repeat(1, 3, 1, 1)
        f = self.features(x).flatten(1)
        e = self.embed(f)
        return self.classifier(e).squeeze(1)   # logits


# --------------------------------------------------------------------------
def load_slices(slice_dir: str) -> torch.Tensor:
    """
    Load exactly NUM_SLICES .npy slices from `slice_dir`, shape (NUM_SLICES,1,H,W)
    """
    files = sorted(f for f in os.listdir(slice_dir) if f.endswith(".npy"))
    assert len(files) == NUM_SLICES, f"Expected {NUM_SLICES} slices, got {len(files)}"
    arr  = np.stack([np.load(os.path.join(slice_dir, f)) for f in files]).astype("float32")
    return torch.from_numpy(arr)[:, None, :, :]     # add channel dim


@torch.no_grad()
def predict_scan(slice_dir: str = SLICE_DIR,
                 weights: str = WEIGHTS_PTH,
                 threshold: float = 0.5):
    """
    Returns: (label, confidence_selected, confidence_negative, confidence_positive)
       label                -> 0 (negative) or 1 (positive) by majority vote
       confidence_selected  -> fraction of votes for the chosen class (0–1)
       confidence_negative  -> fraction of slices predicted negative (0–1)
       confidence_positive  -> fraction of slices predicted positive (0–1)
    """
    # 1. Model
    model = SMRIExpertClassifier().to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()

    # 2. Data
    x = load_slices(slice_dir).to(DEVICE)           # (NUM_SLICES,1,H,W)
    logits = model(x)
    probs  = torch.sigmoid(logits)

    # Use average of probabilities as "real confidence"
    mean_prob = probs.mean().item()

    # Final label based on thresholded mean probability
    label = int(mean_prob > threshold)

    # Confidence is just the model's average certainty for the predicted class
    confidence_selected = mean_prob if label == 1 else 1 - mean_prob
    conf_pos = mean_prob
    conf_neg = 1 - mean_prob

    return label, confidence_selected


# --------------------------------------------------------------------------
# CLI test
if __name__ == "__main__":
    lbl, conf_sel, conf_neg, conf_pos = predict_scan()
    print(f"Label: {lbl}   Confidence (chosen class): {conf_sel:.2%}")
    print(f"Confidence negative: {conf_neg:.2%}\nConfidence positive: {conf_pos:.2%}")
