
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt



# --- CONFIG ---
SMRI_DATA_DIR     = r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/DataSet/dataset_splits_sMRI/dataset_splits"
FMRI_DATA_DIR     = r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/DataSet/dataset_splits_fMRI/dataset_splits"
SMRI_WEIGHTS_PATH = r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/Model Weights/best_smri_expert.pth"
FMRI_WEIGHTS_PATH = r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/Model Weights/best_fmri_expert.pth"


BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utility: Load data splits ---
def load_split(data_dir: str, split: str):
    """
    Load features X and labels y for a given split ('train', 'val', 'test').
    Returns:
        X: Tensor of shape (N, C, H, W)
        y: Tensor of shape (N,)
    """
    x_path = os.path.join(data_dir, split, f"X_{split}.npy")
    y_path = os.path.join(data_dir, split, f"y_{split}.npy")

    X = np.load(x_path)            # (N, H, W, C)
    y = np.load(y_path)            # (N,)

    X = torch.from_numpy(X).permute(0, 3, 1, 2).float()
    y = torch.from_numpy(y).float()
    return X, y

# Load splits
X_train_s, y_train_s = load_split(SMRI_DATA_DIR, "train")
X_val_s,   y_val_s   = load_split(SMRI_DATA_DIR, "val")
X_test_s,  y_test_s  = load_split(SMRI_DATA_DIR, "test")

X_train_f, y_train_f = load_split(FMRI_DATA_DIR, "train")
X_val_f,   y_val_f   = load_split(FMRI_DATA_DIR, "val")
X_test_f,  y_test_f  = load_split(FMRI_DATA_DIR, "test")

# Ensure labels align
assert torch.equal(y_train_s, y_train_f), "Train labels do not match"
assert torch.equal(y_val_s,   y_val_f),   "Validation labels do not match"
assert torch.equal(y_test_s,  y_test_f),  "Test labels do not match"

# Create paired datasets
train_ds = TensorDataset(X_train_s, X_train_f, y_train_s)
val_ds   = TensorDataset(X_val_s,   X_val_f,   y_val_s)
test_ds  = TensorDataset(X_test_s,  X_test_f,  y_test_s)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# --- Expert Models ---
class ExpertClassifier(nn.Module):
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.embed = nn.Linear(feat_dim, emb_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        f = self.features(x).view(x.size(0), -1)
        e = self.embed(f)
        return self.classifier(e).squeeze(1)

# Load and freeze experts
smri_expert = ExpertClassifier().to(DEVICE)
fmri_expert = ExpertClassifier().to(DEVICE)
smri_expert.load_state_dict(torch.load(SMRI_WEIGHTS, map_location=DEVICE))
fmri_expert.load_state_dict(torch.load(FMRI_WEIGHTS, map_location=DEVICE))
for p in smri_expert.parameters(): p.requires_grad = False
for p in fmri_expert.parameters(): p.requires_grad = False

# --- Full Bi-Modal Transformer ---
class FullBiModalTransformer(nn.Module):
    def __init__(self, emb_dim: int = 512, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.type_emb = nn.Parameter(torch.randn(2, emb_dim))
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(emb_dim, num_heads,
                                                    dropout=dropout,
                                                    batch_first=True),
                'cross_attn': nn.MultiheadAttention(emb_dim, num_heads,
                                                     dropout=dropout,
                                                     batch_first=True),
                'norm1': nn.LayerNorm(emb_dim),
                'norm2': nn.LayerNorm(emb_dim)
            }))

    def forward(self, e1: torch.Tensor, e2: torch.Tensor):
        tokens = torch.stack([e1, e2], dim=1) + self.type_emb.unsqueeze(0)
        for layer in self.layers:
            sa_out, _ = layer['self_attn'](tokens, tokens, tokens)
            tokens = layer['norm1'](tokens + sa_out)
            t0, t1 = tokens[:, :1, :], tokens[:, 1:, :]
            ca0, _ = layer['cross_attn'](t0, t1, t1)
            ca1, _ = layer['cross_attn'](t1, t0, t0)
            tokens = layer['norm2'](tokens + torch.cat([ca0, ca1], dim=1))
        return tokens[:, 0, :], tokens[:, 1, :]

# --- Fusion Model ---
class DualFrozenFusionModel(nn.Module):
    def __init__(self, emb_dim: int = 512, gate_h: int = 128):
        super().__init__()
        self.smri_exp = smri_expert
        self.fmri_exp = fmri_expert
        self.bmt      = FullBiModalTransformer(emb_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * emb_dim, gate_h),
            nn.ReLU(inplace=True),
            nn.Linear(gate_h, 2),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(emb_dim // 2, 1)
        )

    def forward(self, x_s: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
        if x_s.size(1) == 1: x_s = x_s.repeat(1, 3, 1, 1)
        if x_f.size(1) == 1: x_f = x_f.repeat(1, 3, 1, 1)
        f1 = self.smri_exp.features(x_s.to(DEVICE)).view(x_s.size(0), -1)
        e1 = self.smri_exp.embed(f1)
        f2 = self.fmri_exp.features(x_f.to(DEVICE)).view(x_f.size(0), -1)
        e2 = self.fmri_exp.embed(f2)
        t1, t2 = self.bmt(e1, e2)
        combined = torch.cat([t1, t2], dim=1)
        g1, g2 = self.gate(combined).chunk(2, dim=1)
        fused = g1 * t1 + g2 * t2
        return self.classifier(fused).squeeze(1)

# --- Training & Evaluation ---
model     = DualFrozenFusionModel().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
best_val  = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xs, xf, y in train_loader:
        xs, xf, y = xs.to(DEVICE), xf.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xs, xf)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Training Loss = {avg_loss:.4f}")

    # Validation with confusion matrix
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xs, xf, y in val_loader:
            xs, xf, y = xs.to(DEVICE), xf.to(DEVICE), y.to(DEVICE)
            out = torch.sigmoid(model(xs, xf))
            preds.extend((out > 0.5).cpu().tolist())
            truths.extend(y.cpu().tolist())
    val_acc = accuracy_score(truths, preds)
    val_conf = confusion_matrix(truths, preds)
    print(f"Epoch {epoch}: Validation Acc = {val_acc:.4f}")
    print(f"Validation Confusion Matrix:\n{val_conf}\n")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_fusion_model.pth")
        print("Saved new best model")

# Final Test Evaluation with confusion matrix
model.load_state_dict(torch.load("best_fusion_model.pth", map_location=DEVICE))
model.eval()
all_preds, all_truths = [], []
with torch.no_grad():
    for xs, xf, y in test_loader:
        xs, xf, y = xs.to(DEVICE), xf.to(DEVICE), y.to(DEVICE)
        out = torch.sigmoid(model(xs, xf))
        all_preds.extend((out > 0.5).cpu().tolist())
        all_truths.extend(y.cpu().tolist())

test_acc = accuracy_score(all_truths, all_preds)
test_conf = confusion_matrix(all_truths, all_preds)
print(f"Final Test Accuracy = {test_acc:.4f}")
print(f"Test Confusion Matrix:\n{test_conf}")
