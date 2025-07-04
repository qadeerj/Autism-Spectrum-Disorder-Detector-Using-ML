import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# --- CONFIG ---
SMRI_DATA_DIR       = r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/DataSet/dataset_splits_sMRI/dataset_splits"
FMRI_DATA_DIR       = r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/DataSet/dataset_splits_fMRI/dataset_splits"
SMRI_WEIGHTS_PATH   = r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/Model Weights/best_smri_expert.pth"
FMRI_WEIGHTS_PATH   = r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/Model Weights/best_fmri_expert.pth"
FUSION_WEIGHTS_PATH = 'best_fusion_model.pth'

BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATA LOADING ---
def load_split(data_dir, split):
    X = np.load(os.path.join(data_dir, split, f"X_{split}.npy"))
    y = np.load(os.path.join(data_dir, split, f"y_{split}.npy"))
    X = torch.tensor(X.transpose(0,3,1,2), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

X_tr_s, y_tr_s = load_split(SMRI_DATA_DIR, 'train')
X_val_s, y_val_s = load_split(SMRI_DATA_DIR, 'val')
X_te_s, y_te_s = load_split(SMRI_DATA_DIR, 'test')
X_tr_f, y_tr_f = load_split(FMRI_DATA_DIR, 'train')
X_val_f, y_val_f = load_split(FMRI_DATA_DIR, 'val')
X_te_f, y_te_f = load_split(FMRI_DATA_DIR, 'test')

assert torch.equal(y_tr_s, y_tr_f), "Train labels mismatch!"
assert torch.equal(y_val_s, y_val_f), "Val labels mismatch!"
assert torch.equal(y_te_s, y_te_f), "Test labels mismatch!"

train_ds = TensorDataset(X_tr_s, X_tr_f, y_tr_s)
val_ds   = TensorDataset(X_val_s, X_val_f, y_val_s)
test_ds  = TensorDataset(X_te_s, X_te_f, y_te_s)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,             pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,             pin_memory=True)

# --- EXPERT MODELS ---
class SMRIExpertClassifier(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features   = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim        = backbone.fc.in_features
        self.embed      = nn.Linear(feat_dim, emb_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(emb_dim, 1)
        )
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        f = self.features(x).flatten(1)
        e = self.embed(f)
        return self.classifier(e).squeeze(1)

class FMRIExpertClassifier(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features   = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim        = backbone.fc.in_features
        self.embed      = nn.Linear(feat_dim, emb_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(emb_dim, 1)
        )
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        f = self.features(x).flatten(1)
        e = self.embed(f)
        return self.classifier(e).squeeze(1)

# load & freeze experts
smri_expert = SMRIExpertClassifier().to(DEVICE)
smri_expert.load_state_dict(torch.load(SMRI_WEIGHTS_PATH, map_location=DEVICE))
fmri_expert = FMRIExpertClassifier().to(DEVICE)
fmri_expert.load_state_dict(torch.load(FMRI_WEIGHTS_PATH, map_location=DEVICE))
for p in smri_expert.parameters(): p.requires_grad = False
for p in fmri_expert.parameters(): p.requires_grad = False
smri_expert.eval()
fmri_expert.eval()

# --- FUSION MODEL ---
class BiModalTokenTransformer(nn.Module):
    def __init__(self, emb_dim=512, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.type_emb    = nn.Parameter(torch.randn(2, emb_dim))
        encoder          = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=layers)
    def forward(self, e1, e2):
        tokens = torch.stack([e1, e2], dim=1) + self.type_emb.unsqueeze(0)
        out    = self.transformer(tokens)
        return out[:,0], out[:,1]

class DualFrozenFusionModel(nn.Module):
    def __init__(self, smri_exp, fmri_exp, emb_dim=512, gate_h=128):
        super().__init__()
        self.smri_exp   = smri_exp
        self.fmri_exp   = fmri_exp
        self.bmt        = BiModalTokenTransformer(emb_dim)
        self.gate       = nn.Sequential(
            nn.Linear(2*emb_dim, gate_h),
            nn.ReLU(inplace=True),
            nn.Linear(gate_h, 2),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(emb_dim//2, 1)
        )
    def forward(self, x_s, x_f):
        if x_s.shape[1] == 1: x_s = x_s.repeat(1, 3, 1, 1)
        if x_f.shape[1] == 1: x_f = x_f.repeat(1, 3, 1, 1)
        f1 = self.smri_exp.features(x_s.to(DEVICE)).flatten(1); e1 = self.smri_exp.embed(f1)
        f2 = self.fmri_exp.features(x_f.to(DEVICE)).flatten(1); e2 = self.fmri_exp.embed(f2)
        t1, t2 = self.bmt(e1, e2)
        g1, g2 = self.gate(torch.cat([t1, t2], dim=1)).chunk(2, dim=1)
        fused  = g1 * t1 + g2 * t2
        return self.classifier(fused).squeeze(1)

fusion_model = DualFrozenFusionModel(smri_expert, fmri_expert).to(DEVICE)

# --- TRAINING SETUP & RUN (unchanged) ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(fusion_model.parameters(), lr=LR)
best_val_acc = 0.0
train_losses, train_accs = [], []
val_losses,   val_accs   = [], []

for epoch in range(1, EPOCHS+1):
    fusion_model.train()
    run_loss, c, t = 0.0, 0, 0
    for xs, xf, y in train_loader:
        xs, xf, y = xs.to(DEVICE), xf.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = fusion_model(xs, xf)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        run_loss += loss.item()
        preds = (torch.sigmoid(out) > 0.5).float()
        c += (preds == y).sum().item(); t += y.size(0)
    train_losses.append(run_loss/len(train_loader)); train_accs.append(c/t)

    fusion_model.eval()
    v_loss, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for xs, xf, y in val_loader:
            xs, xf, y = xs.to(DEVICE), xf.to(DEVICE), y.to(DEVICE)
            out = fusion_model(xs, xf)
            v_loss += criterion(out, y).item()
            pv = (torch.sigmoid(out)>0.5).float()
            vc += (pv==y).sum().item(); vt += y.size(0)
    val_losses.append(v_loss/len(val_loader)); val_accs.append(vc/vt)

    print(f"[{epoch}/{EPOCHS}] Train L:{train_losses[-1]:.4f} A:{train_accs[-1]:.4f} | Val L:{val_losses[-1]:.4f} A:{val_accs[-1]:.4f}")
    if val_accs[-1] > best_val_acc:
        best_val_acc = val_accs[-1]
        torch.save(fusion_model.state_dict(), FUSION_WEIGHTS_PATH)
        print(f"*** Saved Best Fusion Model (Acc {best_val_acc:.4f}) ***\n")

# --- METRICS PLOTS & FINAL TEST EVAL (unchanged) ---
epochs_r = range(1, EPOCHS+1)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(epochs_r, train_losses, label='Train Loss')
plt.plot(epochs_r, val_losses,   label='Val Loss')
plt.title('Loss per Epoch'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.subplot(1,2,2)
plt.plot(epochs_r, train_accs, label='Train Acc')
plt.plot(epochs_r, val_accs,   label='Val Acc')
plt.title('Accuracy per Epoch'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.ylim(0,1); plt.legend()
plt.tight_layout(); plt.show()

fusion_model.load_state_dict(torch.load(FUSION_WEIGHTS_PATH, map_location=DEVICE))
fusion_model.eval()
test_preds, test_labs = [], []
with torch.no_grad():
    for xs, xf, y in test_loader:
        xs, xf, y = xs.to(DEVICE), xf.to(DEVICE), y.to(DEVICE)
        out = fusion_model(xs, xf)
        p = (torch.sigmoid(out) > 0.5).float()
        test_preds.extend(p.cpu().tolist()); test_labs.extend(y.cpu().tolist())

print(f"*** Final Test Acc: {accuracy_score(test_labs, test_preds):.4f} ***")
print("Test Confusion Matrix:")
print(confusion_matrix(test_labs, test_preds))

# --- RANDOM VISUALIZATION WITH BOTH CONFIDENCES ---
all_s, all_f, all_preds, all_confs, all_labels = [], [], [], [], []
with torch.no_grad():
    for xs, xf, y in test_loader:
        xs, xf, y = xs.to(DEVICE), xf.to(DEVICE), y.to(DEVICE)
        logits = fusion_model(xs, xf)
        probs  = torch.sigmoid(logits)         # P(ASD)
        preds  = (probs > 0.5).float()
        all_s.extend(xs.cpu()); all_f.extend(xf.cpu())
        all_preds.extend(preds.cpu()); all_confs.extend(probs.cpu()); all_labels.extend(y.cpu())

# compute P(not‑ASD) = 1 – P(ASD)
all_confs = torch.tensor(all_confs)
p_asd     = all_confs
p_not     = 1 - p_asd
num_vis   = 8
indices   = random.sample(range(len(all_labels)), num_vis)

plt.figure(figsize=(16, 2 * num_vis))
for i, idx in enumerate(indices):
    smri_img = all_s[idx][0].numpy()
    fmri_img = all_f[idx][0].numpy()
    act      = int(all_labels[idx].item())
    pr       = int(all_preds[idx].item())
    conf1    = p_asd[idx].item()
    conf0    = p_not[idx].item()

    plt.subplot(num_vis, 2, 2*i+1)
    plt.imshow(smri_img, cmap='gray')
    plt.title(f"sMRI\nAct:{act} Pr:{pr}\nP(not‑ASD):{conf0:.2f}  P(ASD):{conf1:.2f}")
    plt.axis('off')

    plt.subplot(num_vis, 2, 2*i+2)
    plt.imshow(fmri_img, cmap='gray')
    plt.title(f"fMRI\nAct:{act} Pr:{pr}\nP(not‑ASD):{conf0:.2f}  P(ASD):{conf1:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()
