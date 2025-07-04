import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_DIR = r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/DataSet/dataset_splits_fMRI/dataset_splits"
BATCH_SIZE = 64
EPOCHS = 50
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- sMRI Model Definition ---
class FMRIExpertClassifier(nn.Module):
    """
    ResNet-18 backbone + embedding head + classifier head for 2D fMRI slices.
    """
    def __init__(self, embedding_dim=512, num_classes=1, dropout_prob=0.5):
        super().__init__()
        # Load pretrained ResNet18 and strip off its final FC
        backbone = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        
        # Embedding projection
        self.embed = nn.Linear(feat_dim, embedding_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        # Convert 1-channel → 3-channel if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Feature extraction
        f = self.features(x).flatten(1)        # (B, feat_dim)
        e = self.embed(f)                      # (B, embedding_dim)
        logits = self.classifier(e).squeeze(1) # (B,)
        return logits

# --- Data Helper for sMRI ---
def load_split(name: str):
    X_path = os.path.join(DATA_DIR, name, f"X_{name}.npy")
    y_path = os.path.join(DATA_DIR, name, f"y_{name}.npy")
    X = np.load(X_path)  # shape (N, H, W, C)
    y = np.load(y_path)  # shape (N,)
    # to torch: (N, C, H, W)
    X = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

# --- Data Diagnostics (optional) ---
def check_data_sanity(X, y, name):
    print(f"\n{name} Set Diagnostics:")
    print(f"Shapes: X={X.shape}, y={y.shape}")
    print(f"Class balance: {y.mean().item():.4f}")
    print(f"Data range: X ∈ [{X.min():.2f}, {X.max():.2f}]")
    print(f"NaN values: X={torch.isnan(X).any()}, y={torch.isnan(y).any()}")
    # Plot sample images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for idx, ax in enumerate(axes.flat):
        img = X[idx][0].cpu().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {y[idx].item()}")
        ax.axis('off')
    plt.suptitle(f"{name} Set Samples")
    plt.show()

# --- Training Setup ---
def initialize_model(train_loader):
    model = FMRIExpertClassifier().to(DEVICE)
    # Class weighting for imbalance
    y_train = train_loader.dataset.tensors[1]
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=3, verbose=True
    )
    return model, criterion, optimizer, scheduler

# --- Training Loop ---
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs):
    best_acc, patience = 0.0, 70
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        val_acc, conf_matrix = evaluate_model(model, val_loader)
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        
        if val_acc > best_acc:
            best_acc, patience = val_acc, 70
            torch.save(model.state_dict(), 'best_fmri_expert.pth')
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break
    
    model.load_state_dict(torch.load('best_fmri_expert.pth'))
    return model, history

# --- Evaluation ---
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return acc, conf_matrix

# --- Main Execution ---
if __name__ == "__main__":
    # Load data splits
    X_train, y_train = load_split('train')
    X_val, y_val     = load_split('val')
    X_test, y_test   = load_split('test')
    
    # (Optional) sanity checks
    check_data_sanity(X_train, y_train, 'Train')
    check_data_sanity(X_val, y_val,     'Validation')
    check_data_sanity(X_test, y_test,   'Test')
    
    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE, pin_memory=True
    )
    
    # Initialize model, loss, optimizer, scheduler
    model, criterion, optimizer, scheduler = initialize_model(train_loader)
    print("\nFMRIExpertClassifier Architecture:\n", model)
    
    # Train
    trained_model, history = train_model(
        model, criterion, optimizer, scheduler,
        train_loader, val_loader, EPOCHS
    )
    
    # Test
    test_acc, test_conf = evaluate_model(
        trained_model,
        DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
    )
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Test Confusion Matrix:\n{test_conf}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title("Accuracy Progress")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
