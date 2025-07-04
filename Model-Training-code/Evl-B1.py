import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset
from torchvision.models import resnet18
from BackBone01 import SMRIExpertClassifier  # 👈 import your model class

# --- Load test data ---
X_test = np.load(r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/DataSet/dataset_splits_sMRI/dataset_splits/test/X_test.npy")
y_test = np.load(r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/DataSet/dataset_splits_sMRI/dataset_splits/test/y_test.npy")

# convert to tensors
X_test = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

test_dataset = TensorDataset(X_test, y_test)

# --- Load trained model ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SMRIExpertClassifier()
model.load_state_dict(torch.load(r"/home/autism/Final_Cap02/Autism Spectrum Disorder Using ABIDE/Model Weights/best_smri_expert.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Visualization Function ---
def visualize_predictions(model, test_dataset, device, num_samples=10):
    model.eval()
    indices = random.sample(range(len(test_dataset)), num_samples)
    
    cols = min(num_samples, 5)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()
    
    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            x, y_true = test_dataset[idx]
            x_input = x.unsqueeze(0).to(device)
            logits = model(x_input)
            y_pred = (torch.sigmoid(logits) > 0.5).item()
            
            img = x.cpu().squeeze()
            if img.ndim == 3:
                img = img[0]
                
            ax.imshow(img, cmap="gray")
            ax.set_title(f"True: {int(y_true.item())}\nPred: {int(y_pred)}")
            ax.axis("off")
    
    for ax in axes[num_samples:]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

# --- Run Visualization ---
if __name__ == "__main__":
    visualize_predictions(model, test_dataset, DEVICE, num_samples=10)
