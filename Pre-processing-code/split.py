
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- CONFIG ---
SLICE_DIR    = r"C:\sMRI_ABIDE_3d-to-2d\resized_slices"
LABELS_CSV   = r"C:\sMRI_ABIDE_3d-to-2d\filtered_abide_1.csv"
OUTPUT_DIR   = r"C:\sMRI_ABIDE_3d-to-2d\dataset_splits"
RANDOM_SEED  = 42
TEST_SIZE    = 0.15
VAL_SIZE     = TEST_SIZE / (1 - TEST_SIZE)

# --- Load & normalize labels CSV ---
print("Loading labels CSV...")
labels_df = pd.read_csv(LABELS_CSV)
labels_df['base_id'] = labels_df['FILE_ID'].str.replace(r'\.nii(\.gz)?$', '', regex=True)
labels_df.set_index('base_id', inplace=True)
print(f"Found {len(labels_df)} entries in labels CSV.\n")

# Pre-sort label IDs by length to match the longest prefix first
sorted_ids = sorted(labels_df.index.astype(str), key=len, reverse=True)

def load_file_paths_and_labels(slice_dir, labels_df):
    file_paths = []
    y = []
    files = [f for f in os.listdir(slice_dir) if f.lower().endswith('.npy')]
    print(f"Found {len(files)} slice files in {slice_dir}\nProcessing filenames and labels...")

    for idx, fname in enumerate(files, 1):
        name_no_ext = os.path.splitext(fname)[0]
        subject = None
        for base_id in sorted_ids:
            if name_no_ext.startswith(base_id + '_') or name_no_ext == base_id:
                subject = base_id
                break
        if subject is None:
            raise KeyError(f"No label for subject ID extracted from file '{fname}'")

        label = labels_df.at[subject, 'labels']
        file_path = os.path.join(slice_dir, fname)
        file_paths.append(file_path)
        y.append(label)

        if idx % 1000 == 0 or idx == len(files):
            print(f"  Processed {idx}/{len(files)} files")
    print("All files processed.\n")
    return file_paths, np.array(y, dtype=np.int64)

# --- Execute loading of file paths and labels ---
file_paths, y = load_file_paths_and_labels(SLICE_DIR, labels_df)
print(f"Total slices: {len(file_paths)}, Total labels: {len(y)}\n")

# --- Train/Test/Val split using file paths ---
print("Splitting into train/val/test...")
# Initial split into train and test
file_paths_train, file_paths_test, y_train, y_test = train_test_split(
    file_paths, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)
# Split train into train and val
file_paths_train, file_paths_val, y_train, y_val = train_test_split(
    file_paths_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=RANDOM_SEED
)
print(f"  Train: {len(file_paths_train)} samples")
print(f"  Val:   {len(file_paths_val)} samples")
print(f"  Test:  {len(file_paths_test)} samples\n")

# --- Function to process and save splits incrementally ---
def process_and_save_split(file_paths_split, y_split, split_name):
    print(f"\nProcessing {split_name} set...")
    X_split = []
    for idx, file_path in enumerate(file_paths_split, 1):
        arr = np.load(file_path).astype(np.float32)
        X_split.append(arr)
        if idx % 1000 == 0 or idx == len(file_paths_split):
            print(f"  Loaded {idx}/{len(file_paths_split)} slices")
    
    # Convert to array and add channel dimension
    X_split = np.array(X_split)
    X_split = X_split[..., np.newaxis]  # Add channel dimension
    
    # Save to disk
    out_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)
    x_path = os.path.join(out_dir, f"X_{split_name}.npy")
    y_path = os.path.join(out_dir, f"y_{split_name}.npy")
    np.save(x_path, X_split)
    np.save(y_path, y_split)
    print(f"Saved {split_name} set with shape {X_split.shape}\n")

# --- Process and save each split ---
process_and_save_split(file_paths_train, y_train, 'train')
process_and_save_split(file_paths_val, y_val, 'val')
process_and_save_split(file_paths_test, y_test, 'test')

print("\nAll done! Your splits are saved as .npy files in:")
print(f"  {OUTPUT_DIR}")
