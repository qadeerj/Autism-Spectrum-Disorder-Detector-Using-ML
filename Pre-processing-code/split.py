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
VAL_SIZE     = TEST_SIZE / (1 - TEST_SIZE)  # ≈ 0.176 to maintain 70/15/15 split

# --- Load and prepare labels ---
print("Loading label CSV...")
labels_df = pd.read_csv(LABELS_CSV)
labels_df['base_id'] = labels_df['FILE_ID'].str.replace(r'\.nii(\.gz)?$', '', regex=True)
labels_df.set_index('base_id', inplace=True)
print(f"Found {len(labels_df)} labeled subjects.\n")

# --- Match slice files with subject IDs ---
print("Matching slices with subject IDs...")
files = [f for f in os.listdir(SLICE_DIR) if f.lower().endswith('.npy')]
subject_to_files = {}

for fname in files:
    name_no_ext = os.path.splitext(fname)[0]
    for base_id in labels_df.index:
        if name_no_ext.startswith(base_id + '_') or name_no_ext == base_id:
            subject_to_files.setdefault(base_id, []).append(os.path.join(SLICE_DIR, fname))
            break

print(f"Matched {len(subject_to_files)} unique subjects with slice files.\n")

# --- Prepare subject-wise list and labels ---
subjects = list(subject_to_files.keys())
labels = [labels_df.at[subj, 'labels'] for subj in subjects]

# --- Subject-level train/val/test split ---
train_subs, test_subs, _, _ = train_test_split(
    subjects, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_SEED
)

train_labels = [labels_df.at[s, 'labels'] for s in train_subs]

# ✅ FIXED: correct variable order to avoid swapped val/train
train_subs, val_subs, _, _ = train_test_split(
    train_subs, train_labels, test_size=VAL_SIZE, stratify=train_labels, random_state=RANDOM_SEED
)

# --- Sanity Check: Ensure no subject leakage ---
def check_no_leakage(train_subs, val_subs, test_subs):
    train_set = set(train_subs)
    val_set = set(val_subs)
    test_set = set(test_subs)

    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("\n❌ DATA LEAK DETECTED!")
        if overlap_train_val:
            print(f"Overlap between Train and Val: {overlap_train_val}")
        if overlap_train_test:
            print(f"Overlap between Train and Test: {overlap_train_test}")
        if overlap_val_test:
            print(f"Overlap between Val and Test: {overlap_val_test}")
        raise ValueError("Subjects appear in multiple splits!")
    else:
        print("\n✅ No data leakage: All subjects are uniquely assigned to one split.")

check_no_leakage(train_subs, val_subs, test_subs)

# --- Helper: Gather slice paths and labels for subjects ---
def collect_slices(subjects_list):
    paths, ys = [], []
    for subj in subjects_list:
        label = labels_df.at[subj, 'labels']
        files = subject_to_files[subj]
        paths.extend(files)
        ys.extend([label] * len(files))
    return paths, np.array(ys, dtype=np.int64)

file_paths_train, y_train = collect_slices(train_subs)
file_paths_val, y_val     = collect_slices(val_subs)
file_paths_test, y_test   = collect_slices(test_subs)

print(f"\nSubjects per split:")
print(f"  Train subjects: {len(train_subs)}")
print(f"  Val subjects:   {len(val_subs)}")
print(f"  Test subjects:  {len(test_subs)}")

print(f"\nSlices per split:")
print(f"  Train slices:   {len(file_paths_train)}")
print(f"  Val slices:     {len(file_paths_val)}")
print(f"  Test slices:    {len(file_paths_test)}")

# --- Save split slices ---
def process_and_save_split(file_paths_split, y_split, split_name):
    print(f"\nProcessing {split_name} set...")
    X_split = [np.load(p).astype(np.float32) for p in file_paths_split]
    X_split = np.array(X_split)[..., np.newaxis]  # Add channel dim

    out_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"X_{split_name}.npy"), X_split)
    np.save(os.path.join(out_dir, f"y_{split_name}.npy"), y_split)
    print(f"✅ Saved {split_name} set: {X_split.shape}")

process_and_save_split(file_paths_train, y_train, 'train')
process_and_save_split(file_paths_val, y_val, 'val')
process_and_save_split(file_paths_test, y_test, 'test')

print("\n🎉 All done! Safe subject-wise splits with verification saved at:")
print(f"   {OUTPUT_DIR}")
