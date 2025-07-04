#---------------------------------Making DIR------------------------------------------------
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from tqdm import tqdm

# Directories
input_dir = r"C:\sMRI_ABIDE_3d-to-2d\Selected_images"  # Your skull-stripped NIfTI files
output_dir = r"C:\sMRI_ABIDE_3d-to-2d"  # Root output directory

# Create subdirectories for each step
os.makedirs(os.path.join(output_dir, "intensity_normalized"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "smoothed"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "slices"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "selected_slices"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "cropped_slices"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "resized_slices"), exist_ok=True)


#-------------------------------NORMLIZATION---------------------------------

def normalize_intensity(img_data):
    return img_data / np.max(img_data)

for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(".nii"):
        img = nib.load(os.path.join(input_dir, filename))
        img_data = img.get_fdata()
        
        # Normalize intensity to [0, 1]
        normalized_data = normalize_intensity(img_data)
        
        # Save normalized image
        normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
        normalized_img.to_filename(os.path.join(output_dir, "intensity_normalized", filename))


# #------------------------------------GAUSSION SMOOTHING---------------------------------

def apply_smoothing(img_data):
    return gaussian_filter(img_data, sigma=0.8)

for filename in tqdm(os.listdir(os.path.join(output_dir, "intensity_normalized"))):
    if filename.endswith(".nii"):
        img = nib.load(os.path.join(output_dir, "intensity_normalized", filename))
        img_data = img.get_fdata()
        
        # Apply smoothing
        smoothed_data = apply_smoothing(img_data)
        
        # Save smoothed image
        smoothed_img = nib.Nifti1Image(smoothed_data, img.affine, img.header)
        smoothed_img.to_filename(os.path.join(output_dir, "smoothed", filename))

# #----------------------------3D-to-2D--------------------------------------------#
def extract_slices(img_data):
    slices = []
    # Sagittal (X-axis)
    for i in range(img_data.shape[0]):
        slices.append(("sagittal", i, img_data[i, :, :]))
    # Coronal (Y-axis)
    for j in range(img_data.shape[1]):
        slices.append(("coronal", j, img_data[:, j, :]))
    # Transverse (Z-axis)
    for k in range(img_data.shape[2]):
        slices.append(("transverse", k, img_data[:, :, k]))
    return slices

for filename in tqdm(os.listdir(os.path.join(output_dir, "smoothed"))):
    if filename.endswith(".nii"):
        img = nib.load(os.path.join(output_dir, "smoothed", filename))
        img_data = img.get_fdata()
        
        # Extract slices
        slices = extract_slices(img_data)
        
        # Save slices as PNGs
        subject_id = filename.split(".")[0]
        for plane, idx, slice_data in slices:
            slice_path = os.path.join(output_dir, "slices", f"{subject_id}_{plane}_{idx}.npy")
            np.save(slice_path, slice_data)  # Save as numpy array for efficiency


#-------------------------------Top 16% Frames Selction------------------------------

# import os
# import re
# import numpy as np
# import shutil
# from tqdm import tqdm

# def calculate_informativeness(slice):
#     N0 = np.sum(slice == 0)  # Paper's formula: I = 1 - (N0 / total_pixels)
#     return 1 - (N0 / slice.size)

# def select_frames(slice_files, top_percent=0.16):
#     # Group by view (sagittal/coronal/transverse)
#     views = {}
#     for file in slice_files:
#         view = re.match(r".*_(sagittal|coronal|transverse)_\d+\.npy", file).group(1)
#         views.setdefault(view, []).append(file)
    
#     selected = []
#     for view, files in views.items():
#         scores = [calculate_informativeness(np.load(f)) for f in files]
#         sorted_files = [f for _, f in sorted(zip(scores, files), reverse=True)]
#         n_selected = round(len(sorted_files) * top_percent)
#         selected.extend(sorted_files[:n_selected])
    
#     return selected

# # Process subjects
# slice_dir = os.path.join(output_dir, "slices")
# all_files = os.listdir(slice_dir)
# subjects = set([re.match(r"^(.*?)_(sagittal|coronal|transverse)_\d+\.npy", f).group(1) 
#                for f in all_files if "_" in f])

# for subject in tqdm(subjects):
#     # Get all slices for this subject
#     slice_files = [f for f in all_files if f.startswith(f"{subject}_")]
#     slice_paths = [os.path.join(slice_dir, f) for f in slice_files]
    
#     # Select top 16% per view
#     selected_files = select_frames(slice_paths)
    
#     # Move selected slices
#     for src in selected_files:
#         dest = os.path.join(output_dir, "selected_slices", os.path.basename(src))
#         shutil.move(src, dest)

import os
import re
import numpy as np
import shutil
from tqdm import tqdm

def calculate_informativeness(slice_array):
    """I = 1 - (N0 / total_pixels)"""
    N0 = np.sum(slice_array == 0)
    return 1 - (N0 / slice_array.size)

def select_top_percent_per_view(slice_files, top_percent=0.16):
    """
    For each view (sagittal/coronal/transverse), pick the top X% most informative slices.
    """
    views = {}
    for fpath in slice_files:
        fname = os.path.basename(fpath)
        view = re.match(r".*_(sagittal|coronal|transverse)_\d+\.npy", fname).group(1)
        views.setdefault(view, []).append(fpath)
    
    selected = []
    for view, paths in views.items():
        # compute scores for this view
        scored = [(calculate_informativeness(np.load(p)), p) for p in paths]
        # sort descending and take top X%
        n_pick = max(1, round(len(scored) * top_percent))
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:n_pick]
        selected.extend([p for _, p in top])
    
    return selected

# --- parameters ---
TOP_PERCENT = 0.16
GLOBAL_CAP   = 15
# ------------------

slice_dir = os.path.join(output_dir, "slices")
all_files = os.listdir(slice_dir)

subjects = {
    re.match(r"^(.*?)_(sagittal|coronal|transverse)_\d+\.npy", f).group(1)
    for f in all_files if "_" in f
}

for subject in tqdm(subjects, desc="Subjects"):
    # full paths for this subject
    slice_paths = [
        os.path.join(slice_dir, f)
        for f in all_files
        if f.startswith(f"{subject}_")
    ]
    
    # 1) select top 16% per view
    selected = select_top_percent_per_view(slice_paths, top_percent=TOP_PERCENT)
    
    # 2) if more than 40, re-rank those and keep only the best 40
    if len(selected) > GLOBAL_CAP:
        reranked = [
            (calculate_informativeness(np.load(p)), p)
            for p in selected
        ]
        selected = [
            p for _, p in sorted(reranked, key=lambda x: x[0], reverse=True)
        ][:GLOBAL_CAP]
    
    # 3) move them
    dest_dir = os.path.join(output_dir, "selected_slices")
    os.makedirs(dest_dir, exist_ok=True)
    for src in selected:
        shutil.move(src, os.path.join(dest_dir, os.path.basename(src)))



#--------------------------------Resize to 145x145 --------------------------------------#

def crop_slice(slice):
    non_zero = np.argwhere(slice > 0)
    if non_zero.size == 0:
        return slice
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)
    return slice[y_min:y_max+1, x_min:x_max+1]

for filename in tqdm(os.listdir(os.path.join(output_dir, "selected_slices"))):
    slice = np.load(os.path.join(output_dir, "selected_slices", filename))
    
    # Crop
    cropped = crop_slice(slice)
    
    # Resize to 145x145 with bicubic interpolation
    resized = resize(cropped, (145, 145), order=3)
    
    # Save final slice
    np.save(os.path.join(output_dir, "resized_slices", filename), resized)
