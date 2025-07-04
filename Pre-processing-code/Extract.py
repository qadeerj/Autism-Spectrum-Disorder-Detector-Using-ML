import os
import shutil
import pandas as pd
from collections import defaultdict
import re

# === Configuration: Update these to your actual paths ===
source_folder = r'C:\sMRI_ABIDE_3d-to-2d\skull_removing'  # Folder with your .nii / .nii.gz files
destination_folder = r'C:\sMRI_ABIDE_3d-to-2d\Selected_images'  # Where to move matching files
csv_path = r'C:\sMRI_ABIDE_3d-to-2d\filtered_abide_1.csv'  # Path to your CSV file

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def normalize_filename(filename):
    """Normalize filename by removing extensions and handling variations"""
    # Convert to lowercase for case-insensitive comparison
    base = filename.lower()
    
    # Remove extensions
    if base.endswith('.nii.gz'):
        base = base[:-7]
    elif base.endswith('.nii'):
        base = base[:-4]
    
    # Remove _brain suffix if present
    if base.endswith('_brain'):
        base = base[:-6]
    
    # Normalize site names
    base = re.sub(r'_+', '_', base)  # Replace multiple underscores with single
    base = re.sub(r'([a-z])([0-9])', r'\1_\2', base)  # Add underscore between letter and number
    
    return base

def get_file_variations(file_id):
    """Generate possible variations of the file name"""
    variations = set()
    
    # Original
    variations.add(file_id)
    
    # With and without _brain suffix
    variations.add(f"{file_id}_brain")
    
    # Handle site variations
    site_variations = {
        'UCLA_1': ['UCLA1', 'UCLA_1', 'UCLA-1'],
        'UCLA_2': ['UCLA2', 'UCLA_2', 'UCLA-2'],
        'UM_1': ['UM1', 'UM_1', 'UM-1'],
        'UM_2': ['UM2', 'UM_2', 'UM-2'],
        'LEUVEN_1': ['LEUVEN1', 'LEUVEN_1', 'LEUVEN-1'],
        'LEUVEN_2': ['LEUVEN2', 'LEUVEN_2', 'LEUVEN-2'],
        'CMU_a': ['CMU_A', 'CMUA', 'CMU-A'],
        'CMU_b': ['CMU_B', 'CMUB', 'CMU-B']
    }
    
    # Add variations with different cases
    variations.add(file_id.upper())
    variations.add(file_id.lower())
    variations.add(file_id.title())
    
    # Add variations without underscores
    variations.add(file_id.replace('_', ''))
    
    return variations

def main():
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Read CSV file
    print_section("Reading CSV")
    df = pd.read_csv(csv_path)
    file_ids = set(df['FILE_ID'].astype(str))
    print(f"Found {len(file_ids)} file IDs in CSV")
    
    # Get list of files in source directory
    print_section("Scanning Source Directory")
    source_files = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                source_files.append(os.path.join(root, file))
    print(f"Found {len(source_files)} .nii/.nii.gz files in source directory")
    
    # Track statistics
    moved_files = []
    missing_files = []
    site_stats = defaultdict(lambda: {'found': 0, 'missing': 0})
    
    # Process each file ID
    print_section("Processing Files")
    for file_id in file_ids:
        found = False
        variations = get_file_variations(file_id)
        
        for source_file in source_files:
            source_basename = os.path.basename(source_file)
            normalized_source = normalize_filename(source_basename)
            
            # Try all variations
            for variation in variations:
                normalized_variation = normalize_filename(variation)
                if normalized_source == normalized_variation:
                    # Found a match
                    dest_path = os.path.join(destination_folder, source_basename)
                    try:
                        shutil.copy2(source_file, dest_path)
                        moved_files.append(source_basename)
                        site = file_id.split('_')[0]
                        site_stats[site]['found'] += 1
                        print(f"Moved: {source_basename}")
                        found = True
                        break
                    except Exception as e:
                        print(f"Error moving {source_basename}: {str(e)}")
            
            if found:
                break
        
        if not found:
            missing_files.append(file_id)
            site = file_id.split('_')[0]
            site_stats[site]['missing'] += 1
    
    # Print summary
    print_section("Summary")
    print(f"Total files in CSV: {len(file_ids)}")
    print(f"Files moved: {len(moved_files)}")
    print(f"Files missing: {len(missing_files)}")
    
    print("\nMissing files by site:")
    for site, stats in sorted(site_stats.items()):
        if stats['missing'] > 0:
            total = stats['missing'] + stats['found']
            print(f"{site}: {stats['missing']} missing out of {total} total ({stats['found']} found)")
    
    print("\nDetailed missing files report:")
    missing_by_site = defaultdict(list)
    for file_id in missing_files:
        site = file_id.split('_')[0]
        missing_by_site[site].append(file_id)
    
    for site in sorted(missing_by_site.keys()):
        print(f"\n{site} missing files:")
        for file_id in sorted(missing_by_site[site]):
            print(f"- {file_id}")
            # Check for similar files
            similar_files = []
            file_number = file_id.split('_')[-1]
            for source_file in source_files:
                if site in source_file and file_number in source_file:
                    similar_files.append(os.path.basename(source_file))
            if similar_files:
                print("  Similar files found in source directory:")
                for similar in similar_files:
                    print(f"  * {similar}")

if __name__ == '__main__':
    main() 
