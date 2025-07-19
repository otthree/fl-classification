# ADNI MRI Preprocessing Documentation

This document provides comprehensive documentation for the ADNI MRI preprocessing pipeline implemented in `scripts/preprocess_mri.py`.

## Overview

The preprocessing script automates the standard preprocessing pipeline for ADNI MRI images:

1. **Resampling** to 1mm isotropic spacing
2. **Registration** to a standard template (ICBM152)
3. **Skull stripping** using FSL BET

The script includes advanced features such as resume functionality, directory filtering, progress tracking, and robust error handling.

## Requirements

### Software Dependencies

- **Python 3.10+**
- **ANTs (Advanced Normalization Tools)**
  - `ResampleImageBySpacing`
  - `antsRegistrationSyN.sh`
- **FSL (FMRIB Software Library)**
  - `bet` (Brain Extraction Tool)

### Python Dependencies

The script uses standard Python libraries:
- `argparse`, `glob`, `logging`, `os`, `subprocess`, `sys`
- `tqdm` for progress bars
- `typing` for type hints

## Usage

### Basic Usage

```bash
python scripts/preprocess_mri.py --input input_folder [OPTIONS]
```

### Command Line Arguments

#### Required Arguments

- `--input` (required): Path to input directory containing MRI images (.nii or .nii.gz)

#### Optional Arguments

- `--output`: Base output directory (default: same folder name in parent directory)
- `--template`: Template for registration (default: `data/ICBM152/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii`)
- `--no-progress`: Disable progress bars
- `--include-dirs-regex`: Comma-separated list of regex patterns to include specific first-level subdirectories

### Examples

#### Basic Preprocessing
```bash
python scripts/preprocess_mri.py --input /path/to/adni_images
```

#### Custom Output Directory
```bash
python scripts/preprocess_mri.py --input /path/to/adni_images --output /path/to/processed_images
```

#### Custom Template
```bash
python scripts/preprocess_mri.py --input /path/to/adni_images --template /path/to/custom_template.nii.gz
```

#### Directory Filtering with Regex
```bash
# Process only directories starting with "34" or "94"
python scripts/preprocess_mri.py --input /path/to/adni_images --include-dirs-regex "34.*,94.*"

# Process only specific subject IDs
python scripts/preprocess_mri.py --input /path/to/adni_images --include-dirs-regex "002_S_0295,002_S_0413,002_S_0559"
```

#### Disable Progress Bars (for logging/scripting)
```bash
python scripts/preprocess_mri.py --input /path/to/adni_images --no-progress
```

## Input Data Structure

The script expects ADNI MRI images organized in the following structure:

```
<input_directory>/
├── <subject_id_1>/
│   └── <session_info>/
│       └── ADNI_<subject_id>_<metadata>_I<image_id>.nii.gz
├── <subject_id_2>/
│   └── <session_info>/
│       └── ADNI_<subject_id>_<metadata>_I<image_id>.nii.gz
└── ...
```

**Supported file formats:**
- `.nii` (uncompressed NIFTI)
- `.nii.gz` (compressed NIFTI)

## Output Structure

For each input directory, the script creates three output directories with processed results:

```
<parent_directory>/
├── <input_dirname>_step1_resampling/
│   └── [same structure as input with resampled images]
├── <input_dirname>_step2_registration/
│   └── [same structure as input with registered images]
└── <input_dirname>_step3_skull_stripping/
    └── [same structure as input with final processed images]
```

### Example Output Structure

**Input:**
```
/data/raw_images/
└── subject001/
    └── session01/
        └── scan.nii.gz
```

**Output:**
```
/data/raw_images_step1_resampling/
└── subject001/
    └── session01/
        └── scan.nii

/data/raw_images_step2_registration/
└── subject001/
    └── session01/
        └── scan.nii.gz

/data/raw_images_step3_skull_stripping/
└── subject001/
    └── session01/
        └── scan.nii.gz
```

## Processing Pipeline Details

### Step 1: Resampling
- **Tool:** `ResampleImageBySpacing` (ANTs)
- **Target spacing:** 1mm × 1mm × 1mm isotropic
- **Purpose:** Standardize voxel spacing across all images

### Step 2: Registration
- **Tool:** `antsRegistrationSyN.sh` (ANTs)
- **Target template:** ICBM152 MNI template
- **Configuration:**
  - 28 threads for parallel processing
  - Random seed 42 for reproducibility
  - SyN (Symmetric Normalization) algorithm
- **Purpose:** Align images to standard anatomical space

### Step 3: Skull Stripping
- **Tool:** `bet` (FSL)
- **Purpose:** Remove non-brain tissue and extract brain only

## Advanced Features

### Resume Functionality
- **Automatic detection** of previously processed files
- **Skip processed files** to avoid redundant computation
- **Reprocess last file** to ensure integrity (handles potential corruption)
- **Progress tracking** with detailed logging

### Directory Filtering
- **Regex-based filtering** of first-level subdirectories
- **Multiple patterns** supported with comma separation
- **Full match** requirement (uses `re.fullmatch()`)
- **Debug logging** for pattern matching

### Error Handling
- **Dependency checking** before processing starts
- **Custom exception handling** with `PreprocessingError`
- **Detailed logging** for troubleshooting
- **Graceful failure** with informative error messages

### Performance Features
- **Progress bars** with `tqdm` (can be disabled)
- **Parallel processing** in ANTs registration (28 threads)
- **Temporary file management** with automatic cleanup
- **Batch processing** with continue-on-error behavior

## Configuration Details

### Default Template
```
data/ICBM152/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii
```

### ANTs Registration Parameters
- **Dimension:** 3D
- **Algorithm:** SyN (Symmetric Normalization)
- **Threads:** 28
- **Random seed:** 42
- **Output:** Warped image in template space

### FSL BET Parameters
- **Default settings** (optimized for T1-weighted MRI)
- **Automatic output** in compressed NIFTI format (.nii.gz)

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```
   Error: Missing required tools: ResampleImageBySpacing (part of ANTs)
   ```
   **Solution:** Install ANTs and ensure it's in your PATH

2. **Template Not Found**
   ```
   Error: Template file not found
   ```
   **Solution:** Provide correct template path or ensure default template exists

3. **No NIFTI Files Found**
   ```
   Error: No NIFTI files (.nii or .nii.gz) found in /input/directory
   ```
   **Solution:** Check input directory structure and file extensions

4. **Regex Pattern No Match**
   ```
   Warning: No directories matched the provided regex patterns
   ```
   **Solution:** Verify regex patterns and directory names

### Logging and Debugging

The script provides comprehensive logging:
- **INFO level:** General progress and status updates
- **DEBUG level:** Detailed pattern matching and file operations
- **ERROR level:** Failure conditions and error details

To enable debug logging, modify the script or redirect output to files for analysis.

### Recovery from Interruption

The script automatically handles interruptions:
1. **Detects completed files** and skips them
2. **Identifies last processed file** and reprocesses it
3. **Continues from interruption point**
4. **Cleans up temporary files** on completion

## Performance Considerations

### Processing Time
- **Resampling:** ~30-60 seconds per image
- **Registration:** ~5-15 minutes per image (depends on image size and complexity)
- **Skull stripping:** ~30-60 seconds per image
- **Total:** ~6-16 minutes per image

### Resource Requirements
- **CPU:** Multi-core recommended (ANTs uses 28 threads by default)
- **Memory:** 4-8 GB RAM recommended
- **Storage:** ~3x input data size for all processing steps
- **Temporary space:** ~2x input data size during processing

### Optimization Tips
1. **Use SSD storage** for faster I/O operations
2. **Ensure sufficient RAM** to avoid swapping
3. **Monitor CPU usage** during registration steps
4. **Use regex filtering** to process subsets for testing

## Integration with ADNI Classification Pipeline

The preprocessed images are ready for use with the ADNI classification models:

1. **Update CSV labels** to point to step 3 output directory
2. **Configure image directory** in training configuration
3. **Ensure consistent naming** between CSV and image files
4. **Use appropriate data loaders** for preprocessed data

The final skull-stripped images in step 3 are the recommended input for the classification models.
