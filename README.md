# ADNI1: Centralized 3-Way Alzheimer's Disease Classification on 3D MRI Data

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-blue.svg?style=flat-square)](https://pytorch.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=flat-square)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-purple.svg?style=flat-square)](https://monai.io/)

</div>

Centralized deep learning pipeline for **CN vs MCI vs AD** 3-way classification using 3D MRI scans from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset.

## Setup

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate

# Install required dependencies
uv pip install -e .
```

## Usage

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Testing

```bash
python scripts/test.py --config configs/default.yaml --checkpoint outputs/<run_name>/checkpoints/best_model.pth
```

### Repository Structure

```
adni1/
├── adni_classification/       # Core classification components
│   ├── models/                # Model implementations (ResNet3D, DenseNet3D, SimpleCNN, RosannaCNN, SecureFedCNN)
│   ├── datasets/              # Dataset implementations
│   │   ├── adni_dataset.py            # NIfTI dataset (normal, cache, smartcache, persistent)
│   │   └── tensor_folder_dataset.py   # Pre-processed .pt tensor dataset
│   ├── utils/                 # Utility functions (training, losses, visualization)
│   └── config/                # Configuration management
│       └── config.py          # Main configuration
├── scripts/                   # Training and utility scripts
│   ├── train.py               # Main training script
│   ├── test.py                # Model evaluation script
│   ├── split_by_patient.py    # Patient-wise train/val split
│   └── preprocess_mri.py      # MRI preprocessing pipeline
├── configs/                   # Configuration YAML files
│   ├── tensor_resnet18.yaml   # Config for .pt tensor training
│   └── ...
├── pyproject.toml             # Project dependencies (uv)
└── README.md
```

## Configuration

The project uses YAML-based configuration with structured dataclasses in `adni_classification/config/`.

### Configuration Structure

#### 1. Data Configuration (`data:`)
- **Dataset paths**: `train_csv_path`, `val_csv_path`, `img_dir`, `tensor_dir`
- **Dataset types**: `normal`, `cache`, `smartcache`, `persistent`, `tensor_folder`
- **Image preprocessing**: `resize_size`, `resize_mode`, `spacing_size`
- **Classification modes**: `CN_MCI_AD` (3-class) or `CN_AD` (2-class)
- **Caching options**: `cache_rate`, `cache_num_workers`, `cache_dir`

#### 2. Model Configuration (`model:`)
- **Model selection**: `resnet3d`, `densenet3d`, `simple_cnn`, `securefed_cnn`, `rosanna_cnn`
- **Architecture params**: `model_depth`, `growth_rate`, `block_config`
- **Pretrained models**: `pretrained_checkpoint`, `freeze_encoder`

#### 3. Training Configuration (`training:`)
- **Optimization**: `batch_size`, `learning_rate`, `weight_decay`, `num_epochs`
- **Advanced features**: `mixed_precision`, `gradient_accumulation_steps`
- **Loss functions**: `cross_entropy`, `focal` (with `focal_alpha`, `focal_gamma`)
- **Class balancing**: `use_class_weights`, `class_weight_type`, `manual_class_weights`
- **Checkpointing**: `save_best`, `save_latest`, `save_regular`, `save_frequency`

#### 4. Weights & Biases Configuration (`wandb:`)
- **Experiment tracking**: `use_wandb`, `project`, `entity`
- **Organization**: `tags`, `notes`, `run_name`

## Data Format

### Option A: Raw NIfTI Images (`.nii` / `.nii.gz`)

- 3D MRI images in .nii or .nii.gz format
- A CSV label file with the following columns:
  - `image_id`: The ID of the image in the ADNI database (without 'I' prefix)
  - `subject_id`: The ID of the subject in the ADNI database
  - `DX`: Diagnosis group (AD, MCI, CN)
  - `DX_bl`: (optional) for filtering MCI subtypes: SMC, EMCI, or LMCI

Images should be organized as:
```
<root_img_dir>/
└── <subject_id>/
    └── <intermediate_metadata_info>/
        └── ADNI_<subject_id>_<metadata_info>_I<image_id>.nii.gz
```

### Option B: Pre-processed PyTorch Tensors (`.pt`)

이미 전처리가 완료된 3D MRI 텐서를 `.pt` 파일로 저장한 경우 `tensor_folder` 데이터셋 타입을 사용합니다.

#### 1. 필요한 파일 준비

디렉토리 구조를 아래와 같이 프로젝트 루트에 배치합니다:

```
fl-adni-classification/          # 프로젝트 루트
├── 3D_tensors/                  # tensor_dir (텐서 폴더)
│   ├── CN/
│   │   ├── 001.pt
│   │   ├── 002.pt
│   │   └── ...
│   ├── MCI/
│   │   ├── 003.pt
│   │   └── ...
│   └── AD/
│       ├── 004.pt
│       └── ...
├── csv_splits_all_mri_scan_list.csv   # 마스터 CSV
└── ...
```

- **`3D_tensors/`**: 레이블 이름(CN, MCI, AD)으로 분류된 `.pt` 파일들. 각 `.pt` 파일은 `torch.save()`로 저장된 4D `[1, D, H, W]` 텐서

**텐서 사양** (`3D Tensor Creation_Custom.py` 기준):

| 항목 | 값 |
|------|-----|
| Shape | `(1, 192, 192, 192)` |
| dtype | `float32` |
| 채널 | 1 (그레이스케일) |
| 공간 해상도 | 192 × 192 × 192 voxel |

생성 과정:
1. `.nii.gz` 로드 → numpy array `(xdim, ydim, zdim)`
2. 256 × 256 × 256으로 zero-padding (중앙 정렬)
3. `scipy.ndimage.zoom`으로 192 × 192 × 192로 리사이즈 (bilinear, order=1)
4. `reshape(1, 192, 192, 192)` → channel 차원 추가
5. `float32` 변환 후 `torch.save()`로 저장
- **마스터 CSV**: 아래 칼럼을 포함해야 합니다

| 칼럼 | 설명 | 예시 |
|------|------|------|
| `pt_index` | `.pt` 파일명 (확장자 제외) | `001` |
| `patient_id` | 환자 고유 ID (split에 사용) | `P_0042` |
| `label` | 진단 레이블 | `CN`, `MCI`, `AD` |
| `image_path` | (선택) 원본 이미지 경로 | `sub-01/anat/T1w.nii.gz` |
| `image_id` | (선택) 이미지 ID | `I12345` |

#### 2. 환자 단위 Train/Val Split 생성

같은 환자의 모든 스캔이 동일한 split에 포함되도록 환자 단위로 분할합니다:

```bash
python scripts/split_by_patient.py \
  --csv csv_splits_all_mri_scan_list.csv \
  --output_dir csv_splits \
  --train_ratio 0.8 \
  --seed 42
```

실행 결과 `csv_splits/train.csv`와 `csv_splits/val.csv`가 생성됩니다.

#### 3. 학습 실행

```bash
python scripts/train.py --config configs/tensor_resnet18.yaml
```

`configs/tensor_resnet18.yaml`의 주요 설정:

```yaml
data:
  train_csv_path: "csv_splits/train.csv"
  val_csv_path: "csv_splits/val.csv"
  tensor_dir: "3D_tensors"           # .pt 파일 루트 폴더
  dataset_type: "tensor_folder"      # 텐서 폴더 데이터셋 사용
  resize_size: [128, 128, 128]
  classification_mode: "CN_MCI_AD"   # 3-class (CN=0, MCI=1, AD=2)

model:
  name: "resnet3d"
  num_classes: 3
  model_depth: 18
```

경로가 다른 경우 yaml 파일의 `tensor_dir`, `train_csv_path`, `val_csv_path`를 수정하면 됩니다.

#### 4. 전체 순서 요약

```bash
# 0. 환경 설정
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .

# 1. 데이터 배치: 3D_tensors/ 폴더와 마스터 CSV를 프로젝트 루트에 복사

# 2. 환자 단위 split 생성
python scripts/split_by_patient.py \
  --csv csv_splits_all_mri_scan_list.csv \
  --output_dir csv_splits

# 3. 학습 시작
python scripts/train.py --config configs/tensor_resnet18.yaml
```

## MRI Preprocessing

The project includes a preprocessing pipeline for standardizing raw ADNI MRI scans:
1. Resampling to 1mm isotropic spacing
2. Registration to ICBM152 standard template
3. Skull stripping using FSL BET

```bash
python scripts/preprocess_mri.py --input input_folder
```

**Requirements:** ANTs, FSL, Python 3.10+

See [docs/MRI_PREPROCESSING.md](docs/MRI_PREPROCESSING.md) for details.

## License

See [LICENSE](LICENSE) file for details.
