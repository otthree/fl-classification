"""Dataset module for loading pre-processed .pt tensor files organized in label folders."""

import os
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class TensorFolderDataset(Dataset):
    """Dataset for pre-processed 3D MRI tensors stored as .pt files.

    Expected directory structure:
        tensor_dir/
            CN/
                001.pt
                002.pt
            MCI/
                003.pt
            AD/
                004.pt

    CSV format (csv_splits_all_mri_scan_list.csv):
        pt_index, image_path, patient_id, image_id, label

    Each .pt file contains a 3D or 4D tensor. If 3D [D,H,W], a channel
    dimension is added to produce [1,D,H,W].
    """

    LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}

    def __init__(
        self,
        csv_path: str,
        tensor_dir: str,
        transform: Optional[Callable] = None,
        classification_mode: str = "CN_MCI_AD",
    ):
        """Initialize the dataset.

        Args:
            csv_path: Path to CSV with columns: pt_index, label (and optionally patient_id, etc.)
            tensor_dir: Root directory containing label subfolders with .pt files
            transform: Optional MONAI-style dict transform applied to {"image": tensor, "label": int}
            classification_mode: "CN_MCI_AD" (3-class) or "CN_AD" (2-class)
        """
        self.csv_path = csv_path
        self.tensor_dir = tensor_dir
        self.transform = transform
        self.classification_mode = classification_mode

        # Build label map based on classification mode
        if classification_mode == "CN_AD":
            self.label_map = {"CN": 0, "MCI": 1, "AD": 1}
        else:
            self.label_map = dict(self.LABEL_MAP)

        # Read CSV and build data list
        df = pd.read_csv(csv_path)
        self._validate_csv(df)

        self.data_list: List[Dict[str, Any]] = []
        skipped = 0

        for _, row in df.iterrows():
            label_str = str(row["label"]).strip()
            if label_str not in self.label_map:
                skipped += 1
                continue

            pt_index = str(row["pt_index"]).strip()
            tensor_path = os.path.join(tensor_dir, label_str, f"{pt_index}.pt")

            if not os.path.isfile(tensor_path):
                skipped += 1
                continue

            self.data_list.append({
                "image": tensor_path,
                "label": self.label_map[label_str],
            })

        print(f"TensorFolderDataset: loaded {len(self.data_list)} samples "
              f"(skipped {skipped}), mode={classification_mode}")

        # Print class distribution
        from collections import Counter
        label_counts = Counter(d["label"] for d in self.data_list)
        for cls_idx in sorted(label_counts.keys()):
            print(f"  Class {cls_idx}: {label_counts[cls_idx]} samples")

    @staticmethod
    def _validate_csv(df: pd.DataFrame) -> None:
        """Check that required columns exist."""
        required = {"pt_index", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.data_list[index]
        tensor = torch.load(entry["image"], map_location="cpu", weights_only=True)

        # Ensure 4D: [C, D, H, W]
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        sample = {"image": tensor, "label": entry["label"]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
