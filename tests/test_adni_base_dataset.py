"""Unit tests for ADNI Base Dataset.

This module contains comprehensive unit tests for the ADNIBaseDataset class,
covering initialization, CSV format detection, data standardization, image file
discovery, and error handling scenarios.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from adni_classification.datasets.adni_base_dataset import ADNIBaseDataset


class TestADNIBaseDatasetInit:
    """Test ADNIBaseDataset initialization and parameter validation."""

    def test_init_valid_parameters(self, temp_dir: Path) -> None:
        """Test initialization with valid parameters."""
        # Create test CSV with original format
        csv_data = {
            "Image Data ID": ["I123", "I456", "I789"],
            "Subject": ["136_S_1227", "137_S_1228", "138_S_1229"],
            "Group": ["CN", "MCI", "AD"],
            "Description": ["Test1", "Test2", "Test3"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        # Create test image files
        img_dir = temp_dir / "images"
        img_dir.mkdir()
        for img_id in ["I123", "I456", "I789"]:
            img_subdir = img_dir / img_id
            img_subdir.mkdir()
            (img_subdir / f"test_{img_id}.nii.gz").touch()

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123" / "test_I123.nii.gz"),
                "I456": str(img_dir / "I456" / "test_I456.nii.gz"),
                "I789": str(img_dir / "I789" / "test_I789.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="CN_MCI_AD",
                verbose=False,
            )

            assert dataset.csv_path == str(csv_path)
            assert dataset.img_dir == str(img_dir)
            assert dataset.classification_mode == "CN_MCI_AD"
            assert dataset.mci_subtype_filter is None
            assert dataset.verbose is False
            assert len(dataset.data_list) == 3

    def test_init_invalid_mci_subtype_filter_type(self, temp_dir: Path) -> None:
        """Test initialization with invalid mci_subtype_filter type."""
        csv_path = temp_dir / "test.csv"
        img_dir = temp_dir / "images"

        with pytest.raises(ValueError, match="mci_subtype_filter must be a single subtype"):
            ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                mci_subtype_filter=123,  # Invalid type
                verbose=False,
            )

    def test_init_empty_mci_subtype_filter_list(self, temp_dir: Path) -> None:
        """Test initialization with empty mci_subtype_filter list."""
        csv_path = temp_dir / "test.csv"
        img_dir = temp_dir / "images"

        with pytest.raises(ValueError, match="mci_subtype_filter list cannot be empty"):
            ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                mci_subtype_filter=[],  # Empty list
                verbose=False,
            )

    def test_init_invalid_mci_subtype(self, temp_dir: Path) -> None:
        """Test initialization with invalid MCI subtype."""
        csv_path = temp_dir / "test.csv"
        img_dir = temp_dir / "images"

        with pytest.raises(ValueError, match="Invalid subtype: INVALID"):
            ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                mci_subtype_filter="INVALID",
                verbose=False,
            )

    def test_init_mci_filter_wrong_classification_mode(self, temp_dir: Path) -> None:
        """Test initialization with MCI filter in wrong classification mode."""
        csv_path = temp_dir / "test.csv"
        img_dir = temp_dir / "images"

        with pytest.raises(ValueError, match="mci_subtype_filter can only be used with classification_mode='CN_AD'"):
            ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="CN_MCI_AD",
                mci_subtype_filter="EMCI",
                verbose=False,
            )

    def test_init_valid_mci_subtype_filter_string(self, temp_dir: Path) -> None:
        """Test initialization with valid MCI subtype filter as string."""
        # Create test CSV with DX_bl column for MCI filtering
        csv_data = {
            "Image Data ID": ["I123", "I456", "I789"],
            "Subject": ["136_S_1227", "137_S_1228", "138_S_1229"],
            "Group": ["CN", "MCI", "AD"],
            "DX_bl": ["CN", "EMCI", "AD"],
            "Description": ["Test1", "Test2", "Test3"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
                "I456": str(img_dir / "I456.nii.gz"),
                "I789": str(img_dir / "I789.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="CN_AD",
                mci_subtype_filter="EMCI",
                verbose=False,
            )

            assert dataset.mci_subtype_filter == ["EMCI"]

    def test_init_valid_mci_subtype_filter_list(self, temp_dir: Path) -> None:
        """Test initialization with valid MCI subtype filter as list."""
        csv_data = {
            "Image Data ID": ["I123", "I456", "I789"],
            "Subject": ["136_S_1227", "137_S_1228", "138_S_1229"],
            "Group": ["CN", "MCI", "AD"],
            "DX_bl": ["CN", "EMCI", "AD"],
            "Description": ["Test1", "Test2", "Test3"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
                "I456": str(img_dir / "I456.nii.gz"),
                "I789": str(img_dir / "I789.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="CN_AD",
                mci_subtype_filter=["EMCI", "LMCI"],
                verbose=False,
            )

            assert dataset.mci_subtype_filter == ["EMCI", "LMCI"]

    def test_init_missing_image_files_error(self, temp_dir: Path) -> None:
        """Test initialization with missing image files raises error."""
        csv_data = {
            "Image Data ID": ["I123", "I456", "I789"],
            "Subject": ["136_S_1227", "137_S_1228", "138_S_1229"],
            "Group": ["CN", "MCI", "AD"],
            "Description": ["Test1", "Test2", "Test3"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"
        img_dir.mkdir()

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            # Only return some image files, not all
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
            }

            with pytest.raises(ValueError, match="Image IDs from the CSV could not be found"):
                ADNIBaseDataset(
                    csv_path=str(csv_path),
                    img_dir=str(img_dir),
                    verbose=False,
                )


class TestCSVFormatDetection:
    """Test CSV format detection functionality."""

    def test_detect_original_format(self, temp_dir: Path) -> None:
        """Test detection of original CSV format."""
        csv_data = {
            "Image Data ID": ["I123", "I456"],
            "Subject": ["136_S_1227", "137_S_1228"],
            "Group": ["CN", "MCI"],
            "Description": ["Test1", "Test2"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
                "I456": str(img_dir / "I456.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                verbose=False,
            )

            assert dataset.csv_format == "original"

    def test_detect_alternative_format(self, temp_dir: Path) -> None:
        """Test detection of alternative CSV format."""
        csv_data = {
            "image_id": ["123", "456"],
            "DX": ["CN", "MCI"],
            "Subject": ["136_S_1227", "137_S_1228"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
                "I456": str(img_dir / "I456.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                verbose=False,
            )

            assert dataset.csv_format == "alternative"

    def test_detect_unknown_format_error(self, temp_dir: Path) -> None:
        """Test error handling for unknown CSV format."""
        csv_data = {
            "unknown_column": ["123", "456"],
            "another_column": ["CN", "MCI"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with pytest.raises(ValueError, match="Unknown CSV format"):
            ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                verbose=False,
            )


class TestClassificationModes:
    """Test different classification modes and label mappings."""

    def test_cn_mci_ad_label_mapping(self, temp_dir: Path) -> None:
        """Test CN_MCI_AD classification mode label mapping."""
        csv_data = {
            "Image Data ID": ["I123", "I456", "I789"],
            "Group": ["CN", "MCI", "AD"],
            "Subject": ["S1", "S2", "S3"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
                "I456": str(img_dir / "I456.nii.gz"),
                "I789": str(img_dir / "I789.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="CN_MCI_AD",
                verbose=False,
            )

            expected_labels = {"CN": 0, "MCI": 1, "AD": 2, "Dementia": 2}
            assert dataset.label_map == expected_labels

            # Check data list labels
            labels = [item["label"] for item in dataset.data_list]
            assert labels == [0, 1, 2]  # CN, MCI, AD

    def test_cn_ad_label_mapping(self, temp_dir: Path) -> None:
        """Test CN_AD classification mode label mapping."""
        csv_data = {
            "Image Data ID": ["I123", "I456", "I789"],
            "Group": ["CN", "MCI", "AD"],
            "Subject": ["S1", "S2", "S3"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
                "I456": str(img_dir / "I456.nii.gz"),
                "I789": str(img_dir / "I789.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="CN_AD",
                verbose=False,
            )

            expected_labels = {"CN": 0, "MCI": 1, "AD": 1, "Dementia": 1}
            assert dataset.label_map == expected_labels

            # Check data list labels - MCI becomes AD (label 1)
            labels = [item["label"] for item in dataset.data_list]
            assert labels == [0, 1, 1]  # CN, MCI->AD, AD

    def test_alternative_format_label_mapping(self, temp_dir: Path) -> None:
        """Test label mapping with alternative CSV format."""
        csv_data = {
            "image_id": ["123", "456", "789"],
            "DX": ["CN", "MCI", "Dementia"],
            "Subject": ["S1", "S2", "S3"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
                "I456": str(img_dir / "I456.nii.gz"),
                "I789": str(img_dir / "I789.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="CN_MCI_AD",
                verbose=False,
            )

            # Check that alternative format DX values are mapped correctly
            groups = dataset.data["Group"].tolist()
            assert groups == ["CN", "MCI", "AD"]  # Dementia -> AD

            labels = [item["label"] for item in dataset.data_list]
            assert labels == [0, 1, 2]  # CN, MCI, Dementia->AD


class TestDataStandardization:
    """Test data standardization functionality."""

    def test_original_format_standardization(self, temp_dir: Path) -> None:
        """Test data standardization for original CSV format."""
        csv_data = {
            "Image Data ID": ["I123", "I456", "I789", "I999"],
            "Group": ["CN", "MCI", "AD", "Unknown"],  # Include invalid group
            "Subject": ["S1", "S2", "S3", "S4"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
                "I456": str(img_dir / "I456.nii.gz"),
                "I789": str(img_dir / "I789.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                verbose=False,
            )

            # Check that invalid groups are filtered out
            valid_groups = dataset.data["Group"].unique()
            assert "Unknown" not in valid_groups
            assert set(valid_groups).issubset({"CN", "MCI", "AD"})

    def test_alternative_format_standardization(self, temp_dir: Path) -> None:
        """Test data standardization for alternative CSV format."""
        csv_data = {
            "image_id": ["123.0", "456", "789.0"],  # Include decimal values
            "DX": ["CN", "MCI", "Dementia"],
            "Subject": ["S1", "S2", "S3"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
                "I456": str(img_dir / "I456.nii.gz"),
                "I789": str(img_dir / "I789.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                verbose=False,
            )

            # Check that Image Data ID is correctly created
            image_ids = dataset.data["Image Data ID"].tolist()
            assert image_ids == ["I123", "I456", "I789"]  # Decimal points removed

            # Check that DX is mapped to Group
            groups = dataset.data["Group"].tolist()
            assert groups == ["CN", "MCI", "AD"]  # Dementia -> AD

    def test_mci_subtype_filtering_original_format(self, temp_dir: Path) -> None:
        """Test MCI subtype filtering for original format."""
        csv_data = {
            "Image Data ID": ["I123", "I456", "I789", "I999"],
            "Group": ["CN", "MCI", "MCI", "AD"],
            "DX_bl": ["CN", "EMCI", "LMCI", "AD"],
            "Subject": ["S1", "S2", "S3", "S4"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
                "I456": str(img_dir / "I456.nii.gz"),
                "I789": str(img_dir / "I789.nii.gz"),
                "I999": str(img_dir / "I999.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="CN_AD",
                mci_subtype_filter=["EMCI"],
                verbose=False,
            )

            # Check that only EMCI MCI samples are kept
            mci_samples = dataset.data[dataset.data["Group"] == "MCI"]
            assert len(mci_samples) == 1
            assert mci_samples.iloc[0]["DX_bl"] == "EMCI"

    def test_mci_subtype_filtering_alternative_format(self, temp_dir: Path) -> None:
        """Test MCI subtype filtering for alternative format."""
        csv_data = {
            "image_id": ["123", "456", "789", "999"],
            "DX": ["CN", "MCI", "MCI", "Dementia"],
            "DX_bl": ["CN", "EMCI", "LMCI", "Dementia"],
            "Subject": ["S1", "S2", "S3", "S4"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": str(img_dir / "I123.nii.gz"),
                "I456": str(img_dir / "I456.nii.gz"),
                "I789": str(img_dir / "I789.nii.gz"),
                "I999": str(img_dir / "I999.nii.gz"),
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="CN_AD",
                mci_subtype_filter=["LMCI"],
                verbose=False,
            )

            # Check that only LMCI MCI samples are kept
            mci_samples = dataset.data[dataset.data["Group"] == "MCI"]
            assert len(mci_samples) == 1
            assert mci_samples.iloc[0]["DX_bl"] == "LMCI"

    def test_mci_filtering_missing_dx_bl_column_error(self, temp_dir: Path) -> None:
        """Test error when DX_bl column is missing for MCI filtering."""
        csv_data = {
            "Image Data ID": ["I123", "I456"],
            "Group": ["CN", "MCI"],
            "Subject": ["S1", "S2"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with pytest.raises(ValueError, match="DX_bl column is required for MCI subtype filtering"):
            ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="CN_AD",
                mci_subtype_filter="EMCI",
                verbose=False,
            )

    def test_empty_data_after_filtering_error(self, temp_dir: Path) -> None:
        """Test error when no valid data remains after filtering."""
        csv_data = {
            "invalid_column": ["value1", "value2"],
            "another_invalid": ["value3", "value4"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with pytest.raises(ValueError, match="Unknown CSV format"):
            ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                verbose=False,
            )


class TestImageFileDiscovery:
    """Test image file discovery and mapping functionality."""

    @patch('os.walk')
    def test_find_image_files_original_format(self, mock_walk, temp_dir: Path) -> None:
        """Test image file discovery for original format."""
        csv_data = {
            "Image Data ID": ["I123", "I456"],
            "Group": ["CN", "MCI"],
            "Subject": ["S1", "S2"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        # Mock os.walk to return test directory structure
        mock_walk.return_value = [
            ("/test/images", ["I123", "I456"], []),
            ("/test/images/I123", [], ["test_I123.nii.gz"]),
            ("/test/images/I456", [], ["test_I456.nii"]),
        ]

        dataset = ADNIBaseDataset(
            csv_path=str(csv_path),
            img_dir="/test/images",
            verbose=False,
        )

        # Check that image paths are correctly mapped
        assert "I123" in dataset.image_paths
        assert "I456" in dataset.image_paths
        assert dataset.image_paths["I123"].endswith("test_I123.nii.gz")
        assert dataset.image_paths["I456"].endswith("test_I456.nii")

    @patch('os.walk')
    def test_find_image_files_alternative_format(self, mock_walk, temp_dir: Path) -> None:
        """Test image file discovery for alternative format."""
        csv_data = {
            "image_id": ["123", "456"],
            "DX": ["CN", "MCI"],
            "Subject": ["S1", "S2"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        # Mock os.walk to return test directory structure
        mock_walk.return_value = [
            ("/test/images", ["123", "456"], []),
            ("/test/images/123", [], ["test_123.nii.gz"]),
            ("/test/images/456", [], ["test_456.nii"]),
        ]

        dataset = ADNIBaseDataset(
            csv_path=str(csv_path),
            img_dir="/test/images",
            verbose=False,
        )

        # Check that image paths are correctly mapped with I prefix
        assert "I123" in dataset.image_paths
        assert "I456" in dataset.image_paths

    def test_extract_id_from_filename_with_i_prefix(self, temp_dir: Path) -> None:
        """Test ID extraction from filename with I prefix."""
        csv_data = {
            "Image Data ID": ["I123"],
            "Group": ["CN"],
            "Subject": ["S1"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {"I123": "/test/image_I123.nii.gz"}

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir="/test/images",
                verbose=False,
            )

            # Test the extraction method
            result = dataset._extract_id_from_filename("some_prefix_I123_suffix.nii.gz")
            assert result == "I123"

    def test_extract_id_from_filename_alternative_format(self, temp_dir: Path) -> None:
        """Test ID extraction from filename for alternative format."""
        csv_data = {
            "image_id": ["123"],
            "DX": ["CN"],
            "Subject": ["S1"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {"I123": "/test/image_123.nii.gz"}

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir="/test/images",
                verbose=False,
            )

            # Test the extraction method for alternative format
            result = dataset._extract_id_from_filename("some_prefix_123_suffix.nii.gz")
            assert result == "I123"

    def test_extract_id_from_filename_no_match(self, temp_dir: Path) -> None:
        """Test ID extraction when no valid ID is found."""
        csv_data = {
            "Image Data ID": ["I123"],
            "Group": ["CN"],
            "Subject": ["S1"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {"I123": "/test/image.nii.gz"}

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir="/test/images",
                verbose=False,
            )

            # Test the extraction method with no valid ID
            result = dataset._extract_id_from_filename("invalid_filename.nii.gz")
            assert result is None

    @patch('os.walk')
    def test_prioritize_nii_gz_over_nii(self, mock_walk, temp_dir: Path) -> None:
        """Test that .nii.gz files are prioritized over .nii files."""
        csv_data = {
            "Image Data ID": ["I123"],
            "Group": ["CN"],
            "Subject": ["S1"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        # Mock os.walk to return both .nii and .nii.gz for same ID
        mock_walk.return_value = [
            ("/test/images/I123", [], ["test_I123.nii", "test_I123.nii.gz"]),
        ]

        dataset = ADNIBaseDataset(
            csv_path=str(csv_path),
            img_dir="/test/images",
            verbose=False,
        )

        # Check that .nii.gz is prioritized
        assert dataset.image_paths["I123"].endswith(".nii.gz")


class TestDataListCreation:
    """Test data list creation functionality."""

    def test_create_data_list_structure(self, temp_dir: Path) -> None:
        """Test the structure of created data list."""
        csv_data = {
            "Image Data ID": ["I123", "I456", "I789"],
            "Group": ["CN", "MCI", "AD"],
            "Subject": ["S1", "S2", "S3"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": "/test/I123.nii.gz",
                "I456": "/test/I456.nii.gz",
                "I789": "/test/I789.nii.gz",
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                verbose=False,
            )

            # Check data list structure
            assert len(dataset.data_list) == 3

            for item in dataset.data_list:
                assert "image" in item
                assert "label" in item
                assert isinstance(item["image"], str)
                assert isinstance(item["label"], int)
                assert item["label"] in [0, 1, 2]  # Valid labels for CN_MCI_AD

    def test_create_data_list_label_mapping(self, temp_dir: Path) -> None:
        """Test correct label mapping in data list."""
        csv_data = {
            "Image Data ID": ["I123", "I456", "I789"],
            "Group": ["CN", "MCI", "AD"],
            "Subject": ["S1", "S2", "S3"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": "/test/I123.nii.gz",
                "I456": "/test/I456.nii.gz",
                "I789": "/test/I789.nii.gz",
            }

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="CN_MCI_AD",
                verbose=False,
            )

            # Check specific label mappings
            labels = [item["label"] for item in dataset.data_list]
            assert labels == [0, 1, 2]  # CN=0, MCI=1, AD=2

            # Check image paths are correctly assigned
            for i, item in enumerate(dataset.data_list):
                expected_id = ["I123", "I456", "I789"][i]
                assert item["image"] == f"/test/{expected_id}.nii.gz"


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_nonexistent_csv_file(self, temp_dir: Path) -> None:
        """Test error handling for nonexistent CSV file."""
        nonexistent_csv = temp_dir / "nonexistent.csv"
        img_dir = temp_dir / "images"

        with pytest.raises(FileNotFoundError):
            ADNIBaseDataset(
                csv_path=str(nonexistent_csv),
                img_dir=str(img_dir),
                verbose=False,
            )

    def test_invalid_classification_mode(self, temp_dir: Path) -> None:
        """Test handling of invalid classification mode."""
        # Note: The current implementation doesn't validate classification_mode
        # This test documents the current behavior and can be updated if validation is added
        csv_data = {
            "Image Data ID": ["I123"],
            "Group": ["CN"],
            "Subject": ["S1"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {"I123": "/test/I123.nii.gz"}

            # This should work (no validation currently implemented)
            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                classification_mode="INVALID_MODE",
                verbose=False,
            )

            # Should use default 3-class mapping
            assert dataset.label_map == {"CN": 0, "MCI": 1, "AD": 2, "Dementia": 2}

    def test_verbose_output_capture(self, temp_dir: Path, capsys) -> None:
        """Test that verbose output is properly generated."""
        csv_data = {
            "Image Data ID": ["I123", "I456"],
            "Group": ["CN", "MCI"],
            "Subject": ["S1", "S2"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {
                "I123": "/test/I123.nii.gz",
                "I456": "/test/I456.nii.gz",
            }

            _ = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                verbose=True,
            )

            captured = capsys.readouterr()
            assert "Initializing dataset" in captured.out
            assert "Detected ORIGINAL CSV format" in captured.out
            assert "Final dataset size: 2 samples" in captured.out

    def test_edge_case_single_sample(self, temp_dir: Path) -> None:
        """Test handling of dataset with single sample."""
        csv_data = {
            "Image Data ID": ["I123"],
            "Group": ["CN"],
            "Subject": ["S1"],
        }
        csv_path = temp_dir / "test.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        img_dir = temp_dir / "images"

        with patch.object(ADNIBaseDataset, '_find_image_files') as mock_find:
            mock_find.return_value = {"I123": "/test/I123.nii.gz"}

            dataset = ADNIBaseDataset(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                verbose=False,
            )

            assert len(dataset.data_list) == 1
            assert dataset.data_list[0]["label"] == 0  # CN
            assert dataset.data_list[0]["image"] == "/test/I123.nii.gz"


# Fixtures for testing
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)
