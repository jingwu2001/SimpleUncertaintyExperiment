import json
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils import data
from torchvision.datasets.folder import pil_loader

class SoftDataset(data.Dataset):
    """A dataset class for handling soft-labeled image data.

    This class extends PyTorch's Dataset class to work with image datasets that have
    soft labels (multiple annotations per image). It supports loading images and their
    corresponding soft labels, and can be used for training, validation, or testing.

    Args:
        name: Name of the dataset.
        root: Root directory where the dataset is stored.
        split: Which split of the data to use. Defaults to 'train'.
        transform: A function/transform that takes in a PIL image
            and returns a transformed version. Defaults to None.
        target_transform: A function/transform that takes in the
            target and transforms it. Defaults to None.

    Raises:
        RuntimeError: If no images are found in the specified root directory.
    """

    def __init__(
        self,
        root: Path,
        split: str = "test",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # Load the soft labels
        root = Path(root)
        self.load_raw_annotations(root / "annotations.json")

        self.root = root.parent
        self.samples = self.file_path_to_img_id.keys()

        # Restrict self.samples to val/test
        current_folds = []
        if split == "val":
            current_folds = [f"fold{i}" for i in range(1, 3)]
        elif split == "test":
            current_folds = [f"fold{i}" for i in range(3, 6)]
        elif split == "all":
            current_folds = [f"fold{i}" for i in range(1, 6)]
        self.samples = [s for s in self.samples if any(f in s for f in current_folds)]

        if len(self.samples) == 0:
            msg = f"Found 0 images in subfolders of {root}"
            raise RuntimeError(msg)

        self.transform = None
        self.target_transform = None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.is_ood = False

    # def __getitem__(self, index: int) -> tuple[Image.Image, Tensor]:
    def __getitem__(self, index: int): # -> Tuple[Image, Tensor]:

        """Retrieves an item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            A tuple containing the image and its soft label.
        """
        path_str = self.samples[index]
        full_path_str = str(self.root / path_str)
        target = self.soft_labels[self.file_path_to_img_id[path_str], :]
        img = pil_loader(full_path_str)

    
        if self.transform is not None:
            img = self.transform(img)
        elif self.transform is not None and self.is_ood:
            rng = np.random.default_rng(seed=index)
            img = self.transform(img, rng)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def set_ood(self) -> None:
        """Sets the dataset to use out-of-distribution transform."""
        self.is_ood = True

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.samples)

    def load_raw_annotations(self, path: Path) -> None:
        """Loads and processes raw annotations from a JSON file.

        Args:
            path: Path to the JSON file containing annotations.
        """
        with path.open() as f:
            raw = json.load(f)

            # Collect all annotations
            img_filepath = []
            labels = []
            for annotator in raw:
                for entry in annotator["annotations"]:
                    # Add only valid annotations to table
                    label = entry["class_label"]
                    if label is not None:
                    # if (label := entry["class_label"]) is not None:
                        img_filepath.append(entry["image_path"])
                        labels.append(label)

            # Summarize the annotations
            unique_img_file_path = sorted(set(img_filepath))
            file_path_to_img_id = {
                filepath: i for i, filepath in enumerate(unique_img_file_path)
            }

            unique_labels = sorted(set(labels))
            class_name_to_label_id = {label: i for i, label in enumerate(unique_labels)}

            soft_labels = np.zeros(
                (len(unique_img_file_path), len(unique_labels)), dtype=np.int64
            )

            # for filepath, classname in zip(img_filepath, labels, strict=True):
            for filepath, classname in zip(img_filepath, labels):
                soft_labels[
                    file_path_to_img_id[filepath], class_name_to_label_id[classname]
                ] += 1

            # soft_labels = np.concatenate(
            #     (soft_labels, soft_labels.argmax(axis=-1, keepdims=True)), axis=-1
            # )
            soft_labels = np.concatenate(
                (soft_labels, soft_labels.argmax(axis=-1)[..., None]),
                axis=-1
            )
            self.soft_labels = torch.from_numpy(soft_labels)
            self.file_path_to_img_id = file_path_to_img_id


class CIFAR10HCombinedLabels(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset, combined_labels: torch.Tensor):
        """
        base_dataset:      your torchvision.datasets.CIFAR10 test split (with any transform)
        combined_labels:   Tensor of shape (N, 12) produced by concatenating
                           the 11 soft labels + 1 hard label in test order
        """
        assert len(base_dataset) == combined_labels.size(0)
        self.base = base_dataset
        self.labels12 = combined_labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]          # ignore the old hard label
        return img, self.labels12[idx]   # returns (C,H,W) and Tensor(12,)

# ——— USAGE ———

# 1) Suppose you've already done:
#    soft_labels: Tensor (N,11)
#    hard_labels: Tensor (N,) or (N,1)
#    test_ids:    array s.t. soft_labels[i] corresponds to hard_labels[test_ids[i]]
#    perm = argsort(test_ids)
#    then:
#    soft_in_test_order = soft_labels[perm]
#    hard = hard_labels.unsqueeze(1)      # ensure shape (N,1)
#    combined = torch.cat([soft_in_test_order, hard], dim=1)  # (N,12)

# 2) Build the combined dataset:
