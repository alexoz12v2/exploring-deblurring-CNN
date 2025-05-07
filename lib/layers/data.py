import random
from itertools import chain
from pathlib import Path

import torch

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

import torchvision.transforms.v2 as v2
from torchvision.io import decode_image, write_png, ImageReadMode


class NormalizeRange(v2.Transform):
    def _transform(self, inpt: torch.Tensor, params):
        if not isinstance(inpt, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if not torch.is_floating_point(inpt):
            inpt = inpt.float()

        return inpt / 255.0


def save_image(image_tensor: torch.Tensor, path: Path):
    # must be C x H x W
    if torch.is_floating_point(image_tensor):
        image_tensor = v2.functional.to_dtype(image_tensor * 255.0, torch.uint8)
    if image_tensor.device.type != 'cpu':
        image_tensor = image_tensor.cpu()

    write_png(image_tensor, str(path))


def train_dataloader(path: Path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = path / "train"

    transform = None
    if use_transform:
        transform = v2.Compose(
            [
                # Autoaugment paper: https://arxiv.org/pdf/1805.09501
                v2.AutoAugment(),
                v2.ToDtype(torch.get_default_dtype()),
                NormalizeRange(),
                v2.ScaleJitter(target_size=(256, 256), scale_range=(0.8, 1.2)),
                v2.RandomResizedCrop(256),
                v2.ColorJitter(), # a quanto pare modificare il colore distrugge tutto?
                v2.RandomInvert(),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        )

    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def test_dataloader(path: Path, batch_size=1, num_workers=0):
    transform = v2.Compose([v2.ToDtype(torch.get_default_dtype()), NormalizeRange()])#, v2.CenterCrop(256)])
    image_dir = path / "test"
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    transform = v2.Compose([v2.ToDtype(torch.get_default_dtype()), NormalizeRange()])#, v2.CenterCrop(256)])
    dataloader = DataLoader(
        DeblurDataset(path / "train", is_valid=True, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir: Path, transform=None, is_test=False, is_valid=False):
        self.image_dir = image_dir
        self.image_list: list[Path] = []
        if is_test:
            self.image_list.extend(
                chain(
                    image_dir.rglob("blur/*.jpeg"),
                    image_dir.rglob("blur/*.jpg"),
                    image_dir.rglob("blur/*.png"),
                )
            )
        else:
            dir_list = sorted(list(image_dir.iterdir()))
            print(dir_list)
            if is_valid:
                for dir in dir_list[int(len(dir_list) * 0.7) :]:
                    self.image_list.extend(
                        chain(
                            dir.rglob("blur/*.jpeg"), dir.rglob("blur/*.jpg"), dir.rglob("blur/*.png")
                        )
                    )
            else:
                for dir in dir_list[: int(len(dir_list) * 0.7)]:
                    print(dir)
                    self.image_list.extend(
                        chain(
                            dir.rglob("blur/*.jpeg"), dir.rglob("blur/*.jpg"), dir.rglob("blur/*.png")
                        )
                    )
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        blur_path: Path = self.image_list[idx]
        image = decode_image(blur_path, mode=ImageReadMode.RGB)
        label = decode_image(self._blur_to_sharp_path(blur_path), mode=ImageReadMode.RGB)

        if self.transform:
            both = torch.cat((image.unsqueeze(0), label.unsqueeze(0)), 0)
            both = self.transform(both)
            image = both[0]
            label = both[1]
        else:
            image = v2.functional.to_dtype(image, torch.get_default_dtype()) / 255.0
            label = v2.functional.to_dtype(label, torch.get_default_dtype()) / 255.0

        if self.is_test:
            return image, label, blur_path.name  # include name if needed
        return image, label

    @staticmethod
    def _blur_to_sharp_path(blur_path: Path):
        parts: list[str] = []
        for part in blur_path.parts:
            if part == "blur":
                parts.append("sharp")
            else:
                parts.append(part)
        parts[len(parts) - 1] = parts[len(parts) - 1].replace("blur", "gt")
        return Path(*parts)

    @staticmethod
    def _check_image(lst: list[Path]):
        for x in lst:
            splits = x.name.split(".")
            if splits[-1] not in ["png", "jpg", "jpeg"]:
                raise ValueError
