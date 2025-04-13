import random
from itertools import chain
from pathlib import Path

import torchvision.transforms as transforms
from PIL import Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


def train_dataloader(path: Path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = path / "train"

    transform = None
    if use_transform:
        transform = PairCompose(
            [PairRandomCrop(256), PairRandomHorizontalFilp(), PairToTensor()]
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
    image_dir = path / "test"
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(path / "train", is_valid=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir: Path, transform=None, is_test=False, is_valid=False):
        self.image_dir = image_dir
        self.image_list = []
        if is_test:
            self.image_list.extend(
                chain(
                    image_dir.rglob("*.jpeg"),
                    image_dir.rglob("*.jpg"),
                    image_dir.rglob("*.png"),
                )
            )
        else:
            dir_list = sorted(list(image_dir.iterdir()))
            if is_valid:
                for dir in dir_list[: -int(len(dir_list) * 0.3)]:
                    self.image_list.extend(
                        chain(
                            dir.rglob("*.jpeg"), dir.rglob("*.jpg"), dir.rglob("*.png")
                        )
                    )
            else:
                for dir in dir_list[: int(len(dir_list) * 0.7)]:
                    self.image_list.extend(
                        chain(
                            dir.rglob("*.jpeg"), dir.rglob("*.jpg"), dir.rglob("*.png")
                        )
                    )
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        blur_path = self.image_list[idx]
        image = Image.open(blur_path)
        label = Image.open(self._blur_to_sharp_path(blur_path))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            return image, label, blur_path  # include name if needed
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


class PairRandomCrop(transforms.RandomCrop):
    def __call__(self, image, label):
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(
                image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode
            )
            label = F.pad(
                label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode
            )
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(
                image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode
            )
            label = F.pad(
                label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode
            )

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairRandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)
