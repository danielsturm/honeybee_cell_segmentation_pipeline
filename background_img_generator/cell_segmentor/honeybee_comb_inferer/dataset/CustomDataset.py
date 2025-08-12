import os
from pathlib import Path

import cv2
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

project_path = Path(__file__).parent.parent.parent


class CustomDataset(Dataset):
    def __init__(self, images_path: str = "data/images"):
        self.images_folder = os.path.join(project_path, images_path)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        self.images = [
            i for i in os.listdir(self.images_folder)
            if i != "README.md" and i.lower().endswith(valid_extensions)
        ]
        self.transforms = self.get_transforms()

    def get_transforms(self) -> A.core.composition.Compose:

        list_trans = [A.Normalize(mean=0, std=1), ToTensorV2()]
        list_trans = A.Compose(list_trans)
        return list_trans

    def __getitem__(self, index: int):

        img_path = os.path.join(self.images_folder, self.images[index])

        image = cv2.imread(img_path, 0)
        if image is None:
            # Log a warning and optionally skip this file or handle it in a defined way
            print(f"Warning: Could not load image: {img_path}")
            raise ValueError(f"Image {img_path} could not be loaded.")

        height = image.shape[0] // 32 * 32
        width = image.shape[1] // 32 * 32

        # save difference in dims for further interpolation of the mask to the same dim as input
        diff_height = image.shape[0] - height
        diff_width = image.shape[1] - width

        image = image[:height, :width]

        transformation = self.transforms(image=image)
        img_aug = transformation["image"]

        return img_aug, img_path, (diff_height, diff_width)

    def __len__(self) -> int:
        return len(self.images)
