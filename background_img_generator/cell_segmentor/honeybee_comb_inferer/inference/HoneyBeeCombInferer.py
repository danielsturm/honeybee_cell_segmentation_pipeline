from typing import List, Optional, Tuple, Union

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from monai.inferers import SlidingWindowInferer
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from cell_segmentor.honeybee_comb_inferer.config import config_default, label_classes_default
from cell_segmentor.honeybee_comb_inferer.dataset.CustomDataset import CustomDataset
from cell_segmentor.honeybee_comb_inferer.model.HoneyBeeCombSegmentationModel import HoneyBeeCombSegmentationModel
from cell_segmentor.honeybee_comb_inferer.utils.utils import (
    get_cmap_and_labels_for_plotting,
    read_config,
    seed_everything,
)


class HoneyBeeCombInferer:
    def __init__(
        self,
        model_name: str,
        path_to_pretrained_models: str,
        label_classes_config: Union[str, List[dict]] = label_classes_default,
        config: Union[str, dict] = config_default,
        sw_inference: bool = True,
        device: str = "cpu",
    ):
        """
        class for performing semantic segmentation of honey bee comb

        Parameters
        ----------
            model_name: str
                filename of the pretrained model to be used for inference.
                Should be located in 'path_to_pretrained_models'.
            path_to_pretrained_models: str
                path where 'model_name' is located
            label_classes_config: Union[str, List[dict]]
                path to json or read json extracted from 'hasty.ai' including classes and colors (used for plotting).
                Default is read from "data/label_classes.json". List of dicts.
            config: Union[str, dict]
                path or dictionary config, which includes parameters for dataloader and sliding-window inference.
            sw_inference: bool
                boolean value (True/False) whether to apply sliding-window inference.
            device: str
                on which device should inference run, pytorch format: 'cpu','cuda','cuda:1'.

        Example usage:
            >>> from honeybee_comb_inferer.inference import HoneyBeeCombInferer
            >>> model = HoneyBeeCombInferer(model_name = 'model-name', path_to_pretrained_models = 'path-to-pretrained-models', device = 'cuda')
            >>> model.infer(image = 'path-to-image')
        """
        self.device = device
        self.sw_inferer = sw_inference

        self.config = self._get_config(config)
        self.cmap, self.patches = get_cmap_and_labels_for_plotting(label_classes_config)

        self.model = HoneyBeeCombSegmentationModel(
            model_name=model_name, device=device, path_to_pretrained_models=path_to_pretrained_models
        )

        if sw_inference:
            self.sw_inferer = SlidingWindowInferer(**self.config["sliding_window_inferer"])

        seed_everything(self.config["random_seed"])

    def infer(
        self, image: Union[Tensor, np.array, str], return_logits: bool = False, pad_to_input_size: bool = True
    ) -> Tensor:

        image = self._check_type_and_read_image_from_str(image)
        if isinstance(image, np.ndarray):
            image, diff_in_dims = self.preprocess_raw_image(image)
            image = image.to(self.device)
        if len(image.size()) < 4:
            image = image.unsqueeze(0)

        if self.sw_inferer:
            inferred_logits = self.sw_inferer(image, self.model)
        else:
            inferred_logits = self.model(image)

        if return_logits:
            return inferred_logits
        else:
            inferred_mask = self._get_mask(inferred_logits)

            if pad_to_input_size:
                return self._pad_mask_to_input_dim(
                    inferred_mask, diff_height=diff_in_dims[0], diff_width=diff_in_dims[1]
                )
            else:
                return inferred_mask

    def infer_batch(self, images_path: str) -> None:
        """
        Function for inferring batch of images. Used for running via the command line to increase speed.
        Saves each inferred mask in the folder 'inferred_masks', which is located parallel to folder with images.
        Each mask will have the same name as the corresponding image used for inference.
        """

        dataset = CustomDataset(images_path)

        for image, image_path, diff_in_dims in tqdm(dataset):
            inferred_mask = self.infer(image.to(self.device), pad_to_input_size=False)
            inferred_mask = self._pad_mask_to_input_dim(
                inferred_mask, diff_height=diff_in_dims[0], diff_width=diff_in_dims[1]
            )

            self._save_batch_inference(inferred_mask, image_path)

        return None

    def infer_without_bees(self, images_path: str) -> None:

        dataset = CustomDataset(images_path)
        dataloader = DataLoader(dataset=dataset, **self.config["dataloader"])

        output_means = []
        output = 0
        c = 0

        for image, image_path, diff_in_dims in tqdm(dataloader):
            inferred_logits = torch.softmax(self.infer(image.to(self.device), return_logits=True).detach(), dim=1).cpu()

            inferred_logits = self._adjust_class_weights(inferred_logits)

            if type(output) is int:
                output = inferred_logits.clone()
            else:
                output = torch.cat([inferred_logits, output])
                if c % 5 == 0:
                    output_means.append(output.mean(dim=0))
                    del output
                    output = 0

            c += 1

        output_means.append(output.mean(dim=0))
        inferred_mask = self._get_mask_no_bees(output_means)

        diff_in_dims = torch.stack(diff_in_dims)
        diff_height = int(diff_in_dims[0, 0])
        diff_width = int(diff_in_dims[1, 0])

        return self._pad_mask_to_input_dim(inferred_mask, diff_height=diff_height, diff_width=diff_width)

    def infer_without_bees_opt(self, images_path: str, skip: int = 1) -> np.ndarray:
        """
        More memory-efficient version that computes a running average of the predicted logits.
        Processes the images in batches and updates a running sum instead of storing all outputs.
        """
        dataset = CustomDataset(images_path)
        dataloader = DataLoader(dataset=dataset, **self.config["dataloader"])

        running_sum = None
        batch_count = 0
        diff_dims = None  # We'll capture diff_in_dims from the first batch
        with torch.no_grad():
            for idx, (image, image_path, diff_in_dims) in enumerate(tqdm(dataloader)):
                # Process only every skip-th batch
                if idx % skip != 0:
                    continue
                try:
                    logits = self.infer(image.to(self.device), return_logits=True)
                except Exception as e:
                    print(f"Skipping batch due to error: {e}")
                    continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Apply softmax to get probabilities and move to CPU
                inferred_logits = torch.softmax(logits.detach(), dim=1).cpu()
                # Adjust class weights as before
                inferred_logits = self._adjust_class_weights(inferred_logits)
                # Compute the mean for this batch
                batch_mean = inferred_logits.mean(dim=0)  # shape: (num_classes, H, W)

                if running_sum is None:
                    running_sum = batch_mean
                    diff_dims = diff_in_dims[0]  # Capture diff dims from first batch (assumed constant)
                else:
                    running_sum += batch_mean
                batch_count += 1

        # Compute the final averaged logits
        final_logits = running_sum / batch_count
        # Apply softmax again then get the predicted mask (does same this as _get_mask_no_bees, using the running sum
        final_logits = torch.softmax(final_logits, dim=0)
        final_mask = torch.argmax(final_logits, dim=0).detach().cpu().numpy()

        # Use the captured diff dimensions to pad back to original image size
        diff_height, diff_width = diff_dims[0], diff_dims[1]
        return self._pad_mask_to_input_dim(final_mask, diff_height=diff_height, diff_width=diff_width)

    def _get_mask(self, inferred_logits: Tensor) -> Tensor:

        inferred_mask = torch.argmax(inferred_logits.squeeze(), dim=0).detach().cpu().numpy()

        return inferred_mask

    def _get_mask_no_bees(self, inferred_logits_means: Tensor) -> Tensor:

        inferred_logits_pred = torch.stack(inferred_logits_means)
        inferred_logits_pred = inferred_logits_pred.mean(dim=0)

        inferred_logits_pred = torch.softmax(inferred_logits_pred, dim=0).clone()
        inferred_mask = torch.argmax(inferred_logits_pred, dim=0).detach().cpu().numpy()

        return inferred_mask

    def _pad_mask_to_input_dim(self, inferred_mask, diff_height: int, diff_width: int) -> np.ndarray:

        return np.pad(inferred_mask, ((0, diff_height), (0, diff_width)))

    def preprocess_raw_image(self, image: np.array) -> Union[Tensor, Tuple[int, int]]:

        height = image.shape[0] // 32 * 32
        width = image.shape[1] // 32 * 32

        diff_height = image.shape[0] - height
        diff_width = image.shape[1] - width

        image = image[:height, :width]

        transformation = self.get_transforms()

        return transformation(image=image)["image"], (diff_height, diff_width)

    def get_transforms(self) -> A.core.composition.Compose:

        list_trans = [A.Normalize(mean=0, std=1), ToTensorV2()]
        list_trans = A.Compose(list_trans)
        return list_trans

    def _adjust_class_weights(self, inferred_logits: Tensor) -> Tensor:

        inferred_logits[:, 1, ...] = 0
        inferred_logits[:, 0, ...] *= 0.35
        inferred_logits[:, 2, ...] *= 0.9
        inferred_logits[:, 8, ...] *= 0.1

        return inferred_logits

    def _check_type_and_read_image_from_str(self, image: Union[str, np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:

        if isinstance(image, str):
            return cv2.imread(image, 0)
        else:
            return image

    def get_processed_labels(self, pred: Tensor):
        # label_processed = np.array([[self.cmap[int(i)] for i in j] for j in tqdm(pred)])  # nested loop label map processing (slower)
        # vectorized label map processing
        max_label = max(self.cmap.keys())
        color_map = np.zeros((max_label + 1, 4), dtype=np.float32)
        # Fill in the color_map with values from self.cmap
        for label, color in self.cmap.items():
            color_map[label] = color
        label_processed = color_map[pred]  # shape will be (H, W, 4)
        return label_processed

    def plot_prediction(
        self,
        pred: Tensor,
        input_image: Optional[Union[np.ndarray, str]] = None,
        mask: Optional[np.ndarray] = None,
    ):

        label_processed = self.get_processed_labels(pred)

        if input_image is not None and mask is not None:
            fig, ax = plt.subplots(3, 1, figsize=(36, 28))

            input_image = self._check_type_and_read_image_from_str(input_image)
            ax[0].imshow(input_image, cmap="gray")
            ax[0].set_title("input image")

            ax[1].imshow(label_processed)
            ax[1].set_title("predicted")

            mask_processed = np.array([[self.cmap[i] for i in j] for j in tqdm(mask)])
            ax[2].imshow(mask_processed)
            ax[2].set_title("ground truth")

        elif input_image is not None or mask is not None:
            fig, ax = plt.subplots(1, 2, figsize=(36, 28))

            if input_image is not None:
                input_image = self._check_type_and_read_image_from_str(input_image)
                ax[0].imshow(input_image, cmap="gray")
                ax[0].set_title("input image")
            elif mask is not None:
                mask_processed = np.array([[self.cmap[i] for i in j] for j in tqdm(mask)])
                ax[0].imshow(input_image, cmap="gray")
                ax[0].set_title("input image")

            ax[1].imshow(label_processed)
            ax[1].set_title("predicted")

        else:
            fig, ax = plt.subplots(1, 1, figsize=(28, 20))

            ax.imshow(label_processed)
            ax.set_title("predicted")

        plt.legend(handles=self.patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        return fig, ax

    def _save_batch_inference(self, pred: Tensor, input_image_path: str) -> None:

        output_path = input_image_path.replace("images", "inferred_masks")

        label_processed = np.array([[self.cmap[int(i)] for i in j] for j in pred])
        fig, ax = plt.subplots(1, 1, figsize=(24, 20))

        ax.imshow(label_processed)
        ax.set_title("predicted")
        plt.legend(handles=self.patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.savefig(output_path)
        plt.close(fig)

        return None

    def _get_config(self, config: Union[str, dict]) -> dict:

        if isinstance(config, str):
            return read_config(config)
        elif isinstance(config, dict):
            return config
        else:
            raise Exception(
                f"'config' should be of type <str> (path) or <dict>, but you provided type <{type(config)}>"
            )
