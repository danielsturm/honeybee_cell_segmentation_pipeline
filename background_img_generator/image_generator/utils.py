import time
from functools import wraps
from pydantic import BaseModel
from typing import Literal


def timed(name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            label = name or func.__name__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"[TIMER] {label} took {end - start:.2f} seconds")
            return result

        return wrapper

    return decorator


class BgImageGenConfig(BaseModel):
    """
    Configuration class for background image generation.

    Attributes:
        window_size (int):
            Number of consecutive frames used to compute each rolling median image.
            Larger window sizes produce smoother intermediate results but require more memory and compute.

        num_median_images (int):
            Number of rolling median images to compute per background generation run.
            The final background is computed by combining these intermediate images.

        max_cycles (int):
            The number of (final) background images that are created per camera and per execution.
            If None is provided it will produce as many as possible, which might take very long.
            None is default

        jump_size_from_last (int):
            Step size to skip from the last processed image before starting the next background computation.
            For example, a jump_size_from_last of 1 will continue from the immediate next image;
            higher values will skip additional images between background generations.

        apply_clahe (Literal["intermediate", "post"]):
            When to apply CLAHE contrast enhancement:
                - "intermediate": applies CLAHE to each rolling median image before combining.
                - "post": applies CLAHE only to the final background image.

        mask_dilation (Literal[0, 9, 15, 25]):
            Size of morphological dilation kernel applied to masks during preprocessing.
            Use 0 to disable dilation.

        median_computation (Literal["cupy", "cuda_support", "masked_array"]):
            Method to compute the median across frames:
                - "cupy": uses CuPy for GPU-accelerated computation.
                - "cuda_support": uses PyTorch tensors with CUDA for computation.
                - "masked_array": uses NumPy masked arrays on CPU (slowest).

        segmentation_model (Literal["unet_effnetb0"]):
            Name of the segmentation model used to generate masks for bee removal.
            Currently supports "unet_effnetb0".

        device (Literal["cuda", "cpu"]):
            Processing device for segmentation and median computations.
            Use "cuda" for GPU acceleration if available, otherwise "cpu".
    """

    window_size: int = 10
    num_median_images: int = 48
    max_cycles: int = None
    jump_size_from_last: int = 1
    apply_clahe: Literal["intermediate", "post"] = "post"
    mask_dilation: Literal[0, 9, 15, 25] = 0
    median_computation: Literal["cupy", "cuda_support", "masked_array"] = "cupy"
    segmentation_model: Literal["unet_effnetb0"] = "unet_effnetb0"
    device: Literal["cuda", "cpu"] = "cuda"
