from pathlib import Path

from frame_extractor.global_video_processor import GlobalVideoProcessor
from image_generator.background_img_generator import BackgroundImageGenerator
from image_generator.utils import BgImageGenConfig

if __name__ == "__main__":
    config = BgImageGenConfig(
        window_size=10,
        num_median_images=200,
        max_cycles=10,
        jump_size_from_last=1,
        apply_clahe="post",
        mask_dilation=15,
        median_computation="cupy",
    )

    # in_dir = Path(r"D:\Bachelorarbeit\frame_extractor_playground\input")
    # extracted_frames_out_dir = Path(r"D:\Bachelorarbeit\frame_extractor_playground\output")
    # background_out_dir = Path(r"D:\Bachelorarbeit\frame_extractor_playground\backgrounds")
    in_dir = Path(r"/mnt/d/Bachelorarbeit/frame_extractor_playground/input")
    extracted_frames_out_dir = Path(r"/mnt/d/Bachelorarbeit/frame_extractor_playground/output")
    background_out_dir = Path(r"/mnt/d/Bachelorarbeit/frame_extractor_playground/backgrounds")

    watcher = GlobalVideoProcessor(base_dir=in_dir, out_dir=extracted_frames_out_dir, interval_in_sec=60, max_workers=2)
    watcher.run()

    big = BackgroundImageGenerator(source_path=extracted_frames_out_dir, output_path=background_out_dir, config=config)
    big.run()
