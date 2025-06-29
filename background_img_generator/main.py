from pathlib import Path

from frame_extractor.global_video_processor import GlobalVideoProcessor

if __name__ == "__main__":
    in_dir = Path(r"D:\Bachelorarbeit\frame_extractor_playground\input")
    out_dir = Path(r"D:\Bachelorarbeit\frame_extractor_playground\output")
    watcher = GlobalVideoProcessor(base_dir=in_dir, out_dir=out_dir, interval_in_sec=180, max_workers=2)
    watcher.run()
