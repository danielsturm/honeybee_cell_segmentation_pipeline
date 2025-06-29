from pathlib import Path
import threading
from datetime import datetime
import re
from collections import defaultdict

from frame_extractor.frame_extractor import FrameExtractor
from frame_extractor.utils import setup_logger


class GlobalVideoProcessor:
    def __init__(
        self,
        base_dir: Path,
        out_dir: Path,
        file_format: str = "png",
        interval_in_sec=5,
        max_workers=2,
        fps=3,
    ):
        self.base_dir = base_dir
        self.out_dir = out_dir
        log_file = self.out_dir / "frame_extractor.log"
        self.logger = setup_logger("GlobalVideoProcessor", log_file)

        self.file_format = file_format
        self.interval_in_sec = interval_in_sec
        self.max_workers = max_workers
        self.frame_extractor = FrameExtractor(logger=self.logger, interval_sec=self.interval_in_sec, fps=fps)

    def _find_video_dirs(self) -> list[Path]:
        pattern = re.compile(r"^\d{8}$")
        valid_dirs = []
        for p in self.base_dir.iterdir():
            if p.is_dir() and pattern.match(p.name):
                try:
                    datetime.strptime(p.name, "%Y%m%d")
                    valid_dirs.append(p)
                except ValueError:
                    pass
        return valid_dirs

    def _find_cam_dirs(self, base_path: Path) -> list[Path]:
        pattern = re.compile(r"^cam-\d$")

        matches = [p for p in base_path.iterdir() if p.is_dir() and pattern.match(p.name)]
        return matches

    def _collect_video_files(self) -> dict[str, list[tuple[Path, Path]]]:
        video_paths = defaultdict(list)
        day_dirs = self._find_video_dirs()
        for day_dir in sorted(day_dirs):
            cam_dirs = self._find_cam_dirs(day_dir)
            for cam_dir in cam_dirs:
                camera_id = cam_dir.name
                video_files = sorted(cam_dir.glob("*.mp4"))
                for video_file in video_files:
                    txt_file = video_file.with_suffix(".txt")
                    if not txt_file.exists():
                        self.logger.warning(f"Skipping {video_file.name}: missing .txt file")
                        continue
                    if not video_file.exists():
                        self.logger.warning(f"Skipping {txt_file.name}: missing .mp4 file")
                        continue
                    video_paths[camera_id].append((video_file, txt_file))
        return video_paths

    def _get_last_processed_timestamps(self, portion_size: int) -> dict[str, list[Path]]:

        pattern_template = r"^{camera_id}_\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z\.png$"

        last_processed = {}

        cam_dirs = self._find_cam_dirs(self.out_dir)
        for cam_dir in cam_dirs:
            camera_id = cam_dir.name
            pattern = re.compile(pattern_template.format(camera_id=camera_id))
            valid_files = [img for img in cam_dir.glob(f"*.{self.file_format}") if pattern.match(img.name)]
            if not valid_files:
                continue
            valid_files.sort()
            last_files = valid_files[-portion_size:]
            last_processed[camera_id] = last_files
        return last_processed

    def _find_resume_index_per_camera(
        self, video_files: list[tuple[Path, Path]], last_processed_ts_files: list[Path]
    ) -> int:
        if not last_processed_ts_files or last_processed_ts_files is None:
            return 0

        extracted_frame_stems = {f.stem for f in last_processed_ts_files}

        for idx in range(len(video_files) - 1, -1, -1):
            video_file, txt_file = video_files[idx]
            if self._get_file_stem_prefix(video_file) in extracted_frame_stems:
                return idx

        return 0

    def _get_file_stem_prefix(self, file: Path) -> str:
        return file.stem.split("--")[0]

    def _get_all_cam_start_indices(
        self,
        video_files: dict[str, list[tuple[Path, Path]]],
        last_extracted_frames: dict[str, list[Path]],
    ) -> dict[str, int]:
        start_indices = {}
        for cam_name, videos in video_files.items():
            last_frames = last_extracted_frames.get(cam_name)
            index_to_resume = self._find_resume_index_per_camera(
                video_files=videos, last_processed_ts_files=last_frames
            )
            start_indices[cam_name] = index_to_resume
        return start_indices

    def _create_out_dirs_per_cam(self, cams: list[str]) -> None:
        for cam in cams:
            Path.mkdir(self.out_dir / cam, exist_ok=True)

    def _process_camera_videos(
        self,
        cam_name: str,
        videos: list[tuple[Path, Path]],
        start_idx: int,
        semaphore: threading.Semaphore,
    ):
        with semaphore:
            self.logger.info(f"Processing camera {cam_name} starting at index {start_idx}")

            if self.interval_in_sec <= 60:
                for video_file, txt_file in videos[start_idx:]:
                    try:
                        self.frame_extractor.interval_sec = self.interval_in_sec
                        self.frame_extractor.extract_from(video_file, self.out_dir / cam_name)
                    except Exception as e:
                        self.logger.error(f"Error processing {video_file.name}: {e}")
            else:
                N = self.interval_in_sec // 60
                for idx in range(start_idx, len(videos), N):
                    video_file, txt_file = videos[idx]
                    try:
                        self.frame_extractor.interval_sec = 60
                        self.frame_extractor.extract_from(video_file, self.out_dir / cam_name)
                    except Exception as e:
                        self.logger.error(f"Error processing {video_file.name}: {e}")

    def run(self):
        video_files = self._collect_video_files()
        self._create_out_dirs_per_cam(list(video_files.keys()))
        portion_size = int(60 / self.interval_in_sec) if self.interval_in_sec < 60 else 1
        last_extracted_frames = self._get_last_processed_timestamps(portion_size=portion_size)
        start_indices = self._get_all_cam_start_indices(video_files, last_extracted_frames)
        self.logger.debug(f"Start indices: {start_indices}")

        threads = []
        semaphore = threading.Semaphore(self.max_workers)

        for cam_name, videos in video_files.items():
            start_idx = start_indices.get(cam_name, 0)

            t = threading.Thread(
                target=self._process_camera_videos,
                args=(cam_name, videos, start_idx, semaphore),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
