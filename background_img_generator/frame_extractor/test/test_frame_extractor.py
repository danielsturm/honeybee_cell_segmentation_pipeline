import subprocess
import time
from pathlib import Path
import pytest
import platform
from background_img_generator.frame_extractor.global_video_processor import VideoWatcher
import threading


@pytest.fixture(scope="module")
def ffmpeg_path() -> Path:
    bin_dir = Path(__file__).parents[1] / "bin"
    assert bin_dir.is_dir()
    return bin_dir / ("ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg_linux")


def generate_test_video(ffmpeg_bin: Path, output_path: Path, duration: int = 60):
    cmd = [
        str(ffmpeg_bin),
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=1280x720:d={duration}",
        "-t",
        str(duration),
        "-vf",
        "drawtext=text='%{pts\\:hms}':x=(w-text_w)/2:y=(h-text_h)/2:" "fontsize=48:fontcolor=white,fps=6",
        "-g",
        "6",
        "-c:v",
        "libx265",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def create_fake_txt_for_video(video_path: Path, start_time: float):
    txt_path = video_path.with_suffix(".txt")
    with txt_path.open("w") as f:
        for i in range(360):
            ts = time.strftime("%Y%m%dT%H%M%S", time.localtime(start_time + i / 6))
            ms = int((i % 6) * 166.6667)
            f.write(f"frame-{ts}.{ms:03d}Z\n")


def clean_old_test_results(video_dir, output_dir):
    for file in video_dir.glob("*.*"):
        file.unlink()

    for f in video_dir.glob("*"), output_dir.glob("extracted_frames/*"):
        for file in f:
            file.unlink()


def create_test_result_dir():
    test_output_dir = Path(__file__).parent / "test_output"
    video_dir = test_output_dir / "videos"
    output_dir = test_output_dir / "results"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return video_dir, output_dir


def test_video_watcher_extracts_frames(ffmpeg_path):

    file_format = "png"
    video_dir, output_dir = create_test_result_dir()
    clean_old_test_results(video_dir, output_dir)

    # Create first video and .txt before starting the watcher
    video1 = video_dir / "video1.mp4"
    start_time1 = time.time()
    generate_test_video(ffmpeg_path, video1)
    create_fake_txt_for_video(video1, start_time1)

    # Start the watcher
    watcher = VideoWatcher(watch_dir=video_dir, out_dir=output_dir, interval_sec=5)
    watcher_thread = threading.Thread(target=watcher.start, daemon=True)
    watcher_thread.start()

    # Give the watcher some time to pick up the first video
    time.sleep(5)

    # Create second video and .txt after watcher has started
    video2 = video_dir / "video2.mp4"
    start_time2 = start_time1 + 60
    generate_test_video(ffmpeg_path, video2)
    create_fake_txt_for_video(video2, start_time2)

    # Allow processing to complete
    time.sleep(15)

    # Collect extracted frames
    extracted_dir = output_dir / "extracted_frames"
    extracted_frames = list(extracted_dir.glob(f"*.{file_format}"))
    assert len(extracted_frames) > 0

    assert all(f.suffix == f".{file_format}" for f in extracted_frames)
