"""Run the ONNX model on a video (CPU), draw ripeness boxes + per-frame counts, save mp4 + GIF.

Usage: uv run python demo.py runs/nano/inference_model.onnx data/video/pexels_10179855.mp4
"""
import subprocess
import sys
import time

import cv2
import numpy as np
import supervision as sv

from evaluate import CLASSES, Detector

COLORS = sv.ColorPalette([sv.Color.from_hex(h) for h in ("#3fa34d", "#f2a541", "#e03131")])  # unripe, turning, ripe
CONF = 0.5


def main(onnx_path: str, video: str, out: str = "assets/demo") -> None:
    det = Detector(onnx_path)
    info = sv.VideoInfo.from_video_path(video)
    box = sv.BoxAnnotator(color=COLORS, thickness=3)
    times = []
    with sv.VideoSink(f"{out}.mp4", info) as sink:
        for frame in sv.get_video_frames_generator(video):
            t = time.perf_counter()
            d = det(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=CONF).with_nms(0.5, class_agnostic=True)
            times.append(time.perf_counter() - t)
            frame = box.annotate(frame, d)
            # Overlay: per-class counts in the box colours (doubles as the legend) + CPU speed.
            s = frame.shape[1] / 1280  # scale text to the video width
            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s, max(1, round(2 * s))
            cv2.rectangle(frame, (0, 0), (frame.shape[1], int(56 * s)), (0, 0, 0), -1)
            x = int(16 * s)
            for i, c in enumerate(CLASSES):
                text = f"{c} {(d.class_id == i).sum()}"
                cv2.putText(frame, text, (x, int(38 * s)), font, fs, COLORS.by_idx(i).as_bgr(), th, cv2.LINE_AA)
                x += cv2.getTextSize(text, font, fs, th)[0][0] + int(36 * s)
            text = f"CPU only | ONNX | {1 / np.mean(times[-30:]):.0f} FPS"
            w = cv2.getTextSize(text, font, fs, th)[0][0]
            cv2.putText(frame, text, (frame.shape[1] - w - int(16 * s), int(38 * s)), font, fs, (255, 255, 255), th,
                        cv2.LINE_AA)
            sink.write_frame(frame)
    print(f"{len(times)} frames, mean {np.mean(times) * 1000:.1f} ms/frame end-to-end (preprocess + model + postprocess)")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{out}.mp4", "-vf",
                    "fps=12,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                    f"{out}.gif"], check=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
