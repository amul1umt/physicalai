# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from IPython.display import Image as IPyImage
from PIL import Image, ImageDraw
from physicalai.inference import InferenceModel


STATE_KEY = "state"
TASK_KEY = "task"
TOP_IMAGE_KEY = "images.top-cam"
GRIPPER_IMAGE_KEY = "images.gripper-cam"
TOP_DATASET_VIDEO_KEY = "observation.images.top-cam"
GRIPPER_DATASET_VIDEO_KEY = "observation.images.gripper-cam"

SO101_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class ReplayEpisode:
    dataset_root: Path
    episode_id: int
    episode_df: pd.DataFrame
    top_video_path: Path
    gripper_video_path: Path
    top_start_frame: int
    gripper_start_frame: int
    fps: float


def download_pi05_package(repo_id: str, assets_dir: Path, model_dir: Path | None = None) -> Path:
    if model_dir is None:
        model_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                local_dir=assets_dir / "models" / repo_id.replace("/", "__"),
                allow_patterns=[
                    "manifest.json",
                    "pi05.xml",
                    "pi05.bin",
                    "tokenizer.xml",
                    "tokenizer.bin",
                    "metadata.yaml",
                    "README.md",
                ],
                local_dir_use_symlinks=False,
            )
        ).resolve()

    required = ["manifest.json", "pi05.xml", "pi05.bin", "tokenizer.xml", "tokenizer.bin"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required PhysicalAI package files in {model_dir}: {missing}")
    return model_dir


def resolve_dataset_root(path: Path, dataset_name: str) -> Path:
    path = Path(path).resolve()
    nested = path / dataset_name
    return nested if nested.exists() else path


def prepare_replay_episode(
    *,
    repo_id: str,
    dataset_name: str,
    assets_dir: Path,
    episode_id: int,
    dataset_dir: Path | None = None,
) -> ReplayEpisode:
    if dataset_dir is None:
        dataset_cache = assets_dir / "datasets" / repo_id.replace("/", "__")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dataset_cache,
            allow_patterns=[f"{dataset_name}/meta/**"],
            local_dir_use_symlinks=False,
        )
        dataset_root = resolve_dataset_root(dataset_cache, dataset_name)
    else:
        dataset_root = resolve_dataset_root(dataset_dir, dataset_name)

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot dataset metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps", 30))

    def download_dataset_file(relative_path: str) -> Path:
        local_path = dataset_root / relative_path
        if local_path.exists():
            return local_path
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=f"{dataset_name}/{relative_path}",
                local_dir=dataset_root.parent,
                local_dir_use_symlinks=False,
            )
        )
        return downloaded.resolve()

    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata files found under {dataset_root / 'meta' / 'episodes'}")

    episodes = pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
    matches = episodes[episodes["episode_index"] == episode_id]
    if matches.empty:
        available = episodes["episode_index"].tolist()
        raise ValueError(f"Episode {episode_id} not found. Available examples: {available[:10]}")
    episode_meta = matches.iloc[0]

    data_chunk = int(episode_meta["data/chunk_index"])
    data_file = int(episode_meta["data/file_index"])
    parquet = download_dataset_file(f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet")
    episode_df = pd.read_parquet(parquet)
    episode_df = episode_df[episode_df["episode_index"] == episode_id].reset_index(drop=True)

    def video_info(video_key: str) -> tuple[Path, int]:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        start_col = f"videos/{video_key}/from_timestamp"
        chunk_index = int(episode_meta[chunk_col]) if chunk_col in episode_meta.index else 0
        file_index = int(episode_meta[file_col]) if file_col in episode_meta.index else data_file
        start_seconds = float(episode_meta[start_col]) if start_col in episode_meta.index else 0.0
        video_path = download_dataset_file(f"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        return video_path, int(round(start_seconds * fps))

    top_video_path, top_start_frame = video_info(TOP_DATASET_VIDEO_KEY)
    gripper_video_path, gripper_start_frame = video_info(GRIPPER_DATASET_VIDEO_KEY)

    return ReplayEpisode(
        dataset_root=dataset_root,
        episode_id=episode_id,
        episode_df=episode_df,
        top_video_path=top_video_path,
        gripper_video_path=gripper_video_path,
        top_start_frame=top_start_frame,
        gripper_start_frame=gripper_start_frame,
        fps=fps,
    )


def read_video_rgb(path: Path, max_frames: int | None = None, start_frame: int = 0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if frames:
        return frames

    try:
        import imageio

        reader = imageio.get_reader(str(path), "ffmpeg")
        try:
            for frame_index, frame in enumerate(reader):
                if frame_index < start_frame:
                    continue
                frames.append(np.asarray(frame[..., :3], dtype=np.uint8))
                if max_frames is not None and len(frames) >= max_frames:
                    break
        finally:
            reader.close()
    except Exception as exc:
        raise RuntimeError(
            "Could not decode replay video. This dataset may require AV1 decode support; "
            "install imageio-ffmpeg or re-encode the videos to H.264."
        ) from exc

    if not frames:
        raise RuntimeError(f"Decoded zero frames from {path}; check codec support and start_frame={start_frame}.")
    return frames


def as_action_vector(values: Any, name: str = "action") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError(f"Expected {name} to be a vector, got scalar value {arr!r}")
    if arr.ndim > 1:
        arr = arr.reshape(-1, arr.shape[-1])[0]
    arr = arr[: len(SO101_JOINT_ORDER)].astype(np.float32)
    if arr.shape[0] != len(SO101_JOINT_ORDER):
        raise ValueError(f"Expected {name} length {len(SO101_JOINT_ORDER)}, got shape {arr.shape}")
    return arr


def make_policy_observation(top_frame: np.ndarray, gripper_frame: np.ndarray, state: Any, task: str) -> dict[str, Any]:
    return {
        STATE_KEY: as_action_vector(state, name="observation.state")[None, :],
        TOP_IMAGE_KEY: top_frame.astype(np.float32)[None, ...] / 255.0,
        GRIPPER_IMAGE_KEY: gripper_frame.astype(np.float32)[None, ...] / 255.0,
        TASK_KEY: [task],
    }


def openvino_config_for_device(cache_dir: Path) -> dict[str, str]:
    return {"CACHE_DIR": str(cache_dir)}


def benchmark_pi05(
    *,
    model_dir: Path,
    replay: ReplayEpisode,
    task: str,
    device: str,
    cache_dir: Path,
    runs: int = 5,
) -> tuple[InferenceModel, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    top_frame = read_video_rgb(replay.top_video_path, max_frames=1, start_frame=replay.top_start_frame)[0]
    gripper_frame = read_video_rgb(replay.gripper_video_path, max_frames=1, start_frame=replay.gripper_start_frame)[0]
    obs = make_policy_observation(top_frame, gripper_frame, replay.episode_df.iloc[0]["observation.state"], task)

    start = time.perf_counter()
    model = InferenceModel.load(model_dir, backend="openvino", device=device, **openvino_config_for_device(cache_dir))
    load_ms = (time.perf_counter() - start) * 1000

    model.reset()
    _ = model.predict_action_chunk(obs)
    first_action = model.select_action(obs)
    timings = []
    for _ in range(runs):
        model.reset()
        start = time.perf_counter()
        chunk = model.predict_action_chunk(obs)
        timings.append((time.perf_counter() - start) * 1000)

    avg_ms = float(np.mean(timings))
    return model, {
        "device": device,
        "load_ms": load_ms,
        "avg_ms": avg_ms,
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "fps": float(1000 / avg_ms),
        "chunk_shape": tuple(chunk.shape),
        "select_action_shape": tuple(np.asarray(first_action).shape),
    }


def benchmark_with_fallback(**kwargs: Any) -> tuple[InferenceModel, dict[str, Any]]:
    device = kwargs["device"]
    try:
        return benchmark_pi05(**kwargs)
    except RuntimeError as exc:
        print(f"[WARN] Pi0.5 OpenVINO failed on {device}: {type(exc).__name__}: {exc}")
        if device == "CPU":
            raise
        print("[INFO] Falling back to CPU so the notebook can continue.")
        kwargs["device"] = "CPU"
        return benchmark_pi05(**kwargs)


def _so101_points(values: np.ndarray, origin: tuple[int, int] = (590, 330), scale: float = 1.0) -> list[tuple[int, int]]:
    vals = np.asarray(values, dtype=np.float32)
    lengths = np.array([70, 58, 48, 34], dtype=np.float32) * scale
    angles = np.deg2rad(
        [
            -90 + vals[0] * 0.55,
            vals[1] * 0.45,
            vals[2] * 0.35,
            vals[3] * 0.25 + vals[4] * 0.08,
        ]
    )
    pts = [np.array(origin, dtype=np.float32)]
    heading = 0.0
    for length, angle in zip(lengths, angles):
        heading += angle
        pts.append(pts[-1] + np.array([np.cos(heading), np.sin(heading)]) * length)
    return [(int(x), int(y)) for x, y in pts]


def _draw_arm(draw: ImageDraw.ImageDraw, values: np.ndarray, color: tuple[int, int, int], width: int = 9) -> None:
    pts = _so101_points(values)
    for a, b in zip(pts[:-1], pts[1:]):
        draw.line([a, b], fill=color, width=width)
        draw.line([a, b], fill=(245, 248, 250), width=max(2, width // 3))
    for p in pts:
        draw.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=(20, 24, 30), outline=color, width=3)
    gripper = float(np.asarray(values)[5])
    ee = pts[-1]
    span = int(8 + np.clip(gripper, 0, 100) * 0.12)
    draw.line([(ee[0] - span, ee[1] - 9), (ee[0] + span, ee[1] + 9)], fill=color, width=4)


def _draw_bars(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    pred: np.ndarray,
    expert: np.ndarray,
) -> None:
    row_h = height // len(SO101_JOINT_ORDER)
    center = x + width // 2
    draw.line([(center, y), (center, y + height)], fill=(80, 90, 102), width=1)
    for i, name in enumerate(SO101_JOINT_ORDER):
        yy = y + i * row_h + 6
        draw.text((x, yy), name, fill=(220, 226, 234))
        for val, color, offset in [(expert[i], (95, 170, 255), 13), (pred[i], (255, 190, 85), 28)]:
            v = float(np.clip(val, -100, 100))
            bar = int((v / 100.0) * (width * 0.32))
            draw.rectangle([min(center, center + bar), yy + offset, max(center, center + bar), yy + offset + 8], fill=color)


def _make_overlay_frame(
    top: np.ndarray,
    gripper: np.ndarray,
    state: np.ndarray,
    pred: np.ndarray,
    expert: np.ndarray,
    frame_idx: int,
    latency_ms: float,
    device: str,
    task: str,
) -> Image.Image:
    canvas = Image.new("RGB", (1120, 720), (18, 22, 28))
    top_img = Image.fromarray(top).resize((512, 288))
    grip_img = Image.fromarray(gripper).resize((512, 288))
    draw_top = ImageDraw.Draw(top_img)
    draw_grip = ImageDraw.Draw(grip_img)
    draw_top.rectangle([0, 0, 92, 26], fill=(0, 0, 0))
    draw_top.text((8, 6), "top-cam", fill=(255, 255, 255))
    draw_grip.rectangle([0, 0, 122, 26], fill=(0, 0, 0))
    draw_grip.text((8, 6), "gripper-cam", fill=(255, 255, 255))
    canvas.paste(top_img, (20, 72))
    canvas.paste(grip_img, (20, 374))

    draw = ImageDraw.Draw(canvas)
    draw.text((20, 24), "SO-101 Pi0.5 Replay: PhysicalAI + OpenVINO", fill=(242, 246, 250))
    draw.text((20, 48), f"task={task} | device={device} | frame={frame_idx:03d} | latency={latency_ms:.1f} ms", fill=(170, 184, 199))
    draw.rectangle([560, 72, 1098, 430], outline=(65, 76, 90), width=2)
    draw.text((580, 92), "Joint-space viewer", fill=(235, 241, 245))
    draw.text((580, 116), "blue: observed state   amber: Pi0.5 predicted target", fill=(170, 184, 199))
    _draw_arm(draw, state, (95, 170, 255), width=11)
    _draw_arm(draw, pred, (255, 190, 85), width=7)
    draw.text((580, 398), f"mean |pred - expert| = {float(np.mean(np.abs(pred - expert))):.2f}", fill=(235, 241, 245))
    draw.rectangle([560, 454, 1098, 704], outline=(65, 76, 90), width=2)
    draw.text((580, 468), "Action comparison in SO-101 normalized joint space", fill=(235, 241, 245))
    _draw_bars(draw, 580, 500, 490, 186, pred, expert)
    return canvas


def describe_replay_mae(mean_mae: float) -> str:
    if mean_mae < 5.0:
        return "low offline MAE on this replay dataset; replay domain looks consistent"
    if mean_mae < 15.0:
        return "moderate offline MAE; inspect camera/domain match in the overlay"
    return "high offline MAE; often caused by dataset, camera, task, or action-distribution mismatch"


def run_replay_visualization(
    *,
    model: InferenceModel,
    model_dir: Path,
    replay: ReplayEpisode,
    task: str,
    device: str,
    cache_dir: Path,
    output_dir: Path,
    max_rendered_frames: int = 120,
    render_stride: int = 3,
) -> dict[str, Any]:
    max_replay_steps = max_rendered_frames * render_stride
    top_frames = read_video_rgb(replay.top_video_path, max_frames=max_replay_steps, start_frame=replay.top_start_frame)
    gripper_frames = read_video_rgb(
        replay.gripper_video_path,
        max_frames=max_replay_steps,
        start_frame=replay.gripper_start_frame,
    )
    n = min(len(top_frames), len(gripper_frames), len(replay.episode_df), max_replay_steps)
    if n == 0:
        raise RuntimeError("No replay frames were decoded.")

    if model is None:
        model = InferenceModel.load(model_dir, backend="openvino", device=device, **openvino_config_for_device(cache_dir))
    model.reset()

    vis_frames: list[Image.Image] = []
    latencies: list[float] = []
    errors: list[float] = []
    predictions: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []

    for frame_idx in range(n):
        state = as_action_vector(replay.episode_df.iloc[frame_idx]["observation.state"], name="observation.state")
        expert = as_action_vector(replay.episode_df.iloc[frame_idx]["action"], name="expert action")
        obs = make_policy_observation(top_frames[frame_idx], gripper_frames[frame_idx], state, task)

        start = time.perf_counter()
        pred = as_action_vector(model.select_action(obs), name="predicted action")
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        errors.append(float(np.mean(np.abs(pred - expert))))
        predictions.append(pred)
        expert_actions.append(expert)
        if frame_idx % render_stride == 0 and len(vis_frames) < max_rendered_frames:
            vis_frames.append(
                _make_overlay_frame(
                    top_frames[frame_idx],
                    gripper_frames[frame_idx],
                    state,
                    pred,
                    expert,
                    frame_idx,
                    latency_ms,
                    device,
                    task,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "so101_pick_place_pi05_openvino.gif"
    vis_frames[0].save(gif_path, save_all=True, append_images=vis_frames[1:], duration=66, loop=0)
    mean_mae = float(np.mean(errors))
    per_joint_mae = np.mean(np.abs(np.stack(predictions) - np.stack(expert_actions)), axis=0)
    return {
        "gif_path": gif_path,
        "gif": IPyImage(filename=str(gif_path)),
        "steps": len(predictions),
        "rendered_frames": len(vis_frames),
        "avg_select_action_ms": float(np.mean(latencies)),
        "avg_mae": mean_mae,
        "per_joint_mae": dict(zip(SO101_JOINT_ORDER, per_joint_mae.round(3).tolist())),
        "interpretation": describe_replay_mae(mean_mae),
    }
