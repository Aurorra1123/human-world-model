# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation


_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=0,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
    condition_type="robot",
    video_roots: Optional[Sequence[str]] = None,
    heatmap_base_path: Optional[str] = None,
    heatmap_file_name: str = "gaze.npy",
    heatmap_camera_views: Optional[Sequence[str]] = None,
    heatmap_resolution_subdir: str = "frame_aligned_videos/downscaled/448",
    heatmap_cache: bool = True,
    heatmap_resize: Optional[Tuple[int, int]] = None,
    heatmap_path_format: Optional[str] = None,
    gazelle_cfg: Optional[dict] = None,
    shuffle: bool = True,
    deterministic_sampling: bool = False,
):
    condition_type = (condition_type or "robot").lower()
    if condition_type == "heatmap":
        dataset = GazeHeatmapVideoDataset(
            video_roots=video_roots if video_roots is not None else [data_path],
            heatmap_root=heatmap_base_path,
            frames_per_clip=frames_per_clip,
            fps=fps,
            transform=transform,
            resolution_subdir=heatmap_resolution_subdir,
            camera_filter=heatmap_camera_views,
            heatmap_file_name=heatmap_file_name,
            cache_heatmap=heatmap_cache,
            resize=heatmap_resize,
            heatmap_path_format=heatmap_path_format,
        )
    elif condition_type == "gazelle":
        dataset = GazelleSceneTokenDataset(
            video_roots=video_roots if video_roots is not None else [data_path],
            frames_per_clip=frames_per_clip,
            fps=fps,
            transform=transform,
            gazelle_cfg=gazelle_cfg or {},
            deterministic_sampling=deterministic_sampling,
        )
    else:
        dataset = DROIDVideoDataset(
            data_path=data_path,
            frames_per_clip=frames_per_clip,
            transform=transform,
            fps=fps,
            camera_views=camera_views,
            frameskip=tubelet_size,
            camera_frame=camera_frame,
        )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("VideoDataset unsupervised data loader created")

    return data_loader, dist_sampler


def get_json(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")


class DROIDVideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_path,
        camera_views=["left_mp4_path", "right_mp4_path"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Camera views
        # ---
        # wrist camera view
        # left camera view
        # right camera view
        self.camera_views = camera_views
        self.h5_name = "trajectory.h5"

        samples = list(pd.read_csv(data_path, header=None, delimiter=" ").values[:, 0])
        self.samples = samples

    def __getitem__(self, index):
        path = self.samples[index]

        # -- keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            try:
                buffer, actions, states, extrinsics, indices = self.loadvideo_decord(path)
                loaded_video = True
            except Exception as e:
                logger.info(f"Encountered exception when loading video {path=} {e=}")
                loaded_video = False
                index = np.random.randint(self.__len__())
                path = self.samples[index]

        return buffer, actions, states, extrinsics, indices

    def poses_to_diffs(self, poses):
        xyz = poses[:, :3]  # shape [T, 3]
        thetas = poses[:, 3:6]  # euler angles, shape [T, 3]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack([d for d in angle_diff], axis=0)
        closedness = poses[:, -1:]
        closedness_delta = closedness[1:] - closedness[:-1]
        return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

    def transform_frame(self, poses, extrinsics):
        gripper = poses[:, -1:]
        poses = poses[:, :-1]

        def pose_to_transform(pose):
            trans = pose[:3]  # shape [3]
            theta = pose[3:6]  # euler angles, shape [3]
            Rot = Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rot
            T[:3, 3] = trans
            return T

        def transform_to_pose(transform):
            trans = transform[:3, 3]
            Rot = transform[:3, :3]
            angle = Rotation.from_matrix(Rot).as_euler("xyz", degrees=False)
            return np.concatenate([trans, angle], axis=0)

        new_pose = []
        for p, e in zip(poses, extrinsics):
            p_transform = pose_to_transform(p)
            e_transform = pose_to_transform(e)
            new_pose_transform = np.linalg.inv(e_transform) @ p_transform
            new_pose += [transform_to_pose(new_pose_transform)]
        new_pose = np.stack(new_pose, axis=0)

        return np.concatenate([new_pose, gripper], axis=1)

    def loadvideo_decord(self, path):
        # -- load metadata
        metadata = get_json(path)
        if metadata is None:
            raise Exception(f"No metadata for video {path=}")

        # -- load trajectory info
        tpath = os.path.join(path, self.h5_name)
        trajectory = h5py.File(tpath)

        # -- randomly sample a camera view
        camera_view = self.camera_views[torch.randint(0, len(self.camera_views), (1,))]
        mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
        camera_name = mp4_name.split(".")[0]
        extrinsics = trajectory["observation"]["camera_extrinsics"][f"{camera_name}_left"]
        states = np.concatenate(
            [
                np.array(trajectory["observation"]["robot_state"]["cartesian_position"]),
                np.array(trajectory["observation"]["robot_state"]["gripper_position"])[:, None],
            ],
            axis=1,
        )  # [T, 7]
        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        # --
        vfps = vr.get_avg_fps()
        fpc = self.frames_per_clip
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        nframes = int(fpc * fstp)
        vlen = len(vr)

        if vlen < nframes:
            raise Exception(f"Video is too short {vpath=}, {nframes=}, {vlen=}")

        # sample a random window of nframes
        ef = np.random.randint(nframes, vlen)
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
        # --
        states = states[indices, :][:: self.frameskip]
        extrinsics = extrinsics[indices, :][:: self.frameskip]
        if self.camera_frame:
            states = self.transform_frame(states, extrinsics)
        actions = self.poses_to_diffs(states)
        # --
        vr.seek(0)  # go to start of video before sampling frames
        buffer = vr.get_batch(indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, extrinsics, indices

    def __len__(self):
        return len(self.samples)


class GazeHeatmapVideoDataset(torch.utils.data.Dataset):
    """
    Dataset reading ego-exo video clips with matching gaze heatmaps.
    Each item returns:
        - clip tensor or numpy array (depending on transform)
        - heatmap tensor [T, H, W]
        - sampled frame indices (torch.LongTensor)
        - string path to video
        - string path to heatmap
    """

    def __init__(
        self,
        video_roots: Sequence[str],
        heatmap_root: str,
        frames_per_clip: int = 16,
        fps: Optional[float] = None,
        transform=None,
        resolution_subdir: str = "frame_aligned_videos/downscaled/448",
        camera_filter: Optional[Sequence[str]] = None,
        heatmap_file_name: str = "gaze.npy",
        cache_heatmap: bool = True,
        resize: Optional[Tuple[int, int]] = None,
        heatmap_path_format: Optional[str] = None,
    ):
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        if not heatmap_root:
            raise ValueError("heatmap_root must be provided for heatmap-conditioned training.")

        self.video_roots = [Path(v).expanduser().resolve() for v in video_roots]
        self.heatmap_root = Path(heatmap_root).expanduser().resolve()
        self.frames_per_clip = int(frames_per_clip)
        self.target_fps = fps
        self.transform = transform
        self.resolution_subdir = Path(resolution_subdir)
        self.camera_filter = set(camera_filter) if camera_filter is not None else None
        self.heatmap_file_name = heatmap_file_name
        self.cache_heatmap = cache_heatmap
        self.resize = resize
        self.heatmap_path_format = heatmap_path_format

        if not self.heatmap_root.exists():
            raise FileNotFoundError(f"Heatmap root not found: {self.heatmap_root}")

        self.samples: List[Tuple[Path, Path]] = self._collect_samples()
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid video/heatmap pairs found. Checked video_roots={self.video_roots}, heatmap_root={self.heatmap_root}"
            )

        self._heatmap_cache: Dict[Path, np.ndarray] = {}

    def _collect_samples(self) -> List[Tuple[Path, Path]]:
        samples: List[Tuple[Path, Path]] = []
        for root in self.video_roots:
            if not root.exists():
                logger.warning("Video root does not exist: %s", root)
                continue

            if root.is_file() and root.suffix in {".csv", ".txt"}:
                try:
                    data = pd.read_csv(root, header=None, delimiter=" ")
                    candidate_paths = [Path(p).expanduser().resolve() for p in data.values[:, 0]]
                except Exception as exc:
                    logger.warning("Failed to parse video list %s (%s)", root, exc)
                    continue
            else:
                candidate_paths = []
                for take_dir in sorted(root.glob("*")):
                    if not take_dir.is_dir():
                        continue
                    video_dir = take_dir / self.resolution_subdir
                    if not video_dir.exists():
                        continue
                    candidate_paths.extend(sorted(video_dir.glob("cam*.mp4")))

            for video_path in candidate_paths:
                if not video_path.exists():
                    continue
                camera_name = video_path.stem
                if self.camera_filter is not None and camera_name not in self.camera_filter:
                    continue
                try:
                    take_name = video_path.parents[3].name
                except IndexError:
                    logger.debug("Skip video with unexpected structure: %s", video_path)
                    continue
                heatmap_path = self._resolve_heatmap_path(video_path, take_name, camera_name)
                if heatmap_path is None or not heatmap_path.exists():
                    logger.debug("Heatmap missing for %s -> %s", video_path, heatmap_path)
                    continue
                samples.append((video_path, heatmap_path))
        return samples

    def _resolve_heatmap_path(self, video_path: Path, take_name: str, camera_name: str) -> Optional[Path]:
        """Resolve heatmap path using optional template or default heuristics."""
        candidates: List[Path] = []
        if self.heatmap_path_format:
            format_kwargs = {
                "take_name": take_name,
                "camera_name": camera_name,
                "stem": video_path.stem,
            }
            for idx, parent in enumerate(video_path.parents):
                format_kwargs[f"parent{idx}"] = parent.name
            try:
                template_str = self.heatmap_path_format.format(**format_kwargs)
                template_path = Path(template_str)
                if not template_path.is_absolute():
                    template_path = self.heatmap_root / template_path
                if template_path.is_dir():
                    candidates.append(template_path / self.heatmap_file_name)
                else:
                    candidates.append(template_path)
            except KeyError as exc:
                logger.warning("heatmap_path_format missing key %s for %s", exc, video_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to apply heatmap_path_format for %s (%s)", video_path, exc)

        candidates.append(self.heatmap_root / take_name / self.heatmap_file_name)
        candidates.append(self.heatmap_root / f"{take_name}_{camera_name}" / self.heatmap_file_name)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_heatmap(self, heatmap_path: Path) -> np.ndarray:
        if self.cache_heatmap:
            cached = self._heatmap_cache.get(heatmap_path)
            if cached is None:
                cached = np.load(heatmap_path, mmap_mode="r")
                self._heatmap_cache[heatmap_path] = cached
            return cached
        return np.load(heatmap_path, allow_pickle=False)

    def __getitem__(self, index: int):
        video_path, heatmap_path = self.samples[index]

        try:
            vr = VideoReader(str(video_path), num_threads=-1, ctx=cpu(0))
        except Exception as exc:
            raise RuntimeError(f"Unable to open video {video_path}") from exc

        total_frames = len(vr)
        if total_frames <= 0:
            raise RuntimeError(f"Video {video_path} has no frames.")

        if self.target_fps is None:
            frame_step = 1
        else:
            try:
                native_fps = vr.get_avg_fps()
            except Exception:
                native_fps = self.target_fps
            frame_step = max(int(round(native_fps / self.target_fps)), 1)

        clip_span = self.frames_per_clip * frame_step
        if total_frames < clip_span:
            raise RuntimeError(
                f"Video {video_path} shorter ({total_frames}) than required clip span ({clip_span})."
            )

        end_idx = np.random.randint(clip_span, total_frames + 1)
        start_idx = end_idx - clip_span
        indices = np.arange(start_idx, end_idx, frame_step, dtype=np.int64)

        buffer = vr.get_batch(indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)

        heatmap_full = self._load_heatmap(heatmap_path)
        if heatmap_full.shape[0] < end_idx:
            raise RuntimeError(
                f"Heatmap {heatmap_path} shorter ({heatmap_full.shape[0]}) than requested end index ({end_idx})."
            )
        heatmap_clip = torch.tensor(heatmap_full[indices], dtype=torch.float32)

        if self.resize is not None:
            heatmap_clip = F.interpolate(
                heatmap_clip.unsqueeze(0),
                size=self.resize,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return (
            buffer,
            heatmap_clip,
            torch.from_numpy(indices),
            str(video_path),
            str(heatmap_path),
        )


class GazelleSceneTokenDataset(torch.utils.data.Dataset):
    """
    Dataset for loading video frames only (no Gazelle processing).
    Gazelle token extraction is handled in the model's forward pass.
    Each sample returns (clip, raw_frames, indices, video_path).
    """

    def __init__(
        self,
        video_roots: Sequence[str],
        frames_per_clip: int = 16,
        fps: Optional[float] = None,
        transform=None,
        gazelle_cfg: Optional[dict] = None,  # Kept for compatibility but not used
        deterministic_sampling: bool = False,  # Use fixed sampling for caching
    ):
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        self.video_paths: List[Path] = []
        for root in video_roots:
            candidate = Path(root).expanduser()
            if candidate.is_file() and candidate.suffix in {".txt", ".csv"}:
                with open(candidate, "r") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            self.video_paths.append(Path(line).expanduser())
            elif candidate.is_file() and candidate.suffix.lower() in {".mp4", ".avi", ".mov"}:
                self.video_paths.append(candidate)
            else:
                for video_file in sorted(candidate.rglob("*.mp4")):
                    self.video_paths.append(video_file)

        if len(self.video_paths) == 0:
            raise RuntimeError(f"No videos found under {video_roots}")

        self.frames_per_clip = int(frames_per_clip)
        self.target_fps = fps
        self.transform = transform
        self.deterministic_sampling = deterministic_sampling
        # Note: gazelle_cfg is ignored - Gazelle processing happens in model

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int):
        video_path = self.video_paths[index]
        try:
            vr = VideoReader(str(video_path), num_threads=-1, ctx=cpu(0))
        except Exception as exc:
            raise RuntimeError(f"Unable to open video {video_path}") from exc

        total_frames = len(vr)
        if total_frames <= 0:
            raise RuntimeError(f"Video {video_path} has no frames.")

        if self.target_fps is None:
            frame_step = 1
        else:
            try:
                native_fps = vr.get_avg_fps()
            except Exception:
                native_fps = self.target_fps
            frame_step = max(int(round(native_fps / self.target_fps)), 1)

        clip_span = self.frames_per_clip * frame_step
        if total_frames < clip_span:
            raise RuntimeError(f"Video {video_path} shorter ({total_frames}) than required clip span ({clip_span}).")

        if self.deterministic_sampling:
            # Use fixed sampling from middle of video for caching efficiency
            mid_frame = total_frames // 2
            start_idx = max(0, mid_frame - clip_span // 2)
            end_idx = start_idx + clip_span
            if end_idx > total_frames:
                end_idx = total_frames
                start_idx = end_idx - clip_span
        else:
            # Random sampling (original behavior)
            end_idx = np.random.randint(clip_span, total_frames + 1)
            start_idx = end_idx - clip_span
        indices = np.arange(start_idx, end_idx, frame_step, dtype=np.int64)

        frames = vr.get_batch(indices).asnumpy()
        raw_frames = torch.from_numpy(frames.copy()).to(dtype=torch.uint8)
        if self.transform is not None:
            frames = self.transform(frames)

        return (
            frames,
            raw_frames,
            torch.from_numpy(indices),
            str(video_path),
        )
