# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import logging
import os
import random
from dataclasses import dataclass
from itertools import islice
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, IterableDataset

from evals.action_anticipation_frozen.epickitchens import DataInfo, SharedEpoch, split_by_node
from src.datasets.utils.worker_init_fn import pl_worker_init_function


logger = logging.getLogger(__name__)


def _find_video_for_take(base_path: str, take_name: str, camera_glob: str) -> str:
    """Resolve a video file path for a given take.

    Search order:
      1) <take>/frame_aligned_videos/downscaled/448/<camera_glob>
      2) <take>/frame_aligned_videos/downscaled/448/cam01*.mp4
      3) Any mp4 under downscaled/448
    """
    cand_dir = os.path.join(base_path, take_name, "frame_aligned_videos", "downscaled", "448")
    if not os.path.exists(cand_dir):
        return None

    def pick(glob_pat: str) -> str:
        xs = sorted(glob.glob(os.path.join(cand_dir, glob_pat)))
        return xs[0] if len(xs) > 0 else None

    for pat in [camera_glob, "cam01*.mp4", "*.mp4"]:
        if pat is None:
            continue
        f = pick(pat)
        if f is not None:
            return f
    return None


def filter_annotations(
    base_path: str,
    train_annotations_path: str,
    val_annotations_path: str,
    camera_glob: str = "aria01*.mp4",
    **kwargs,
) -> Dict:
    """Build action class space and per-take segment annotations for KeyStep.

    Returns dict with keys compatible to downstream code:
      - actions: {unique_id -> class_idx}
      - val_actions: set of class indices present in val
      - train: (paths_list, {path -> pd.DataFrame(segments)})
      - val: (paths_list, {path -> pd.DataFrame(segments)})
    """

    def load_json(p):
        with open(p, "r") as f:
            return json.load(f)

    tr = load_json(train_annotations_path)
    va = load_json(val_annotations_path)

    # Collect unique leaf action ids from train (stable across files)
    def collect_unique_ids(obj) -> List[int]:
        segs = []
        for take in obj.get("annotations", {}).values():
            for s in take.get("segments", []):
                uid = s.get("step_unique_id")
                if uid is not None:
                    segs.append(int(uid))
        return segs

    train_unique = sorted(set(collect_unique_ids(tr)))
    # Use string keys to match downstream lookup in eval loop
    action_classes = {str(uid): idx for idx, uid in enumerate(train_unique)}

    def build_index(obj) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
        paths, annos = [], {}
        for take in obj.get("annotations", {}).values():
            take_name = take.get("take_name")
            vpath = _find_video_for_take(base_path, take_name, camera_glob)
            if vpath is None or (not os.path.exists(vpath)):
                logger.info(f"video not found for {take_name=}")
                continue
            segs = take.get("segments", [])
            if len(segs) == 0:
                continue
            # Build DataFrame with times and mapped action id (keep original uid for debug)
            rows = []
            for s in segs:
                uid = int(s.get("step_unique_id"))
                if str(uid) not in action_classes:
                    # Ignore actions unseen in train space
                    continue
                rows.append(
                    dict(
                        start_time=float(s.get("start_time")),
                        end_time=float(s.get("end_time")),
                        action_uid=uid,
                        action=int(action_classes[str(uid)]),
                    )
                )
            if len(rows) == 0:
                continue
            df = pd.DataFrame(rows).sort_values(by="start_time").reset_index(drop=True)
            paths.append(vpath)
            annos[vpath] = df
        return paths, annos

    train_paths, train_annos = build_index(tr)
    val_paths, val_annos = build_index(va)

    val_action_classes = set()
    for df in val_annos.values():
        val_action_classes |= set(df["action"].tolist())

    return dict(
        actions=action_classes,
        val_actions=val_action_classes,
        train=(train_paths, train_annos),
        val=(val_paths, val_annos),
    )


class decode_videos_to_clips(wds.PipelineStage):
    def __init__(
        self,
        annotations: Dict[str, pd.DataFrame],
        frames_per_clip: int = 16,
        fps: int = 4,
        transform=None,
        anticipation_time_sec=(0.5, 2.0),
        anticipation_point=(0.1, 0.1),
    ):
        self.annotations = annotations
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.anticipation_time = anticipation_time_sec
        self.anticipation_point = anticipation_point

    def run(self, src):
        for path in src:
            try:
                df = self.annotations[path]
            except KeyError:
                continue

            try:
                vr = VideoReader(path, num_threads=-1, ctx=cpu(0))
                vr.seek(0)
                vfps = max(1.0, float(vr.get_avg_fps()))
                fpc = int(self.frames_per_clip)
                fstp = int(np.ceil(vfps / float(self.fps)))
                nframes = int(fpc * fstp)
            except Exception as e:
                logger.info(f"Exception opening video {path}: {e}")
                continue

            for _, seg in df.iterrows():
                st, et = float(seg["start_time"]), float(seg["end_time"])
                a = random.uniform(*self.anticipation_time)
                ap = random.uniform(*self.anticipation_point)
                af = st * ap + (1.0 - ap) * et - a
                # Convert anchor time to frame index in video timeline
                aframes = int(af * vfps)
                indices = np.arange(aframes - nframes, aframes, fstp).astype(np.int64)
                if len(indices) == 0:
                    continue
                indices[indices < 0] = 0

                try:
                    buffer = vr.get_batch(indices).asnumpy()
                except Exception as e:
                    logger.info(f"Exception reading frames {path}: {e}")
                    continue

                if self.transform is not None:
                    buffer = self.transform(buffer)

                yield dict(
                    video=buffer,
                    action=int(seg["action"]),
                    anticipation_time=float(a),
                )


class ResampledShards(IterableDataset):
    def __init__(self, urls, epoch, training):
        super().__init__()
        self.epoch = epoch
        self.training = training
        self.urls = np.array(urls)

    def __iter__(self):
        if self.training:
            epoch = self.epoch.get_value()
            gen = torch.Generator()
            gen.manual_seed(epoch)
            yield from self.urls[torch.randperm(len(self.urls), generator=gen)]
        else:
            yield from self.urls[torch.arange(len(self.urls))]


def get_video_wds_dataset(
    batch_size,
    input_shards,
    video_decoder,
    training,
    epoch=0,
    world_size=1,
    rank=0,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
):
    epoch = SharedEpoch(epoch=epoch)
    pipeline = [
        ResampledShards(input_shards, epoch=epoch, training=training),
        split_by_node(rank=rank, world_size=world_size),
        wds.split_by_worker,
        video_decoder,
        wds.to_tuple("video", "action", "anticipation_time"),
        wds.batched(batch_size, partial=True, collation_fn=torch.utils.data.default_collate),
    ]
    dataset = wds.DataPipeline(*pipeline)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
        worker_init_fn=pl_worker_init_function,
        pin_memory=pin_memory,
    )

    return dataset, DataInfo(dataloader=dataloader, shared_epoch=epoch)


def make_webvid(
    base_path,
    annotations_path,
    batch_size,
    transform,
    frames_per_clip=16,
    fps=4,
    num_workers=8,
    world_size=1,
    rank=0,
    anticipation_time_sec=(0.5, 2.0),
    persistent_workers=True,
    pin_memory=True,
    training=True,
    anticipation_point=(0.1, 0.1),
    **kwargs,
):
    paths, annotations = annotations_path

    num_clips = sum([len(a) for a in annotations.values()])

    video_decoder = decode_videos_to_clips(
        annotations=annotations,
        frames_per_clip=frames_per_clip,
        fps=fps,
        transform=transform,
        anticipation_time_sec=anticipation_time_sec,
        anticipation_point=anticipation_point,
    )

    dataset, datainfo = get_video_wds_dataset(
        batch_size=batch_size,
        input_shards=paths,
        epoch=0,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        video_decoder=video_decoder,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        training=training,
    )

    # Estimate iterations per epa 
    datainfo.dataloader.num_batches = max(1, num_clips // max(1, (world_size * batch_size)))
    datainfo.dataloader.num_samples = num_clips

    return dataset, datainfo.dataloader, datainfo


