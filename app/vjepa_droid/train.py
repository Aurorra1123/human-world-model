# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None

from app.vjepa_droid.droid import init_data
from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_opt, init_video_model, load_checkpoint, load_pretrained
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def _maybe_cap_gazelle_workers(requested_workers, condition_mode, cfg_section, default_cap, cfg_path):
    """
    Gazelle conditioning spins up a full Dinov2/Gazelle backbone inside every worker process.
    On multi-GPU eval runs this quickly multiplies memory usage (num_ranks * num_workers copies),
    so we cap the worker count unless the config explicitly disables the cap (set to -1).
    """

    if condition_mode != "gazelle":
        return requested_workers

    cap = cfg_section.get("max_gazelle_workers")
    if cap is None:
        cap = default_cap

    if cap is None or cap < 0 or requested_workers <= cap:
        return requested_workers

    logger.warning(
        "%s requested %d workers for gazelle conditioning. "
        "Each worker loads a full Gazelle scene-token extractor, which is very memory-heavy. "
        "Reducing num_workers to %d. Set %s.max_gazelle_workers = -1 to keep the original value.",
        cfg_path,
        requested_workers,
        cap,
        cfg_path,
    )
    return cap


def _maybe_cap_gazelle_batch_size(gazelle_cfg, condition_mode, cfg_path, default_cap=4):
    """
    Gazelle token extraction runs a full Dinov2 per sample. Keep batch size modest to avoid OOM.
    """

    if condition_mode != "gazelle" or not gazelle_cfg:
        return gazelle_cfg

    cap = gazelle_cfg.get("max_batch_size")
    if cap is None:
        cap = default_cap
    if cap is None or cap < 0:
        return gazelle_cfg

    cfg_copy = dict(gazelle_cfg)
    current_bs = int(cfg_copy.get("batch_size", cap))
    if current_bs > cap:
        logger.warning(
            "%s.batch_size=%d is heavy for online Gazelle extraction. "
            "Reducing to %d (set %s.max_batch_size = -1 to disable).",
            cfg_path,
            current_bs,
            cap,
            cfg_path,
        )
        cfg_copy["batch_size"] = cap
    elif "batch_size" not in cfg_copy:
        cfg_copy["batch_size"] = cap
    return cfg_copy


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    load_predictor = cfgs_meta.get("load_predictor", False)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    eval_freq = cfgs_meta.get("eval_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    eval_only = bool(cfgs_meta.get("eval_only", False))
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    wandb_cfg = (cfgs_meta or {}).get("wandb", {})
    use_wandb = bool(wandb_cfg.get("enable", False))

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)
    condition_mode = cfgs_model.get("condition_mode", "robot")
    heatmap_model_cfg = cfgs_model.get("heatmap", {})
    action_embed_dim = cfgs_model.get("action_embed_dim", 7)
    freeze_encoder = cfgs_model.get("freeze_encoder", condition_mode in ("heatmap", "gazelle"))

    # -- DATA
    cfgs_data = args.get("data")
    datasets = cfgs_data.get("datasets", [])
    if isinstance(datasets, str):
        datasets = [datasets]
    if len(datasets) == 0:
        raise ValueError("At least one dataset path must be specified.")
    dataset_path = datasets[0]
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    condition_mode = cfgs_data.get("condition_mode", condition_mode)
    camera_frame = cfgs_data.get("camera_frame", False)
    camera_views = cfgs_data.get("camera_views", ["left_mp4_path"])
    stereo_view = cfgs_data.get("stereo_view", False)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    num_workers = _maybe_cap_gazelle_workers(
        num_workers,
        condition_mode,
        cfgs_data,
        default_cap=1 if condition_mode == "gazelle" else -1,
        cfg_path="data",
    )
    persistent_workers = cfgs_data.get("persistent_workers", True)
    if condition_mode == "gazelle" and num_workers <= 1:
        if persistent_workers:
            logger.warning("data.persistent_workers disabled because gazelle loader uses ≤1 worker.")
        persistent_workers = False
        if num_workers == 0:
            pin_mem = False
    heatmap_cfg = cfgs_data.get("heatmap", {})
    heatmap_base_path = heatmap_cfg.get("base_path", cfgs_data.get("heatmap_base_path"))
    heatmap_file_name = heatmap_cfg.get("file_name", "gaze.npy")
    heatmap_camera_views = heatmap_cfg.get("camera_views")
    heatmap_resolution_subdir = heatmap_cfg.get(
        "resolution_subdir", "frame_aligned_videos/downscaled/448"
    )
    heatmap_resize = heatmap_cfg.get("resize")
    if heatmap_resize is not None:
        heatmap_resize = tuple(heatmap_resize)
    elif condition_mode == "heatmap":
        heatmap_resize = (64, 64)
    heatmap_cache = heatmap_cfg.get("cache", True)
    heatmap_path_format = heatmap_cfg.get("path_format")
    gazelle_cfg = cfgs_data.get("gazelle", {})
    gazelle_cfg = _maybe_cap_gazelle_batch_size(
        gazelle_cfg,
        condition_mode,
        cfg_path="data.gazelle",
        default_cap=4,
    )
    
    raw_val_cfg = cfgs_data.get("val")
    if raw_val_cfg is None:
        val_cfg = {}
    elif isinstance(raw_val_cfg, dict):
        val_cfg = raw_val_cfg
    else:
        val_cfg = {"datasets": raw_val_cfg}

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    normalize_reps = cfgs_loss.get("normalize_reps")
    auto_steps = min(cfgs_loss.get("auto_steps", 1), max_num_frames)
    gaze_loss_weight = float(cfgs_loss.get("gaze_weight", 0.0))
    # --
    tokens_per_frame = int((crop_size // patch_size) ** 2)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    resume_path = None
    resume_candidates = []
    if r_file:
        if os.path.isabs(r_file):
            resume_candidates.append(r_file)
        else:
            resume_candidates.extend([r_file, os.path.join(folder, r_file)])
    else:
        resume_candidates.append(latest_path)

    for resume_candidate in resume_candidates:
        if resume_candidate and os.path.exists(resume_candidate):
            resume_path = resume_candidate
            break

    if r_file and resume_path is None:
        logger.warning(
            "Resume checkpoint %s not found. Proceeding without resuming.",
            resume_candidates[-1],
        )

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        mode="+a",
    )

    # -- init model
    if condition_mode == "heatmap":
        heatmap_model_cfg = dict(heatmap_model_cfg)
        heatmap_model_cfg.setdefault("enable_aux_loss", gaze_loss_weight > 0)

    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        condition_type=condition_mode,
        heatmap_kwargs=heatmap_model_cfg if condition_mode in ("heatmap", "gazelle") else None,
        freeze_encoder=freeze_encoder,
        gazelle_cfg=gazelle_cfg if condition_mode == "gazelle" else None,
    )
    target_encoder = copy.deepcopy(encoder)
    target_encoder.eval()
    for param in target_encoder.parameters():
        param.requires_grad = False

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    video_collator = torch.utils.data.default_collate
    transform = make_transforms(
        random_horizontal_flip=horizontal_flip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    val_transform = transform
    val_data_aug_cfg = val_cfg.get("data_aug") if isinstance(val_cfg, dict) else None
    if val_data_aug_cfg:
        val_transform = make_transforms(
            random_horizontal_flip=val_data_aug_cfg.get("horizontal_flip", False),
            random_resize_aspect_ratio=tuple(val_data_aug_cfg.get("random_resize_aspect_ratio", ar_range)),
            random_resize_scale=tuple(val_data_aug_cfg.get("random_resize_scale", rr_scale)),
            reprob=val_data_aug_cfg.get("reprob", 0.0),
            auto_augment=val_data_aug_cfg.get("auto_augment", False),
            motion_shift=val_data_aug_cfg.get("motion_shift", False),
            crop_size=val_data_aug_cfg.get("crop_size", crop_size),
        )

    if condition_mode == "heatmap" and heatmap_base_path is None:
        raise ValueError("heatmap_base_path must be specified for heatmap-conditioned training.")
    loader_tubelet = 1 if condition_mode == "robot" else 1
    video_roots = datasets if condition_mode in ("heatmap", "gazelle") else None

    # -- init data-loaders/samplers
    unsup_loader_kwargs = None
    unsupervised_loader = None
    unsupervised_sampler = None
    loader = None
    _dlen = 0
    if not eval_only:
        unsup_loader_kwargs = dict(
            data_path=dataset_path,
            batch_size=batch_size,
            frames_per_clip=max_num_frames,
            tubelet_size=loader_tubelet,
            fps=fps,
            camera_views=camera_views,
            camera_frame=camera_frame,
            stereo_view=stereo_view,
            transform=transform,
            collator=video_collator,
            num_workers=num_workers,
            world_size=world_size,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            rank=rank,
            condition_type=condition_mode,
            video_roots=video_roots,
            heatmap_base_path=heatmap_base_path,
            heatmap_file_name=heatmap_file_name,
            heatmap_camera_views=heatmap_camera_views,
            heatmap_resolution_subdir=heatmap_resolution_subdir,
            heatmap_cache=heatmap_cache,
            heatmap_resize=heatmap_resize,
            heatmap_path_format=heatmap_path_format,
            gazelle_cfg=gazelle_cfg if condition_mode == "gazelle" else None,
            deterministic_sampling=cfgs_data.get("deterministic_sampling", False),
        )
        (unsupervised_loader, unsupervised_sampler) = init_data(**unsup_loader_kwargs)
        _dlen = len(unsupervised_loader)
        if ipe is None:
            ipe = _dlen
        logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")
    else:
        logger.info("eval_only=True → skipping training DataLoader initialization to save memory.")

    val_loader = None
    val_sampler = None
    val_loader_kwargs = None
    val_condition_mode = val_cfg.get("condition_mode", condition_mode) if isinstance(val_cfg, dict) else condition_mode
    val_dataset_paths = []
    if isinstance(val_cfg, dict):
        val_dataset_paths = (
            val_cfg.get("datasets")
            or val_cfg.get("dataset")
            or val_cfg.get("dataset_val")
            or val_cfg.get("dataset_path")
        )
    if isinstance(val_dataset_paths, str):
        val_dataset_paths = [val_dataset_paths]

    if val_dataset_paths:
        val_dataset_path = val_dataset_paths[0]
        val_video_roots = val_dataset_paths if val_condition_mode in ("heatmap", "gazelle") else None
        val_batch_size = val_cfg.get("batch_size", batch_size)
        val_num_workers = val_cfg.get("num_workers", num_workers)
        val_num_workers = _maybe_cap_gazelle_workers(
            val_num_workers,
            val_condition_mode,
            val_cfg,
            default_cap=1 if val_condition_mode == "gazelle" else -1,
            cfg_path="data.val",
        )
        val_pin_mem = val_cfg.get("pin_mem", pin_mem)
        if val_num_workers == 0:
            val_pin_mem = False
        val_persistent_workers = val_cfg.get("persistent_workers", False) and val_num_workers > 0
        if val_condition_mode == "gazelle" and val_num_workers <= 1 and val_persistent_workers:
            logger.warning(
                "data.val.persistent_workers disabled because gazelle validation loader uses ≤1 worker."
            )
            val_persistent_workers = False
        val_fps = val_cfg.get("fps", fps)
        val_tubelet_size = val_cfg.get("tubelet_size", tubelet_size)
        val_camera_views = val_cfg.get("camera_views", camera_views)
        val_camera_frame = val_cfg.get("camera_frame", camera_frame)
        val_stereo_view = val_cfg.get("stereo_view", stereo_view)
        val_dataset_fpcs = val_cfg.get("dataset_fpcs")
        val_frames_per_clip = max(val_dataset_fpcs) if val_dataset_fpcs else max_num_frames
        val_heatmap_cfg = val_cfg.get("heatmap", {})
        val_heatmap_base_path = val_heatmap_cfg.get("base_path", heatmap_base_path)
        val_heatmap_file_name = val_heatmap_cfg.get("file_name", heatmap_file_name)
        val_heatmap_camera_views = val_heatmap_cfg.get("camera_views", heatmap_camera_views)
        val_heatmap_resolution_subdir = val_heatmap_cfg.get(
            "resolution_subdir", heatmap_resolution_subdir
        )
        val_heatmap_cache = val_heatmap_cfg.get("cache", heatmap_cache)
        val_heatmap_resize = val_heatmap_cfg.get("resize", heatmap_resize)
        if val_heatmap_resize is not None:
            val_heatmap_resize = tuple(val_heatmap_resize)
        elif val_condition_mode == "heatmap":
            val_heatmap_resize = (64, 64)
        val_heatmap_path_format = val_heatmap_cfg.get("path_format", heatmap_path_format)
        val_gazelle_cfg = val_cfg.get("gazelle", gazelle_cfg)
        val_gazelle_cfg = _maybe_cap_gazelle_batch_size(
            val_gazelle_cfg,
            val_condition_mode,
            cfg_path="data.val.gazelle",
            default_cap=4,
        )

        val_loader_kwargs = dict(
            data_path=val_dataset_path,
            batch_size=val_batch_size,
            frames_per_clip=val_frames_per_clip,
            tubelet_size=val_tubelet_size,
            fps=val_fps,
            camera_views=val_camera_views,
            stereo_view=val_stereo_view,
            transform=val_transform,
            collator=video_collator,
            num_workers=val_num_workers,
            pin_mem=val_pin_mem,
            persistent_workers=val_persistent_workers,
            drop_last=False,
            world_size=world_size,
            rank=rank,
            camera_frame=val_camera_frame,
            condition_type=val_condition_mode,
            video_roots=val_video_roots,
            heatmap_base_path=val_heatmap_base_path,
            heatmap_file_name=val_heatmap_file_name,
            heatmap_camera_views=val_heatmap_camera_views,
            heatmap_resolution_subdir=val_heatmap_resolution_subdir,
            heatmap_cache=val_heatmap_cache,
            heatmap_resize=val_heatmap_resize,
            heatmap_path_format=val_heatmap_path_format,
            gazelle_cfg=val_gazelle_cfg if val_condition_mode == "gazelle" else None,
            shuffle=False,
            deterministic_sampling=val_cfg.get("deterministic_sampling", cfgs_data.get("deterministic_sampling", False)),
        )
        val_loader, val_sampler = init_data(**val_loader_kwargs)
        val_ipe = len(val_loader)
        logger.info(f"Validation loader initialized (len={val_ipe})")
        if ipe is None:
            ipe = val_ipe
    elif eval_freq and eval_freq > 0:
        logger.warning("eval_freq > 0 but data.val not provided; skipping validation runs.")

    wandb_run = None
    if use_wandb and rank == 0:
        if wandb is None:
            raise ImportError(
                "wandb 未安装，但 meta.wandb.enable 为 True。请先 `pip install wandb` 或在配置中关闭。"
            )
        wandb_kwargs = dict(
            project=wandb_cfg.get("project"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name"),
            group=wandb_cfg.get("group"),
            notes=wandb_cfg.get("notes"),
            config=args,
        )
        tags = wandb_cfg.get("tags")
        if isinstance(tags, str):
            tags = [tags]
        if tags:
            wandb_kwargs["tags"] = tags
        if wandb_cfg.get("id"):
            wandb_kwargs["id"] = wandb_cfg["id"]
        if wandb_cfg.get("resume", False):
            wandb_kwargs["resume"] = "allow"
        wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
        wandb_run = wandb.init(**wandb_kwargs)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
        freeze_encoder=freeze_encoder,
    )
    def _has_trainable_params(module):
        if module is None:
            return False
        return any(p.requires_grad for p in module.parameters())

    if world_size > 1:
        if _has_trainable_params(encoder):
            encoder = DistributedDataParallel(encoder, static_graph=True)
        else:
            encoder = encoder.to(device)

        if _has_trainable_params(predictor):
            predictor = DistributedDataParallel(
                predictor, static_graph=False, find_unused_parameters=True
            )
        else:
            predictor = predictor.to(device)

        if _has_trainable_params(target_encoder):
            target_encoder = DistributedDataParallel(target_encoder)
        else:
            target_encoder = target_encoder.to(device)
    else:
        encoder = encoder.to(device)
        predictor = predictor.to(device)
        target_encoder = target_encoder.to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- load pretrained weights (optional)
    if p_file:
        (
            encoder,
            predictor,
            target_encoder,
        ) = load_pretrained(
            r_path=p_file,
            encoder=encoder,
            predictor=predictor,
            context_encoder_key=context_encoder_key,
            target_encoder_key=target_encoder_key,
            target_encoder=target_encoder,
            load_predictor=load_predictor,
            load_encoder=load_encoder,
        )
    else:
        logger.info("No pretrain checkpoint provided; skipping load_pretrained.")

    start_epoch = 0
    if resume_path is not None:
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=resume_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
    elif eval_only:
        raise ValueError(
            "eval_only=True 但未找到可用的 resume checkpoint（请提供 --resume_checkpoint 或确保 latest.pt 存在）"
        )
    else:
        logger.info("No checkpoint loaded; starting from scratch.")

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    if unsupervised_sampler is not None:
        logger.info("Initializing loader...")
        unsupervised_sampler.set_epoch(start_epoch)
        loader = iter(unsupervised_loader)

    def _rebuild_unsupervised_loader_for_failure(current_epoch: int) -> bool:
        nonlocal unsupervised_loader, unsupervised_sampler, loader, unsup_loader_kwargs
        if not unsup_loader_kwargs or unsup_loader_kwargs.get("num_workers", 0) == 0:
            return False
        logger.warning(
            "Training DataLoader worker failed; switching to single-threaded loader (num_workers=0)."
        )
        fallback_kwargs = dict(unsup_loader_kwargs)
        fallback_kwargs.update(
            dict(
                num_workers=0,
                pin_mem=False,
                persistent_workers=False,
            )
        )
        unsup_loader_kwargs = fallback_kwargs
        unsupervised_loader, unsupervised_sampler = init_data(**unsup_loader_kwargs)
        unsupervised_sampler.set_epoch(current_epoch)
        loader = iter(unsupervised_loader)
        return True

    if skip_batches > 0 and loader is not None:
        logger.info(f"Skip {skip_batches} batches")
        # -- update distributed-data-loader epoch

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception as exc:
                err_str = str(exc)
                worker_failed = "DataLoader worker" in err_str and "killed" in err_str.lower()
                rebuilt = worker_failed and _rebuild_unsupervised_loader_for_failure(start_epoch)
                if not rebuilt:
                    loader = iter(unsupervised_loader)
                    _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    def loss_fn(pred_tokens, target_tokens):
        return torch.mean(torch.abs(pred_tokens - target_tokens) ** loss_exp) / loss_exp

    def get_predictor_module():
        if hasattr(predictor, "module"):
            return predictor.module
        return predictor

    def downsample_heatmap_for_loss(heatmap_tensor, target_h, target_w):
        if heatmap_tensor.dim() == 4:
            heatmap_tensor = heatmap_tensor.unsqueeze(2)
        B, T = heatmap_tensor.shape[:2]
        pooled = F.adaptive_avg_pool2d(
            heatmap_tensor.reshape(B * T, 1, heatmap_tensor.size(-2), heatmap_tensor.size(-1)),
            (target_h, target_w),
        )
        return pooled.view(B, T, target_h * target_w)

    def prepare_batch(sample, mode):
        if mode in ("heatmap", "gazelle"):
            clips = sample[0].to(device, non_blocking=True)
            batch = dict(
                clips=clips,
                condition_mode=mode,
                actions=None,
                states=None,
                extrinsics=None,
            )
            if mode == "heatmap":
                condition_tensor = sample[1].to(device, dtype=torch.float32, non_blocking=True)
                batch["condition"] = condition_tensor
                batch["heatmap"] = condition_tensor
            elif mode == "gazelle":
                batch["gazelle_raw_frames"] = sample[1]
                # Add cache keys for Gazelle token caching
                batch["frame_indices"] = sample[2]  # Frame indices
                batch["video_paths"] = sample[3]     # Video paths
            return batch
        clips = sample[0].to(device, non_blocking=True)
        actions = sample[1].to(device, dtype=torch.float32, non_blocking=True)
        states = sample[2].to(device, dtype=torch.float32, non_blocking=True)
        extrinsics = sample[3].to(device, dtype=torch.float32, non_blocking=True)
        return dict(clips=clips, condition=None, actions=actions, states=states, extrinsics=extrinsics)

    def forward_target_tokens(clips_tensor):
        bsz = clips_tensor.size(0)
        num_frames = clips_tensor.size(2)
        with torch.no_grad():
            ctx = clips_tensor.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            tokens = target_encoder(ctx)
            tokens = tokens.view(bsz, num_frames, -1, tokens.size(-1)).flatten(1, 2)
            if normalize_reps:
                tokens = F.layer_norm(tokens, (tokens.size(-1),))
            return tokens

    def forward_predictions(z, actions, states, extrinsics):
        def _step_predictor(_z, _a, _s, _e):
            _z = predictor(_z, _a, _s, _e)
            if normalize_reps:
                _z = F.layer_norm(_z, (_z.size(-1),))
            return _z

        base_tokens = z[:, :-tokens_per_frame]
        z_tf = _step_predictor(base_tokens, actions, states[:, :-1], extrinsics[:, :-1])
        combined = torch.cat([z[:, : tokens_per_frame], z_tf[:, : tokens_per_frame]], dim=1)
        for n in range(1, auto_steps):
            step_actions = actions[:, : n + 1]
            step_states = states[:, : n + 1]
            step_extrinsics = extrinsics[:, : n + 1]
            step_pred = _step_predictor(combined, step_actions, step_states, step_extrinsics)
            step_pred = step_pred[:, -tokens_per_frame:]
            combined = torch.cat([combined, step_pred], dim=1)
        z_ar = combined[:, tokens_per_frame:]
        return z_tf, z_ar

    def compute_losses(batch, mode):
        clips = batch["clips"]
        targets = forward_target_tokens(clips)
        if mode in ("heatmap", "gazelle"):
            context_tokens = targets[:, :-tokens_per_frame]
            target_tokens = targets[:, tokens_per_frame:]
            condition_tensor = batch.get("condition")
            condition_context = None
            if condition_tensor is not None:
                condition_context = condition_tensor[:, 1:]
            gazelle_frames = batch.get("gazelle_raw_frames")
            video_paths = batch.get("video_paths")
            frame_indices = batch.get("frame_indices")
            if gazelle_frames is not None:
                gazelle_frames = gazelle_frames[:, 1:]
            if frame_indices is not None:
                # Adjust indices to skip first frame (context frames only)
                frame_indices = [idx[1:] if isinstance(idx, torch.Tensor) else idx for idx in frame_indices]
            predictor_mod = get_predictor_module()
            grid_h = getattr(predictor_mod, "grid_height", crop_size // patch_size)
            grid_w = getattr(predictor_mod, "grid_width", crop_size // patch_size)
            supports_aux = getattr(predictor_mod, "supports_gaze_aux", False)
            cond_mode = batch.get("condition_mode", mode)
            preds_out = (
                predictor(
                    context_tokens,
                    condition=condition_context,
                    condition_mode=cond_mode,
                    gazelle_frames=gazelle_frames,
                    video_paths=video_paths,
                    frame_indices=frame_indices,
                    return_gaze=True,
                )
                if gaze_loss_weight > 0 and supports_aux and cond_mode == "heatmap"
                else predictor(
                    context_tokens,
                    condition=condition_context,
                    condition_mode=cond_mode,
                    gazelle_frames=gazelle_frames,
                    video_paths=video_paths,
                    frame_indices=frame_indices,
                )
            )
            if isinstance(preds_out, tuple):
                preds, gaze_pred = preds_out
            else:
                preds = preds_out
                gaze_pred = None
            if normalize_reps:
                preds = F.layer_norm(preds, (preds.size(-1),))
                target_tokens = F.layer_norm(target_tokens, (target_tokens.size(-1),))
            loss = loss_fn(preds, target_tokens)
            if gaze_pred is not None and gaze_loss_weight > 0 and cond_mode == "heatmap":
                gaze_targets = downsample_heatmap_for_loss(condition_context, grid_h, grid_w)
                gaze_pred = gaze_pred.view_as(gaze_targets)
                gaze_loss = F.mse_loss(gaze_pred, gaze_targets)
                loss = loss + gaze_loss_weight * gaze_loss
            else:
                gaze_loss = loss.new_zeros(())
            return loss, loss, gaze_loss

        actions = batch["actions"]
        states = batch["states"]
        extrinsics = batch["extrinsics"]
        if actions is None or states is None or extrinsics is None:
            raise ValueError("robot batch missing actions/states/extrinsics tensors")
        z_tf, z_ar = forward_predictions(targets, actions=actions, states=states, extrinsics=extrinsics)
        tf_targets = targets[:, tokens_per_frame : tokens_per_frame + z_tf.size(1)]
        ar_targets = targets[:, tokens_per_frame : tokens_per_frame + z_ar.size(1)]
        jloss = loss_fn(z_tf, tf_targets)
        sloss = loss_fn(z_ar, ar_targets)
        return jloss + sloss, jloss, sloss

    def _rebuild_val_loader_for_failure() -> bool:
        nonlocal val_loader, val_sampler, val_loader_kwargs
        if val_loader_kwargs is None:
            return False
        current_workers = int(val_loader_kwargs.get("num_workers", 0))
        if current_workers == 0:
            return False
        next_workers = 0 if current_workers == 1 else max(current_workers // 2, 1)
        logger.warning(
            "Validation DataLoader worker failed; reducing num_workers from %d to %d and retrying.",
            current_workers,
            next_workers,
        )
        fallback_kwargs = dict(val_loader_kwargs)
        fallback_kwargs.update(
            dict(
                num_workers=next_workers,
                pin_mem=fallback_kwargs.get("pin_mem", False) and next_workers > 0,
                persistent_workers=False,
            )
        )
        val_loader_kwargs = fallback_kwargs
        val_loader, val_sampler = init_data(**val_loader_kwargs)
        return True

    def run_validation(epoch_idx):
        nonlocal val_loader, val_sampler, val_loader_kwargs
        if val_loader is None:
            return
        if val_sampler is not None:
            val_sampler.set_epoch(epoch_idx)

        prev_encoder_mode = encoder.training
        prev_predictor_mode = predictor.training
        prev_target_mode = target_encoder.training
        encoder.train(False)
        predictor.train(False)
        target_encoder.train(False)

        val_loss_meter = AverageMeter()
        val_joint_meter = AverageMeter()
        val_seq_meter = AverageMeter()
        val_iter_time_meter = AverageMeter()
        val_gpu_time_meter = AverageMeter()
        val_data_time_meter = AverageMeter()

        try:
            for v_itr, sample in enumerate(val_loader):
                itr_start_time = time.time()
                batch = prepare_batch(sample, val_condition_mode)
                data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

                def val_step():
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                            loss_t, jloss_t, sloss_t = compute_losses(batch, val_condition_mode)
                    return dict(
                        val_loss=float(loss_t),
                        val_joint_loss=float(jloss_t),
                        val_seq_loss=float(sloss_t),
                    )

                metrics, gpu_etime_ms = gpu_timer(val_step)
                iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

                val_loss_meter.update(metrics["val_loss"])
                val_joint_meter.update(metrics["val_joint_loss"])
                val_seq_meter.update(metrics["val_seq_loss"])
                val_iter_time_meter.update(iter_elapsed_time_ms)
                val_gpu_time_meter.update(gpu_etime_ms)
                val_data_time_meter.update(data_elapsed_time_ms)
        except RuntimeError as exc:
            err_str = str(exc)
            if "DataLoader worker" in err_str and "killed" in err_str.lower():
                if _rebuild_val_loader_for_failure():
                    return run_validation(epoch_idx)
            raise

        logger.info(
            "[val][%d] loss: %.3f [%.2f, %.2f] [iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
            % (
                epoch_idx + 1,
                val_loss_meter.avg,
                val_joint_meter.avg,
                val_seq_meter.avg,
                val_iter_time_meter.avg,
                val_gpu_time_meter.avg,
                val_data_time_meter.avg,
            )
        )

        if use_wandb and rank == 0:
            wandb.log(
                dict(
                    val_loss=val_loss_meter.avg,
                    val_joint_loss=val_joint_meter.avg,
                    val_seq_loss=val_seq_meter.avg,
                    val_iter_ms=val_iter_time_meter.avg,
                    val_gpu_ms=val_gpu_time_meter.avg,
                    val_data_ms=val_data_time_meter.avg,
                    epoch=epoch_idx + 1,
                ),
                step=(epoch_idx + 1) * ipe,
            )

        encoder.train(prev_encoder_mode)
        predictor.train(prev_predictor_mode)
        target_encoder.train(prev_target_mode)

    if eval_only:
        if val_loader is None:
            raise ValueError("eval_only=True but no validation dataset configured under data.val")
        logger.info("eval_only=True → skipping training loop and running validation once.")
        run_validation(max(start_epoch - 1, 0))
        if use_wandb and rank == 0 and wandb_run is not None:
            wandb.finish()
        return

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        jloss_meter = AverageMeter()
        sloss_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    err_str = str(e)
                    worker_failed = "DataLoader worker" in err_str and "killed" in err_str.lower()
                    if worker_failed and _rebuild_unsupervised_loader_for_failure(epoch):
                        iter_retries = 0
                        continue
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            batch = prepare_batch(sample, condition_mode)
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                with torch.amp.autocast(
                    device_type="cuda", dtype=dtype, enabled=mixed_precision
                ):
                    loss_t, jloss_t, sloss_t = compute_losses(batch, condition_mode)

                if mixed_precision:
                    scaler.scale(loss_t).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss_t.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                return dict(
                    train_loss=float(loss_t),
                    train_joint_loss=float(jloss_t),
                    train_seq_loss=float(sloss_t),
                    lr=float(_new_lr),
                    weight_decay=float(_new_wd),
                )

            metrics, gpu_etime_ms = gpu_timer(train_step)
            loss = metrics["train_loss"]
            jloss = metrics["train_joint_loss"]
            sloss = metrics["train_seq_loss"]
            _new_lr = metrics["lr"]
            _new_wd = metrics["weight_decay"]
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                should_log = (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss)
                if should_log:
                    logger.info(
                        "[%d, %5d] loss: %.3f [%.2f, %.2f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            jloss_meter.avg,
                            sloss_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

                if use_wandb and rank == 0:
                    wandb.log(
                        dict(
                            train_loss=loss_meter.avg,
                            train_joint_loss=jloss_meter.avg,
                            train_seq_loss=sloss_meter.avg,
                            train_lr=_new_lr,
                            train_weight_decay=_new_wd,
                            train_iter_ms=iter_time_meter.avg,
                            train_gpu_ms=gpu_time_meter.avg,
                            train_data_ms=data_elapsed_time_meter.avg,
                            epoch=epoch + 1,
                        ),
                        step=epoch * ipe + itr,
                    )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)

        if eval_freq and eval_freq > 0:
            if ((epoch + 1) % eval_freq == 0) or (epoch == num_epochs - 1):
                run_validation(epoch)

        # -- Save Last
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)

    if use_wandb and rank == 0 and wandb_run is not None:
        wandb.finish()
