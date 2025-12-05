import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from app.vjepa_droid.gazelle_tokens import get_gazelle_extractor
from src.models.utils.modules import ACBlock as Block
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_


class VisionTransformerPredictorGazeAC(nn.Module):
    """
    Vision Transformer predictor that injects gaze-conditioned tokens (heatmaps or Gazelle scene tokens).
    """

    supports_gaze_aux = True

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        heatmap_patch_size=16,
        scene_token_dim=256,
        scene_token_grid=32,
        enable_gaze_aux=False,
        gazelle_cfg=None,
        **_,
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.use_activation_checkpointing = use_activation_checkpointing
        self.enable_gaze_aux = enable_gaze_aux
        self.supports_gaze_aux = enable_gaze_aux

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.grid_height = self.img_height // self.patch_size
        self.grid_width = self.img_width // self.patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.heatmap_pool = nn.AdaptiveAvgPool2d((self.grid_height, self.grid_width))
        self.heatmap_proj = nn.Linear(1, predictor_embed_dim)
        self.scene_grid_size = scene_token_grid
        self.scene_proj = nn.Linear(scene_token_dim, predictor_embed_dim)
        if self.enable_gaze_aux:
            self.gaze_decoder = nn.Linear(predictor_embed_dim, 1)
        self.gazelle_cfg = gazelle_cfg or {}
        self._gazelle_extractor = None
        if self.gazelle_cfg.get("checkpoint"):
            self._init_gazelle_extractor()

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        self._attn_mask_cache = {}
        self._gazelle_token_cache = {}  # Cache for scene tokens: {cache_key: tokens}
        self._cache_enabled = True  # Enable/disable caching

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @torch.no_grad()
    def downsample_heatmap(self, heatmap):
        """
        Down-sample raw heatmap to patch grid.
        Returns tensor of shape [B, T, grid_h, grid_w].
        """
        if heatmap.dim() == 4:
            heatmap = heatmap.unsqueeze(2)
        B, T = heatmap.shape[:2]
        pooled = self.heatmap_pool(heatmap.reshape(B * T, 1, heatmap.size(-2), heatmap.size(-1)))
        return pooled.view(B, T, self.grid_height, self.grid_width)

    def _prepare_heatmap_tokens(self, heatmap):
        pooled = self.downsample_heatmap(heatmap)
        B, T = pooled.shape[:2]
        tokens = pooled.view(B, T, self.grid_height * self.grid_width, 1)
        return self.heatmap_proj(tokens)

    def _prepare_scene_tokens(self, scene_tokens):
        B, T, HW, C = scene_tokens.shape
        grid_src = int(math.sqrt(HW))
        if grid_src * grid_src != HW:
            raise ValueError(f"Scene tokens must form a square grid, got {HW}.")
        tokens = scene_tokens.reshape(B * T, grid_src, grid_src, C).permute(0, 3, 1, 2)
        tokens = F.interpolate(
            tokens,
            size=(self.grid_height, self.grid_width),
            mode="bilinear",
            align_corners=False,
        )
        tokens = tokens.permute(0, 2, 3, 1).reshape(
            B, T, self.grid_height * self.grid_width, C
        )
        return self.scene_proj(tokens)

    def _prepare_condition_tokens(self, condition, mode: str):
        mode = (mode or "heatmap").lower()
        if mode == "heatmap":
            return self._prepare_heatmap_tokens(condition)
        if mode == "gazelle":
            return self._prepare_scene_tokens(condition)
        raise ValueError(f"Unsupported condition mode: {mode}")

    def _init_gazelle_extractor(self):
        if self._gazelle_extractor is not None:
            return
        if not self.gazelle_cfg or "checkpoint" not in self.gazelle_cfg:
            raise ValueError("gazelle_cfg with valid checkpoint is required for gazelle conditioning.")
        cfg = self.gazelle_cfg
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[GazellePredictor] Initializing Gazelle extractor with device={cfg.get('device', 'cpu')}")
        self._gazelle_extractor = get_gazelle_extractor(
            checkpoint=cfg["checkpoint"],
            model_name=cfg.get("model_name", "gazelle_dinov2_vitb14_inout"),
            device=cfg.get("device", "cpu"),
            max_batch_size=int(cfg.get("batch_size", 8)),
            module_path=cfg.get("python_path") or cfg.get("module_path"),
            min_cuda_free_mb=int(cfg.get("min_cuda_free_mb", 2048)),
        )
        # Force lazy initialization by accessing properties
        _ = self._gazelle_extractor.scene_dim
        _ = self._gazelle_extractor.scene_grid
        logger.info(f"[GazellePredictor] Gazelle extractor initialized successfully (dim={self._gazelle_extractor.scene_dim}, grid={self._gazelle_extractor.scene_grid})")

    def _get_gazelle_extractor(self):
        self._init_gazelle_extractor()
        return self._gazelle_extractor

    def _extract_gazelle_scene_tokens(self, gazelle_frames, video_paths=None, frame_indices=None):
        import logging
        logger = logging.getLogger(__name__)
        
        # Try to use cache if enabled and cache keys are provided
        if self._cache_enabled and video_paths is not None and frame_indices is not None:
            batch_tokens = []
            uncached_indices = []
            uncached_frames = []
            uncached_paths = []
            uncached_frame_indices = []
            
            for i, (vpath, findices) in enumerate(zip(video_paths, frame_indices)):
                # Create cache key from video path and frame indices
                if isinstance(findices, torch.Tensor):
                    findices = findices.cpu().numpy()
                cache_key = (vpath, tuple(findices.tolist()))
                
                if cache_key in self._gazelle_token_cache:
                    # Move cached tokens from CPU to GPU
                    cached_tokens = self._gazelle_token_cache[cache_key]
                    batch_tokens.append(cached_tokens.to(self.predictor_embed.weight.device, non_blocking=True))
                else:
                    uncached_indices.append(i)
                    uncached_frames.append(gazelle_frames[i])
                    uncached_paths.append(vpath)
                    uncached_frame_indices.append(findices)
            
            # Log cache hit rate
            cache_hits = len(batch_tokens)
            cache_misses = len(uncached_indices)
            if cache_hits + cache_misses > 0:
                hit_rate = cache_hits / (cache_hits + cache_misses) * 100
                logger.debug(f"[GazellePredictor] Cache hit rate: {hit_rate:.1f}% ({cache_hits}/{cache_hits + cache_misses})")
            
            # Extract tokens for uncached frames
            if uncached_frames:
                logger.debug(f"[GazellePredictor] Extracting tokens for {len(uncached_frames)} uncached clips")
                extractor = self._get_gazelle_extractor()
                for i, (clip, vpath, findices) in enumerate(zip(uncached_frames, uncached_paths, uncached_frame_indices)):
                    if isinstance(clip, torch.Tensor):
                        clip_np = clip.detach().cpu().numpy()
                    else:
                        clip_np = np.asarray(clip)
                    
                    tokens_np = extractor(clip_np)
                    tokens = torch.from_numpy(tokens_np.astype(np.float32, copy=False))
                    
                    # Store in cache on CPU to save GPU memory
                    cache_key = (vpath, tuple(findices.tolist()))
                    self._gazelle_token_cache[cache_key] = tokens.cpu()  # Cache on CPU!
                    
                    # Move to GPU for use
                    tokens_gpu = tokens.to(self.predictor_embed.weight.device, non_blocking=True)
                    batch_tokens.insert(uncached_indices[i], tokens_gpu)
                
                logger.info(f"[GazellePredictor] Token cache size: {len(self._gazelle_token_cache)} entries")
            
            # Stack all tokens
            return torch.stack(batch_tokens, dim=0) if len(batch_tokens) > 1 else batch_tokens[0].unsqueeze(0)
        
        # Fallback to non-cached extraction
        logger.debug(f"[GazellePredictor] Extracting scene tokens for batch_size={len(gazelle_frames)} (no cache)")
        extractor = self._get_gazelle_extractor()
        clips = []
        for i, clip in enumerate(gazelle_frames):
            if isinstance(clip, torch.Tensor):
                clip_np = clip.detach().cpu().numpy()
            else:
                clip_np = np.asarray(clip)
            logger.debug(f"[GazellePredictor] Processing clip {i+1}/{len(gazelle_frames)}, shape={clip_np.shape}")
            clips.append(extractor(clip_np))
        tokens_np = np.stack(clips, axis=0)
        tokens = torch.from_numpy(tokens_np.astype(np.float32, copy=False))
        logger.debug(f"[GazellePredictor] Scene tokens extracted, shape={tokens.shape}")
        return tokens.to(self.predictor_embed.weight.device, non_blocking=True)

    def forward(
        self,
        x,
        condition=None,
        condition_mode: str = "heatmap",
        gazelle_frames=None,
        video_paths=None,
        frame_indices=None,
        return_gaze=False,
    ):
        import logging
        logger = logging.getLogger(__name__)
        cond_mode = (condition_mode or "heatmap").lower()
        if condition is None and cond_mode == "gazelle":
            if gazelle_frames is None:
                raise ValueError("gazelle_frames must be provided when condition_mode='gazelle'.")
            logger.info(f"[GazellePredictor] Forward pass with gazelle mode, extracting scene tokens...")
            condition = self._extract_gazelle_scene_tokens(
                gazelle_frames, 
                video_paths=video_paths,
                frame_indices=frame_indices
            )
            logger.info(f"[GazellePredictor] Scene tokens extracted successfully, shape={condition.shape}")
        if condition is None:
            raise ValueError("Condition tensor is required for VisionTransformerPredictorGazeAC.")

        B, N, _ = x.shape
        x = self.predictor_embed(x)
        T = N // (self.grid_height * self.grid_width)
        x = x.view(B, T, self.grid_height * self.grid_width, -1)

        cond_tokens = self._prepare_condition_tokens(condition, cond_mode)
        x = x + cond_tokens
        x = x.flatten(1, 2)

        attn_mask = None
        if self.is_frame_causal:
            cache_key = (T, x.size(1))
            mask = self._attn_mask_cache.get(cache_key)
            if mask is None or mask.size(0) < x.size(1):
                mask = build_action_block_causal_attention_mask(
                    T, self.grid_height, self.grid_width, add_tokens=0
                )
                self._attn_mask_cache[cache_key] = mask
            attn_mask = mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=0,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=0,
                )

        x = x.view(B, T, self.grid_height * self.grid_width, -1)
        gaze_logits = None
        if self.enable_gaze_aux and return_gaze and cond_mode == "heatmap":
            gaze_logits = self.gaze_decoder(x).squeeze(-1)

        x = x.flatten(1, 2)
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        if gaze_logits is not None:
            return x, gaze_logits
        return x


def vit_gaze_ac_predictor(**kwargs):
    return VisionTransformerPredictorGazeAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )

