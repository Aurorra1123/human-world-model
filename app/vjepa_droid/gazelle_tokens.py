import functools
import logging
import os
import sys
import types
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_GAZELLE_REPO = "/data3/lg2/human_wm/gazelle"
logger = logging.getLogger(__name__)


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot  # noqa: F401
        return
    except ImportError:
        pass

    pyplot = types.ModuleType("matplotlib.pyplot")

    class _SimpleCM:
        @staticmethod
        def jet(arr):
            data = np.asarray(arr, dtype=np.float32)
            data = np.clip(data, 0.0, 1.0)
            rgba = np.zeros(data.shape + (4,), dtype=np.float32)
            rgba[..., 0] = data
            rgba[..., 1] = data
            rgba[..., 2] = data
            rgba[..., 3] = 1.0
            return rgba

    pyplot.cm = _SimpleCM()

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot

    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)


def _ensure_sklearn_metrics():
    try:
        from sklearn.metrics import roc_auc_score  # noqa: F401
        return
    except ImportError:
        pass

    def roc_auc_score(y_true, y_score):
        y_true_arr = np.asarray(y_true).ravel()
        y_score_arr = np.asarray(y_score).ravel()

        if y_true_arr.size != y_score_arr.size:
            raise ValueError("y_true and y_score must be the same size")

        pos_mask = y_true_arr > 0
        neg_mask = ~pos_mask

        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        if n_pos == 0 or n_neg == 0:
            # Undefined AUC; fall back to 0.5
            return 0.5

        order = np.argsort(-y_score_arr)
        y_true_sorted = y_true_arr[order]

        tpr = np.cumsum(y_true_sorted) / n_pos
        fpr = np.cumsum(1 - y_true_sorted) / n_neg
        return np.trapz(tpr, fpr)

    metrics_module = types.ModuleType("sklearn.metrics")
    metrics_module.roc_auc_score = roc_auc_score

    sklearn_module = types.ModuleType("sklearn")
    sklearn_module.metrics = metrics_module

    sys.modules.setdefault("sklearn", sklearn_module)
    sys.modules.setdefault("sklearn.metrics", metrics_module)


class GazelleSceneTokenExtractor:
    """
    Thin wrapper around Gazelle (Gaze-LLE) to expose the intermediate scene tokens.

    Returns tokens of shape [T, 1024, dim] where dim depends on the Gazelle backbone (default 256).
    """

    def __init__(
        self,
        checkpoint: str,
        model_name: str = "gazelle_dinov2_vitb14_inout",
        device: str = "cuda",
        max_batch_size: int = 32,
        module_path: Optional[str] = None,
        min_cuda_free_mb: int = 2048,
    ):
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.device_name = device
        self.max_batch_size = max_batch_size
        self.module_path = module_path
        self.min_cuda_free_bytes = max(int(min_cuda_free_mb), 1) * 1024 * 1024

        self._model = None
        self._transform = None
        self._device = None
        self._utils = None
        self._scene_grid = None
        self._scene_dim = None

    def _lazy_init(self):
        if self._model is not None:
            return
        _ensure_matplotlib()
        _ensure_sklearn_metrics()
        self._ensure_module_path()
        try:
            from gazelle.model import get_gazelle_model
            import gazelle.utils as gazelle_utils
        except ImportError as exc:
            raise ImportError(
                "Unable to import gazelle. Set `data.gazelle.python_path` in your config or export "
                "`GAZELLE_REPO` so that /path/to/gazelle is on PYTHONPATH."
            ) from exc

        device = self._select_device(self.device_name)
        model, transform = get_gazelle_model(self.model_name)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        model.load_gazelle_state_dict(state)
        self._model = model
        self._move_model(device)

        self._transform = transform
        self._utils = gazelle_utils
        self._scene_grid = model.featmap_h, model.featmap_w
        self._scene_dim = model.dim

    def _select_device(self, preferred: str) -> torch.device:
        if preferred is None:
            preferred = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            device = torch.device(preferred)
        except (TypeError, RuntimeError):
            return torch.device("cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if device.type == "cuda" and not self._has_sufficient_cuda_mem(device):
            logger.warning(
                "Gazelle requested %s but available free memory is below %d MiB. Falling back to CPU.",
                device,
                self.min_cuda_free_bytes // (1024 * 1024),
            )
            return torch.device("cpu")
        return device

    def _has_sufficient_cuda_mem(self, device: torch.device) -> bool:
        if device.type != "cuda":
            return False
        index = device.index if device.index is not None else torch.cuda.current_device()
        try:
            with torch.cuda.device(index):
                free_bytes, _ = torch.cuda.mem_get_info()
        except RuntimeError:
            return False
        return free_bytes >= self.min_cuda_free_bytes

    def _move_model(self, device: torch.device):
        if device.type.startswith("cuda") and not torch.cuda.is_available():
            device = torch.device("cpu")
        try:
            self._model.to(device)
            self._device = device
        except RuntimeError as exc:  # pragma: no cover - best effort fallback
            if device.type == "cuda" and "out of memory" in str(exc).lower():
                logger.warning(
                    "Gazelle model OOM on %s. Falling back to CPU. (%s)", device, exc
                )
                torch.cuda.empty_cache()
                self._model.to(torch.device("cpu"))
                self._device = torch.device("cpu")
            else:
                raise
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

    def _ensure_module_path(self):
        if "gazelle.model" in sys.modules:
            return

        candidates = []
        if self.module_path:
            candidates.append(self.module_path)
        env_path = os.environ.get("GAZELLE_REPO") or os.environ.get("GAZELLE_PYTHONPATH")
        if env_path:
            candidates.append(env_path)
        candidates.append(DEFAULT_GAZELLE_REPO)

        for path in candidates:
            if not path:
                continue
            expanded = os.path.abspath(os.path.expanduser(path))
            if not os.path.isdir(expanded):
                continue
            if expanded not in sys.path:
                sys.path.insert(0, expanded)
            return
        # If no valid path found we still attempt import which will raise a helpful error later.

    @property
    def scene_dim(self) -> int:
        self._lazy_init()
        return self._scene_dim

    @property
    def scene_grid(self):
        self._lazy_init()
        return self._scene_grid

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Args:
            frames: numpy array shaped [T, H, W, 3] in RGB uint8.

        Returns:
            tokens: numpy array shaped [T, Hc*Wc, dim]
        """
        self._lazy_init()
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected frames in shape [T, H, W, 3], got {frames.shape}")

        batches = []
        start = 0
        while start < len(frames):
            end = min(len(frames), start + self.max_batch_size)
            tensors = [self._transform(frame) for frame in frames[start:end]]
            batch = torch.stack(tensors, dim=0).to(
                self._device, non_blocking=self._device.type == "cuda"
            )
            features = self._run_forward(batch)
            batches.append(features.cpu())
            start = end

        tokens = torch.cat(batches, dim=0).cpu().detach().numpy()
        return tokens

    def _run_forward(self, batch: torch.Tensor) -> torch.Tensor:
        try:
            with torch.no_grad():
                return self._forward_scene_tokens(batch)
        except RuntimeError as exc:  # pragma: no cover - fallback path
            if self._device.type == "cuda" and "out of memory" in str(exc).lower():
                logger.warning(
                    "Gazelle forward OOM on %s with batch size %d. Switching to CPU. (%s)",
                    self._device,
                    batch.size(0),
                    exc,
                )
                torch.cuda.empty_cache()
                self._move_model(torch.device("cpu"))
                batch_cpu = batch.to(self._device)
                with torch.no_grad():
                    return self._forward_scene_tokens(batch_cpu)
            raise

    def _forward_scene_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Mirrors Gazelle forward pass up to transformer output (before heatmap head),
        assuming single-person scenes (one dummy bbox per frame).
        Returns shape [T, grid_h * grid_w, dim].
        """
        model = self._model
        utils = self._utils
        num_people = [1] * images.size(0)

        x = model.backbone.forward(images)
        x = model.linear(x)
        x = x + model.pos_embed
        x = utils.repeat_tensors(x, num_people)

        # Single person per frame → bbox None
        head_maps = torch.cat(model.get_input_head_maps([[None]] * images.size(0)), dim=0).to(x.device)
        head_map_embed = head_maps.unsqueeze(1) * model.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embed
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if model.inout:
            inout_tokens = model.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = torch.cat([inout_tokens, x], dim=1)

        x = model.transformer(x)
        if model.inout:
            x = x[:, 1:, :]

        return x


@functools.lru_cache(maxsize=4)
def get_gazelle_extractor(
    checkpoint: str,
    model_name: str = "gazelle_dinov2_vitb14_inout",
    device: str = "cuda",
    max_batch_size: int = 32,
    module_path: Optional[str] = None,
    min_cuda_free_mb: int = 2048,
) -> GazelleSceneTokenExtractor:
    return GazelleSceneTokenExtractor(
        checkpoint=checkpoint,
        model_name=model_name,
        device=device,
        max_batch_size=max_batch_size,
        module_path=module_path,
        min_cuda_free_mb=min_cuda_free_mb,
    )

