from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Tuple
import torch


@dataclass
class ClearMLConfig:
    enabled: bool = False
    project_name: str = "AegisTrack"
    task_name: str = ""
    queue_name: str = "default"
    remote: bool = False
    dataset_id: str = ""


@dataclass
class TensorBoardConfig:
    enabled: bool = True
    log_dir: str = ""


@dataclass
class TrainConfig:
    data: str = ""
    out: str = "runs/train"
    epochs: int = 50
    batch: int = 8
    lr: float = 1e-4
    pairs_per_epoch: int = 20000
    max_gap: int = 20
    num_workers: int = 4
    dataloader_timeout: int = 60
    ckpt: str = ""


@dataclass
class ValidationConfig:
    enabled: bool = True
    data: str = ""
    every: int = 1
    max_sequences: int = 5
    max_frames_per_sequence: int = 300


@dataclass
class DatasetSourceConfig:
    enabled: bool = True
    path: str = ""
    modalities: Tuple[str, ...] = ()


@dataclass
class DatasetsConfig:
    root: str = "/home/neuro/dataset/UAV-TRACKING"
    sources: dict[str, DatasetSourceConfig] = field(default_factory=lambda: {
        "anti_uav_rgbt": DatasetSourceConfig(
            enabled=True,
            path="Anti-UAV-RGBT",
            modalities=("infrared", "visible"),
        ),
        "anti_uav410": DatasetSourceConfig(enabled=True, path="Anti-UAV410"),
        "cst_anti_uav": DatasetSourceConfig(enabled=True, path="CST-AntiUAV"),
        "anti_uav_4th": DatasetSourceConfig(enabled=True, path="The_4th_Anti-UAV_Dataset_Test"),
    })


@dataclass
class AegisConfig:
    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    profile: str = "tiny_uav"
    debug: bool = False

    # Images
    template_size: int = 128
    search_size: int = 256
    patch_size: int = 16  # legacy/ViT patch; localization uses local_feature_stride
    local_feature_stride: int = 4
    local_feature_dim: int = 128
    in_chans: int = 3

    # Backbone
    embed_dim: int = 192
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # Search / candidates
    topk_tracking: int = 5
    topk_lost: int = 12

    # Runtime top-K selection. Center-distance NMS is more stable than IoU NMS
    # for tiny objects because 1-2 px shifts can destroy IoU.
    center_nms_radius_px: float = 10.0
    center_nms_max_keep_tracking: int = 5
    center_nms_max_keep_lost: int = 8
    tracking_score_shape_weight: float = 0.70
    tracking_score_motion_weight: float = 0.30
    recovery_score_shape_weight: float = 0.35
    recovery_score_identity_weight: float = 0.35
    recovery_score_size_weight: float = 0.15
    recovery_score_objectness_quality_weight: float = 0.15
    recovery_min_shape_score: float = 0.45
    recovery_min_identity_score: float = 0.10
    recovery_min_size_prior_score: float = 0.35

    # Adaptive FOV
    base_search_factor: float = 18.0
    adaptive_search_min_factor: float = 2.5
    adaptive_search_scale_ref: float = 32.0
    min_crop_side: int = 144
    max_crop_side: int = 768
    crop_lost_gain: float = 1.90

    # Response windowing
    window_weight_tracking: float = 0.15
    window_weight_lost: float = 0.0

    # BBox stability
    stable_size_lr_tiny: float = 0.03
    stable_size_lr_generic: float = 0.10
    growth_bad_high_tiny: float = 1.8
    growth_bad_low_tiny: float = 0.55
    growth_bad_high_generic: float = 2.8
    growth_bad_low_generic: float = 0.35
    growth_full_bad: float = 3.0
    pred_norm_bad: float = 0.35
    bbox_decode_center_window: int = 3
    bbox_decode_size_window: int = 3

    # Gates
    min_raw_score: float = 0.25

    # Training
    center_sigma_factor: float = 0.15
    center_sigma_min: float = 1.5
    tiny_center_positive_px: float = 4.0
    candidate_positive_iou: float = 0.3
    rank_margin: float = 0.2

    # Pair sampling / train target visibility
    train_search_jitter: float = 0.35
    train_motion_jitter: float = 0.15
    train_min_visible_iou: float = 0.98
    train_scale_jitter: float = 0.25
    blur_aug_enabled: bool = True
    blur_aug_prob: float = 0.15
    motion_blur_prob: float = 0.75
    motion_blur_kernel_min: int = 3
    motion_blur_kernel_max: int = 11
    blur_horizontal_prob: float = 0.35
    blur_vertical_prob: float = 0.35
    blur_diagonal_prob: float = 0.20
    blur_angle_jitter_deg: float = 10.0
    blur_bbox_expand_parallel_factor: float = 0.30
    blur_bbox_expand_parallel_max_px: float = 6.0
    blur_bbox_expand_perp_px: float = 1.0
    blur_aug_max_bbox_expand_ratio: float = 1.6

    # UETrack-like runtime state
    uetrack_trust_thr: float = 0.45
    tracking_score_thr: float = 0.45
    recovery_score_thr: float = 0.45
    uetrack_lost_frames: int = 3
    uetrack_recover_frames: int = 2


    # Core matching / attention-lite
    template_kernel_size: int = 5
    target_gate_min: float = 0.25
    size_pred_blend_tiny: float = 0.15
    size_pred_blend_generic: float = 0.35

    # Local core correlation prior
    corr_response_weight: float = 0.35

    # ClearML
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)

    def topk_for_state(self, state: str) -> int:
        if state == "TRACKING":
            return self.topk_tracking
        return self.topk_lost

    @property
    def feat_hw(self) -> int:
        return self.search_size // self.local_feature_stride

    @property
    def feature_stride(self) -> int:
        return self.local_feature_stride

    @property
    def template_hw(self) -> int:
        return self.template_size // self.patch_size

    def growth_limits(self):
        if self.profile == "tiny_uav":
            return self.growth_bad_low_tiny, self.growth_bad_high_tiny
        return self.growth_bad_low_generic, self.growth_bad_high_generic

    def to_dict(self) -> dict:
        return asdict(self)


def config_to_dict(cfg: AegisConfig) -> dict:
    return cfg.to_dict()


def enabled_dataset_entries(cfg: AegisConfig, split: str) -> list[dict]:
    root = Path(cfg.datasets.root)
    entries = []
    for source in cfg.datasets.sources.values():
        if not source.enabled:
            continue
        source_path = Path(source.path)
        if not source_path.is_absolute():
            source_path = root / source_path
        split_path = source_path / split
        entries.append({
            "path": str(split_path if split_path.exists() else source_path),
            "modalities": list(source.modalities),
        })
    return entries


def load_aegis_config(path: str | Path | None = None, **overrides: Any) -> AegisConfig:
    cfg = AegisConfig()
    if path:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            import yaml

            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            _set_config_value(cfg, key, value)
    for key, value in overrides.items():
        if value is not None:
            _set_config_value(cfg, key, value)
    return cfg


def _set_config_value(cfg: AegisConfig, key: str, value: Any) -> None:
    if key == "clearml":
        for clearml_key, clearml_value in (value or {}).items():
            if hasattr(cfg.clearml, clearml_key):
                setattr(cfg.clearml, clearml_key, clearml_value)
        return
    if key == "tensorboard":
        for tb_key, tb_value in (value or {}).items():
            if hasattr(cfg.tensorboard, tb_key):
                setattr(cfg.tensorboard, tb_key, tb_value)
        return
    if key == "train":
        for train_key, train_value in (value or {}).items():
            if hasattr(cfg.train, train_key):
                setattr(cfg.train, train_key, train_value)
        return
    if key == "validation":
        for val_key, val_value in (value or {}).items():
            if hasattr(cfg.validation, val_key):
                setattr(cfg.validation, val_key, val_value)
        return
    if key == "datasets":
        datasets_value = value or {}
        if "root" in datasets_value:
            cfg.datasets.root = datasets_value["root"]
        for source_key, source_value in (datasets_value.get("sources") or {}).items():
            current = cfg.datasets.sources.get(source_key, DatasetSourceConfig(path=source_key))
            for source_attr, source_attr_value in (source_value or {}).items():
                if hasattr(current, source_attr):
                    if source_attr == "modalities" and isinstance(source_attr_value, list):
                        source_attr_value = tuple(str(v) for v in source_attr_value)
                    setattr(current, source_attr, source_attr_value)
            cfg.datasets.sources[source_key] = current
        return
    if not hasattr(cfg, key):
        return
    setattr(cfg, key, value)
