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
    use_learned_runtime_heads: bool = False

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
    topk_uncertain: int = 8
    topk_lost: int = 12
    max_candidates: int = 96
    nms_iou: float = 0.65

    # Runtime top-K selection. Center-distance NMS is more stable than IoU NMS
    # for tiny objects because 1-2 px shifts can destroy IoU.
    center_nms_radius_px: float = 10.0
    center_nms_max_keep_tracking: int = 5
    center_nms_max_keep_lost: int = 8
    tracking_candidate_score_response_weight: float = 0.65
    tracking_candidate_score_motion_weight: float = 0.35

    # Adaptive FOV
    base_search_factor: float = 18.0
    adaptive_search_min_factor: float = 2.5
    adaptive_search_scale_ref: float = 32.0
    min_crop_side: int = 144
    max_crop_side: int = 768
    crop_uncertain_gain: float = 1.35
    crop_verify_gain: float = 1.65
    crop_lost_gain: float = 1.90

    # Score fusion
    alpha_center: float = 1.0
    beta_objectness: float = 0.5
    gamma_quality: float = 0.5
    window_weight_tracking: float = 0.15
    window_weight_uncertain: float = 0.05
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
    min_objectness: float = 0.25
    min_track_score: float = 0.35
    min_identity_score: float = 0.35
    min_recovery_score: float = 0.55
    max_negative_score: float = 0.80
    max_distractor_score: float = 0.75
    min_update_score: float = 0.55
    min_present_score: float = 0.55
    max_update_risk: float = 0.35
    low_present_for_lost_frames: int = 2
    quarantine_confirm_frames: int = 3

    # Motion / egomotion
    egomotion_enabled: bool = True
    motion_sigma0: float = 12.0
    motion_sigma_vel_gain: float = 0.25
    motion_sigma_unc_gain: float = 10.0
    impossible_jump_factor: float = 8.0

    # Memory
    stable_memory_size: int = 4
    recent_memory_size: int = 8
    negative_memory_size: int = 96
    distractor_memory_size: int = 48
    quarantine_memory_size: int = 8
    memory_ema: float = 0.95

    # Temporal encoder
    temporal_input_dim: int = 32
    temporal_hidden_dim: int = 96
    temporal_history: int = 32

    # Decision/presence feature dims
    candidate_feature_dim: int = 64
    decision_hidden_dim: int = 128

    # Target-instance selector / anti-switch
    use_target_instance_selector: bool = True
    selector_raw_weight: float = 0.42
    selector_motion_weight: float = 0.22
    selector_identity_weight: float = 0.18
    selector_size_weight: float = 0.10
    selector_quality_weight: float = 0.08
    selector_negative_weight: float = 0.25
    selector_distractor_weight: float = 0.25
    selector_jump_penalty: float = 0.35
    selector_near_factor: float = 4.0
    selector_near_min_px: float = 20.0
    selector_far_raw_advantage: float = 0.15
    selector_ambiguity_margin: float = 0.07
    selector_min_score: float = 0.22
    selector_lost_min_score: float = 0.18


    # Aegis Basic runtime: local_safe is the primary production path.
    default_runtime_mode: str = "local_safe"
    local_good_score: float = 0.42
    local_weak_score: float = 0.24
    local_lost_score: float = 0.14
    local_recovery_multipliers: Tuple[float, ...] = (1.0, 1.45, 2.10)
    local_lost_multipliers: Tuple[float, ...] = (1.45, 2.10, 2.80)

    # Conservative dynamic template. Initial template remains immutable.
    dynamic_template_enabled: bool = True
    dynamic_template_lr_tiny: float = 0.015
    dynamic_template_lr_generic: float = 0.04
    dynamic_template_mix_tiny: float = 0.20
    dynamic_template_mix_generic: float = 0.35
    dynamic_template_score_thr: float = 0.45
    dynamic_template_max_jump_factor: float = 3.0

    # Runtime cost control
    recovery_in_tracking: bool = False
    encode_recovery_candidates: bool = True

    # Training auxiliary loss weights
    decision_loss_weight: float = 0.05
    ranker_loss_weight: float = 0.20
    presence_loss_weight: float = 0.03
    update_recovery_loss_weight: float = 0.02

    # Recovery
    enable_log_dog_recovery: bool = True
    enable_tile_recovery: bool = True
    enable_detector_proposals: bool = True
    tile_grid: Tuple[int, int] = (3, 3)
    log_sigmas: Tuple[float, ...] = (1.0, 1.6, 2.4, 3.2)
    max_log_candidates: int = 24
    max_tile_candidates: int = 18
    max_detector_candidates: int = 12

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

    # UETrack-like runtime state
    uetrack_trust_thr: float = 0.45
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
        if state in ("UNCERTAIN", "VERIFYING", "REACQUIRED"):
            return self.topk_uncertain
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
    if key == "tile_grid" and isinstance(value, list):
        value = tuple(int(v) for v in value)
    elif key == "log_sigmas" and isinstance(value, list):
        value = tuple(float(v) for v in value)
    setattr(cfg, key, value)
