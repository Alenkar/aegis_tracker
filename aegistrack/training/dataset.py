from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import random
import re
import cv2
import torch
from torch.utils.data import Dataset

from ..config import AegisConfig
from ..utils.crop_policy import adaptive_search_crop_policy
from ..utils.image import crop_with_context, crop_to_tensor, frame_bbox_to_crop
from ..utils.box_ops import bbox_center
from .augment import tiny_uav_augment
from .blur_aug import apply_train_blur_stretch

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
JSON_LABEL_NAMES = ('IR_label.json', 'infrared_label.json', 'TIR_label.json', 'label.json', 'infrared.json', 'visible.json')
VIDEO_READ_TIMEOUT_MSEC = 5000


@dataclass(frozen=True)
class VideoFrameRef:
    video: Path
    index: int


def read_gt(path: Path):
    boxes = []
    for line in path.read_text().strip().splitlines():
        vals = [float(x) for x in re.split(r'[,\s\t]+', line.strip()) if x]
        if len(vals) >= 4:
            boxes.append(tuple(vals[:4]))
    return boxes


def read_anti_uav_json(path: Path):
    data = json.loads(path.read_text())
    boxes = data.get('gt_rect') or data.get('gt') or data.get('groundtruth_rect') or data.get('groundtruth') or data.get('bbox')
    if boxes is None:
        return []
    exists = data.get('exist') or data.get('exists') or data.get('present') or data.get('valid')
    out = []
    for i, box in enumerate(boxes):
        if exists is not None and i < len(exists) and not bool(exists[i]):
            out.append(None)
            continue
        if box is None or len(box) < 4:
            out.append(None)
            continue
        vals = tuple(float(x) for x in box[:4])
        out.append(vals if vals[2] > 0 and vals[3] > 0 else None)
    return out


def image_files(seq_dir: Path):
    img_dir = seq_dir / 'img'
    if not img_dir.exists():
        img_dir = seq_dir
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def video_frames(video_path: Path, n: int):
    return [VideoFrameRef(video_path, i) for i in range(n)]


def read_frame(ref):
    if isinstance(ref, VideoFrameRef):
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            VIDEO_READ_TIMEOUT_MSEC,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            VIDEO_READ_TIMEOUT_MSEC,
        ]
        try:
            cap = cv2.VideoCapture(str(ref.video), cv2.CAP_FFMPEG, params)
        except Exception:
            cap = cv2.VideoCapture(str(ref.video))
        try:
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, ref.index)
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()
    return cv2.imread(str(ref), cv2.IMREAD_COLOR)


def find_sequence_dirs(root: Path):
    direct = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if any((d / 'groundtruth.txt').exists() or any((d / name).exists() for name in JSON_LABEL_NAMES) for d in direct):
        return direct
    labels = list(root.rglob('groundtruth.txt'))
    for name in JSON_LABEL_NAMES:
        labels.extend(root.rglob(name))
    seen = set()
    seq_dirs = []
    for label in sorted(labels):
        d = label.parent
        if d not in seen:
            seen.add(d)
            seq_dirs.append(d)
    return seq_dirs


def load_sequence(seq_dir: Path, modalities: tuple[str, ...] = ()):
    return next(iter(load_sequences(seq_dir, modalities)), None)


def load_sequences(seq_dir: Path, modalities: tuple[str, ...] = ()):
    if modalities:
        for modality in modalities:
            json_path = seq_dir / f'{modality}.json'
            video_path = seq_dir / f'{modality}.mp4'
            if json_path.exists() and video_path.exists():
                gts = read_anti_uav_json(json_path)
                imgs = video_frames(video_path, len(gts))
                seq = build_sequence(imgs, gts, f'{seq_dir.name}:{modality}')
                if seq is not None:
                    yield seq
        return

    gt_path = seq_dir / 'groundtruth.txt'
    json_path = next((seq_dir / name for name in JSON_LABEL_NAMES if (seq_dir / name).exists()), None)
    if gt_path.exists():
        gts = read_gt(gt_path)
    elif json_path is not None:
        gts = read_anti_uav_json(json_path)
    else:
        return
    imgs = image_files(seq_dir)
    if not imgs and json_path is not None:
        video_path = json_path.with_suffix('.mp4')
        if video_path.exists():
            imgs = video_frames(video_path, len(gts))
    seq = build_sequence(imgs, gts, seq_dir.name)
    if seq is not None:
        yield seq


def build_sequence(imgs, gts, name: str):
    n = min(len(imgs), len(gts))
    pairs = [(img, gt) for img, gt in zip(imgs[:n], gts[:n]) if gt is not None]
    if len(pairs) < 2:
        return None
    seq_imgs, seq_gts = zip(*pairs)
    return list(seq_imgs), list(seq_gts), name


def bbox_inside_crop_ratio(bbox, meta) -> float:
    x, y, w, h = bbox
    x1, y1 = meta['x1'], meta['y1']
    x2, y2 = x1 + meta['side'], y1 + meta['side']
    ix1, iy1 = max(x, x1), max(y, y1)
    ix2, iy2 = min(x + w, x2), min(y + h, y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(inter / max(1e-6, w * h))


class SOTPairDataset(Dataset):
    def __init__(self, root: str | list, cfg: AegisConfig, pairs_per_epoch: int = 20000, max_gap: int = 100, augment: bool = True):
        self.cfg = cfg
        self.pairs_per_epoch = pairs_per_epoch
        self.max_gap = max_gap
        self.augment = augment
        self.training = bool(augment)
        self.seqs = []
        roots = root if isinstance(root, list) else [root]
        for item in roots:
            if isinstance(item, dict):
                data_root = Path(item['path'])
                modalities = tuple(item.get('modalities') or ())
            else:
                data_root = Path(item)
                modalities = ()
            for d in find_sequence_dirs(data_root):
                for seq in load_sequences(d, modalities):
                    self.seqs.append(seq)
        if not self.seqs:
            raise RuntimeError(f'No SOT sequences found in {roots}')

    def __len__(self):
        return self.pairs_per_epoch

    def _sample_pair_ids(self, n: int):
        i = random.randrange(0, n - 1)
        # Bias toward short gaps; long gaps are still sampled sometimes.
        max_gap = min(self.max_gap, n - i - 1)
        if max_gap <= 1:
            return i, i + 1
        if random.random() < 0.75:
            gap = random.randint(1, min(5, max_gap))
        else:
            gap = random.randint(1, max_gap)
        return i, i + gap

    def __getitem__(self, idx):
        for _ in range(20):
            imgs, gts, name = random.choice(self.seqs)
            n = len(imgs)
            i, j = self._sample_pair_ids(n)
            z = read_frame(imgs[i])
            x = read_frame(imgs[j])
            if z is None or x is None:
                continue

            bz, bx = gts[i], gts[j]
            if self.augment:
                z = tiny_uav_augment(z, blur=False)
                x = tiny_uav_augment(x, blur=False)

            max_obj = max(float(bz[2]), float(bz[3]), float(bx[2]), float(bx[3]))
            z_side = max(max(bz[2], bz[3]) * 4.0 + 16.0, float(self.cfg.template_size) * 0.5)

            gt_cx, gt_cy = bbox_center(bx)
            H, W = x.shape[:2]
            crop_policy = adaptive_search_crop_policy((bx[0], bx[1], max_obj, max_obj), W, H, self.cfg)
            base_side = crop_policy.crop_side
            scale_jitter = float(getattr(self.cfg, 'train_scale_jitter', 0.25))
            if scale_jitter > 0:
                x_side = base_side * random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
            else:
                x_side = base_side
            frame_cap = 1.25 * max(float(W), float(H))
            max_crop_side = min(float(getattr(self.cfg, 'max_crop_side', frame_cap)), frame_cap)
            min_crop_side = min(float(getattr(self.cfg, 'min_crop_side', 144.0)), max_crop_side)
            x_side = min(max(x_side, min_crop_side), max_crop_side)
            search_factor = x_side / max(max_obj, 1e-6)
            object_in_search_px = max_obj * float(self.cfg.search_size) / max(x_side, 1e-6)
            if getattr(self.cfg, 'debug', False):
                print(
                    'train_crop_policy '
                    f'crop_side={x_side:.2f} '
                    f'search_factor={search_factor:.2f} '
                    f'object_in_search_px={object_in_search_px:.2f}',
                    flush=True,
                )

            # Train the local core on a crop where the target is visible, but not always centered.
            # The previous implementation centered the search crop on the template-frame target center;
            # for UAV sequences this often made the object extremely tiny/off-center and the center head
            # learned a near-static prior instead of localizing the current target.
            jitter_abs = max(max_obj * self.cfg.train_motion_jitter, x_side * self.cfg.train_search_jitter)
            search_center = (
                gt_cx + random.uniform(-jitter_abs, jitter_abs),
                gt_cy + random.uniform(-jitter_abs, jitter_abs),
            )

            z_crop, _ = crop_with_context(z, bz, z_side)
            x_crop, meta = crop_with_context(x, search_center, x_side)
            if bbox_inside_crop_ratio(bx, meta) < self.cfg.train_min_visible_iou:
                search_center = (gt_cx, gt_cy)
                x_crop, meta = crop_with_context(x, search_center, x_side)

            gt_crop = frame_bbox_to_crop(bx, meta, self.cfg.search_size)
            x_crop = cv2.resize(x_crop, (self.cfg.search_size, self.cfg.search_size), interpolation=cv2.INTER_LINEAR)
            if self.training:
                x_crop, gt_crop = apply_train_blur_stretch(x_crop, gt_crop, self.cfg)

            template = crop_to_tensor(z_crop, self.cfg.template_size, 'cpu')[0]
            search = crop_to_tensor(x_crop, self.cfg.search_size, 'cpu')[0]
            return {
                'template': template,
                'search': search,
                'gt_bbox': torch.tensor(gt_crop, dtype=torch.float32),
                'seq': name,
                'crop_side': torch.tensor(float(x_side), dtype=torch.float32),
                'search_factor': torch.tensor(float(search_factor), dtype=torch.float32),
                'object_in_search_px': torch.tensor(float(object_in_search_px), dtype=torch.float32),
            }
        raise RuntimeError('Failed to sample a valid SOT pair')
